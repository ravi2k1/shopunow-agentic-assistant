import os
from typing import Annotated, Literal, TypedDict
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# Configuration
PERSIST_DIRECTORY = "./chroma_db"
COLLECTION_NAME = "shopunow_faqs"
ALLOWED_DEPARTMENTS = ["HR", "IT_Support", "Billing", "Shipping", "Conversational"]

# Initialize Vector Store & LLMs
embeddings = OpenAIEmbeddings()
vector_store = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings
)
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Build Sparse BM25 Index from Ingested Corpus
raw_data = vector_store.get()
bm25_retriever = BM25Retriever.from_texts(
    texts=raw_data["documents"],
    metadatas=raw_data["metadatas"]
)
bm25_retriever.k = 5


# 1. Classification Schema
class QueryClassification(BaseModel):
    sentiment: Literal["positive", "neutral", "negative"]
    department: Literal["HR", "IT_Support", "Billing", "Shipping", "Conversational", "Other"]
    reasoning: str


classifier = llm.with_structured_output(QueryClassification)


# 2. LangGraph State Schema
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    department: str
    sentiment: str
    standalone_query: str
    context: str
    escalation_needed: bool
    escalation_reason: str


# Node 1: Intent & Sentiment Classification
# In agent.py inside classify_node:

def classify_node(state: AgentState):
    system_prompt = (
        "Classify the user query into a department:\n"
        "- HR (leave, payroll, performance, policies)\n"
        "- IT_SUPPORT (laptops, hardware, software, digital access)\n"
        "- Billing (invoices, refunds, payment methods)\n"
        "- Shipping (order tracking, delivery delays, returns)\n"
        "- Conversational (greetings, thanks, small talk, OR meta-questions about chat history like 'what was my last query?', 'what is my name?')\n"
        "- Other (out-of-scope enterprise queries)\n\n"
        "SENTIMENT GUIDELINES:\n"
        "- NEUTRAL / POSITIVE: Questions, problem statements, and conversational chatter.\n"
        "- NEGATIVE: Extreme anger, hostile complaints, or severe frustration."
    )
    last_msg = state["messages"][-1].content
    analysis = classifier.invoke(f"{system_prompt}\n\nUser Query: {last_msg}")

    dept = analysis.department

    escalate = False
    reason = ""

    if analysis.sentiment == "negative":
        escalate = True
        reason = "Negative sentiment detected."
    elif dept == "Other" or dept not in ALLOWED_DEPARTMENTS:
        escalate = True
        reason = f"Department '{dept}' is not handled by self-service."

    return {
        "department": dept,
        "sentiment": analysis.sentiment,
        "escalation_needed": escalate,
        "escalation_reason": reason
    }


# Node 2: Contextual Query Condenser
def condense_node(state: AgentState):
    if state.get("escalation_needed", False):
        return {"standalone_query": state["messages"][-1].content}

    latest_msg = state["messages"][-1].content

    if len(state["messages"]) > 1:
        condense_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "Given the conversation history and follow-up question, rewrite the question into "
                "a standalone search query for a knowledge base.\n"
                "CRITICAL: Strip out personal questions (e.g., 'What is my name?') so the query "
                "strictly focuses on organizational policy/technical topics."
            ),
            MessagesPlaceholder(variable_name="messages"),
            ("user", "{question}")
        ])

        condenser_chain = condense_prompt | llm
        condensed = condenser_chain.invoke({
            "messages": state["messages"][:-1],
            "question": latest_msg
        }).content
    else:
        condensed = latest_msg

    return {"standalone_query": condensed}


# Node 3: Hybrid Retrieval (BM25 + Dense Vector Search)
def hybrid_retrieve_node(state: AgentState):
    if state.get("escalation_needed", False):
        return {"context": ""}

    dept = state["department"]

    # Bypass Vector DB search for pleasantries / greetings
    if dept == "Conversational":
        return {"context": "CONVERSATIONAL_RESPONSE"}

    # Exact metadata match against JSON department tag
    dense_retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,
            "filter": {"department": dept}
        }
    )

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever],
        weights=[0.5, 0.5]
    )

    raw_docs = ensemble_retriever.invoke(state["standalone_query"])

    # Filter ensemble output documents strictly matching current department
    dept_docs = [
        doc for doc in raw_docs
        if doc.metadata.get("department") == dept
    ]

    if not dept_docs:
        return {"context": "NO_DOCS_FOUND"}

    context = "\n\n".join([doc.page_content for doc in dept_docs])
    return {"context": context}


# Node 4: Answer Generation
def generate_node(state: AgentState):
    if state.get("escalation_needed", False):
        return {
            "messages": [AIMessage(content=f"Escalating request due to: {state['escalation_reason']}. Ticket logged.")]
        }

    # Handle polite chatter / thank yous gracefully
    if state.get("context") == "CONVERSATIONAL_RESPONSE":
        chat_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a helpful ShopUNow customer support assistant.\n"
                "1. Respond politely to greetings or closing remarks.\n"
                "2. If the user asks about previous messages (e.g., 'What was my last query?'), answer accurately using the conversation history."
            ),
            MessagesPlaceholder(variable_name="messages"),
        ])
        chain = chat_prompt | llm
        response = chain.invoke({"messages": state["messages"]})
        return {"messages": [response]}

    if state.get("context") == "NO_DOCS_FOUND" or not state.get("context"):
        response_text = "I couldn't find specific documentation on that topic in our knowledge base. Please try rephrasing your question or contact support directly."
        return {"messages": [AIMessage(content=response_text)]}

    qa_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an assistant for ShopUNow.\n"
            "1. Answer personal/conversational queries (e.g., user name) using thread history.\n"
            "2. Answer policy/procedural queries using the retrieved context.\n\n"
            "Retrieved Context:\n{context}"
        ),
        MessagesPlaceholder(variable_name="messages"),
    ])

    chain = qa_prompt | llm
    response = chain.invoke({"context": state["context"], "messages": state["messages"]})
    return {"messages": [response]}


# Build Graph Workflow
workflow = StateGraph(AgentState)
workflow.add_node("classify", classify_node)
workflow.add_node("condense", condense_node)
workflow.add_node("retrieve", hybrid_retrieve_node)
workflow.add_node("generate", generate_node)

workflow.add_edge(START, "classify")
workflow.add_edge("classify", "condense")
workflow.add_edge("condense", "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

app = workflow.compile(checkpointer=MemorySaver())


def process_user_query(query: str, thread_id: str = "default_thread") -> str:
    config = {"configurable": {"thread_id": thread_id}}
    input_state = {
        "messages": [HumanMessage(content=query)],
        "department": "",
        "sentiment": "neutral",
        "standalone_query": query,
        "context": "",
        "escalation_needed": False,
        "escalation_reason": ""
    }
    result = app.invoke(input_state, config=config)
    return result["messages"][-1].content


if __name__ == "__main__":
    thread_id = "architect_test_101"

    # Turn 1
    q1 = "my laptop is running slow?"
    print(f"\n[Turn 1] USER: {q1}\n[Turn 1] AGENT:\n{process_user_query(q1, thread_id=thread_id)}")

    # Turn 2
    q2 = "Can you remind me what my name is and how many hours my manager has to review it?"
    print(f"\n[Turn 2] USER: {q2}\n[Turn 2] AGENT:\n{process_user_query(q2, thread_id=thread_id)}")

    # Turn 3
    q3 = "Thank you!"
    print(f"\n[Turn 3] USER: {q3}\n[Turn 3] AGENT:\n{process_user_query(q3, thread_id=thread_id)}")