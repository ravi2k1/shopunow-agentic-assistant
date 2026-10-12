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
ALLOWED_DEPARTMENTS = ["HR", "IT Support", "Billing & Payments", "Shipping & Delivery"]

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

# Classification Schema
class QueryClassification(BaseModel):
    sentiment: Literal["positive", "neutral", "negative"]
    department: Literal["HR", "IT Support", "Facilities & Admin", "Billing & Payments", "Shipping & Delivery", "Other"]
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
def classify_node(state: AgentState):
    system_prompt = (
        "Classify the user query into a department (HR, IT Support, Facilities & Admin, "
        "Billing & Payments, Shipping & Delivery, Other) and evaluate sentiment (positive, neutral, negative)."
    )
    last_msg = state["messages"][-1].content
    analysis = classifier.invoke(f"{system_prompt}\n\nUser Query: {last_msg}")

    escalate = analysis.sentiment == "negative" or analysis.department not in ALLOWED_DEPARTMENTS
    reason = "Negative sentiment" if analysis.sentiment == "negative" else f"Department {analysis.department} unhandled"

    return {
        "department": analysis.department,
        "sentiment": analysis.sentiment,
        "escalation_needed": escalate,
        "escalation_reason": reason
    }

# Node 2: Contextual Query Condenser (Anaphora Resolution)
def condense_node(state: AgentState):
    if state["escalation_needed"]:
        return {"standalone_query": state["messages"][-1].content}

    condense_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Given the chat history and follow-up question, rewrite the follow-up question into "
            "a clear, standalone domain search query for a knowledge base.\n"
            "CRITICAL: Remove all user-chat conversational fluff (e.g., 'What is my name?') "
            "so the query strictly targets organizational policy/subject matter."
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("user", "{question}")
    ])
    
    condenser_chain = condense_prompt | llm
    latest_msg = state["messages"][-1].content

    if len(state["messages"]) > 1:
        condensed = condenser_chain.invoke({
            "messages": state["messages"][:-1],
            "question": latest_msg
        }).content
        print(f"\n[Query Condenser]: '{latest_msg}' -> Standalone Search: '{condensed}'")
    else:
        condensed = latest_msg

    return {"standalone_query": condensed}

# Node 3: Hybrid Retrieval (BM25 + Dense Vector Search)
def hybrid_retrieve_node(state: AgentState):
    if state["escalation_needed"]:
        return {"context": ""}

    # Dense Retriever with Department Filter
    dense_retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,
            "filter": {"department": state["department"]}
        }
    )

    # Hybrid Ensemble (0.5 BM25 + 0.5 Dense Vector)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever],
        weights=[0.5, 0.5]
    )

    docs = ensemble_retriever.invoke(state["standalone_query"])

    if not docs:
        return {"context": "NO_DOCS_FOUND"}

    # In production: Apply Cross-Encoder Reranker here on top 'docs' candidate pool
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"context": context}

# Node 4: Answer Generation
def generate_node(state: AgentState):
    if state["escalation_needed"]:
        return {"messages": [AIMessage(content=f"Escalating request due to: {state['escalation_reason']}. Ticket logged.")]}

    if state["context"] == "NO_DOCS_FOUND":
        return {"messages": [AIMessage(content="I could not find relevant internal documentation.")]}

    qa_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an assistant for ShopUNow.\n"
            "1. Answer personal/conversational queries (e.g., user name) using thread history.\n"
            "2. Answer policy/procedural queries using the retrieved context.\n"
            "3. State exact figures, hours, and days when available in the context.\n\n"
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
    result = app.invoke({"messages": [HumanMessage(content=query)]}, config=config)
    return result["messages"][-1].content

if __name__ == "__main__":
    thread_id = "architect_test_101"

    # Turn 1
    q1 = "Hi, my name is Alex and I work in sales. How do I apply for paid time off (PTO)?"
    print(f"\n[Turn 1] USER: {q1}\n[Turn 1] AGENT:\n{process_user_query(q1, thread_id=thread_id)}")

    # Turn 2
    q2 = "Can you remind me what my name is and how many hours my manager has to review it?"
    print(f"\n[Turn 2] USER: {q2}\n[Turn 2] AGENT:\n{process_user_query(q2, thread_id=thread_id)}")
    q3 = "What is the langgraph?"
    print(f"\n[Turn 3] USER: {q3}\n[Turn 3] AGENT:\n{process_user_query(q3, thread_id=thread_id)}")