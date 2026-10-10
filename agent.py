import os
from typing import Literal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

# Configuration
PERSIST_DIRECTORY = "./chroma_db"
COLLECTION_NAME = "shopunow_faqs"
ALLOWED_DEPARTMENTS = ["HR", "IT Support", "Billing & Payments", "Shipping & Delivery"]

# Initialize Vector Store & LLM
embeddings = OpenAIEmbeddings()
vector_store = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings
)
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Pydantic Schema for Categorization
class QueryClassification(BaseModel):
    sentiment: Literal["positive", "neutral", "negative"] = Field(
        description="Sentiment of the user query."
    )
    department: Literal["HR", "IT Support", "Facilities & Admin", "Billing & Payments", "Shipping & Delivery", "Other"] = Field(
        description="Target department for the user query."
    )
    reasoning: str = Field(
        description="Brief 1-sentence rationale for the classification."
    )

classifier = llm.with_structured_output(QueryClassification)

# Pydantic Schema for Human Escalation Form (Stretch Goal Option 2)
class EscalationForm(BaseModel):
    full_name: str
    email: str
    phone_number: str
    issue_summary: str

def collect_escalation_details(query: str, reason: str) -> str:
    """
    Interactively collects customer contact details when human escalation is triggered.
    """
    print("\n--- HUMAN SUPPORT ESCALATION FORM ---")
    print(f"Reason for Escalation: {reason}")
    print("Please provide your contact details so our support team can reach out:\n")
    
    name = input("Enter your Full Name: ")
    email = input("Enter your Email Address: ")
    phone = input("Enter your Phone Number: ")
    
    form_data = EscalationForm(
        full_name=name,
        email=email,
        phone_number=phone,
        issue_summary=query
    )
    
    print("\n------------------------------------")
    return (
        f"Thank you, {form_data.full_name}. Your inquiry ('{form_data.issue_summary}') "
        f"has been forwarded to a human representative. We will contact you shortly at "
        f"{form_data.email} or {form_data.phone_number}."
    )

def process_user_query(query: str) -> str:
    """
    Analyzes sentiment and department, then routes to standard RAG or Human Escalation.
    """
    # System Prompt defining company structure
    system_prompt = (
        "You are an intent and sentiment classification agent for ShopUNow.\n"
        "Classify the user query into one of these departments:\n"
        "- HR: Internal employee leave, payroll, policy, performance.\n"
        "- IT Support: Internal employee laptop, network, login, software issues.\n"
        "- Facilities & Admin: Workspace logistics, maintenance, access cards, office supplies.\n"
        "- Billing & Payments: External customer invoices, refunds, payment methods, overcharges.\n"
        "- Shipping & Delivery: External customer order tracking, delivery delays, damaged goods, return pickups.\n"
        "- Other: Query does not fit any department above.\n\n"
        "Evaluate sentiment as: positive, neutral, or negative."
    )

    # Classify Sentiment and Department
    analysis = classifier.invoke(f"{system_prompt}\n\nUser Query: {query}")
    print(f"\n[Classification Log]: Department='{analysis.department}' | Sentiment='{analysis.sentiment}'")
    print(f"[Reasoning]: {analysis.reasoning}")

    # Route Trigger: Negative Sentiment
    if analysis.sentiment == "negative":
        print("\n[Routing Action]: Escalating due to negative sentiment.")
        return collect_escalation_details(query, reason="Negative sentiment detected.")

    # Route Trigger: Unrecognized/Invalid Department
    if analysis.department not in ALLOWED_DEPARTMENTS:
        print("\n[Routing Action]: Escalating due to unhandled department scope.")
        return collect_escalation_details(query, reason=f"Department '{analysis.department}' is not handled by self-service.")

    # Route Trigger: Standard RAG Workflow (Positive/Neutral + Valid Department)
    print(f"\n[Routing Action]: Querying ChromaDB knowledge base for '{analysis.department}'...")
    retriever = vector_store.as_retriever(
        search_kwargs={"filter": {"category": analysis.department}, "k": 3}
    )
    docs = retriever.invoke(query)
    
    if not docs:
        return "I could not find relevant internal documentation to answer your request. Please contact support."

    context = "\n\n".join([doc.page_content for doc in docs])

    qa_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful customer and employee support assistant for ShopUNow.\n"
            "Answer the user query accurately using only the context provided below.\n\n"
            "Context:\n{context}"
        ),
        ("user", "{question}")
    ])
    
    rag_chain = qa_prompt | llm
    response = rag_chain.invoke({"context": context, "question": query})
    return response.content

# Driver Code to test queries
if __name__ == "__main__":
    test_queries = [
        "How do I apply for paid time off (PTO)?",
        "My order arrived completely broken and your service is terrible!",
        "Where can I buy company shares on the stock market?"
    ]
    
    for q in test_queries:
        print("\n" + "="*60)
        print(f"USER QUERY: {q}")
        output = process_user_query(q)
        print("\nAGENT RESPONSE:")
        print(output)