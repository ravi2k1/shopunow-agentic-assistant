import os
import json
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Load environment variables
load_dotenv()

DATA_PATH = os.path.join("data", "faqs.json")
PERSIST_DIRECTORY = "chroma_db"

def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"File {DATA_PATH} not found. Run generate_data.py first.")
    
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def build_vector_store():
    raw_data = load_data()
    documents = []

    for item in raw_data:
        doc = Document(
            page_content=item["doc"],
            metadata={
                "id": item["id"],
                "department": item["department"],
                "audience": item["audience"],
                "question": item["question"]
            }
        )
        documents.append(doc)

    print(f"Loaded {len(documents)} document chunks.")

    # Initialize Embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Build and persist Chroma Vector DB
    print(f"Creating vector store in local directory: '{PERSIST_DIRECTORY}'...")
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY,
        collection_name="shopunow_faqs"
    )

    print("Vector database build complete!")
    return vector_store

def test_metadata_filtering(vector_store):
    print("\n--- Testing Metadata Filtering (Department: HR) ---")
    query = "How do I take leave or time off?"
    results = vector_store.similarity_search(
        query, 
        k=2, 
        filter={"department": "HR"}
    )
    for i, res in enumerate(results, 1):
        print(f"\n[Result {i}] Department: {res.metadata['department']}")
        print(f"Content: {res.page_content[:100]}...")

if __name__ == "__main__":
    v_store = build_vector_store()
    test_metadata_filtering(v_store)