import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Check for API Key
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in .env file.")

# Define Pydantic schema for structured output
class QAPair(BaseModel):
    id: str = Field(description="Unique ID for the QA pair, e.g., 'hr_01'")
    department: str = Field(description="Department name (HR, IT_Support, Billing, Shipping)")
    audience: str = Field(description="Target audience: 'internal' or 'external'")
    question: str = Field(description="Frequently asked question")
    answer: str = Field(description="Detailed and helpful answer")
    doc: str = Field(description="Combined formatted string: Q: <question>\nA: <answer>")

class DepartmentDataset(BaseModel):
    qa_pairs: list[QAPair] = Field(description="List of generated QA pairs")

# Initialize LLM with structured output
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
structured_llm = llm.with_structured_output(DepartmentDataset)

# Department configurations
DEPARTMENTS = [
    {
        "name": "HR",
        "audience": "internal",
        "prefix": "hr",
        "description": "Handles employee lifecycle matters like paid time off (PTO), payroll, performance reviews, internal transfers, and HR policies."
    },
    {
        "name": "IT_Support",
        "audience": "internal",
        "prefix": "it",
        "description": "Manages technical issues related to employee laptops, software access, VPN setup, password resets, and digital tools."
    },
    {
        "name": "Billing",
        "audience": "external",
        "prefix": "bill",
        "description": "Assists retail customers with invoice issues, payment methods, refund requests, overcharges, and billing disputes."
    },
    {
        "name": "Shipping",
        "audience": "external",
        "prefix": "ship",
        "description": "Handles retail customer queries about order tracking, delivery delays, damaged goods upon arrival, and return pickups."
    }
]

def generate_all_data():
    all_faqs = []

    for dept in DEPARTMENTS:
        print(f"Generating 12 QA pairs for department: {dept['name']}...")
        
        prompt = f"""
        Act as an domain expert for the '{dept['name']}' department at ShopUNow, a large retail company.
        
        Target Audience: {dept['audience']} users.
        Department Responsibility: {dept['description']}
        
        Generate exactly 12 realistic and detailed Question & Answer (QA) pairs that {dept['audience']} users would ask this department.
        
        Requirements:
        1. Set 'id' using prefix '{dept['prefix']}_01', '{dept['prefix']}_02', etc.
        2. Set 'department' to '{dept['name']}'.
        3. Set 'audience' to '{dept['audience']}'.
        4. Make answers clear, policy-oriented, and specific to ShopUNow.
        5. 'doc' field must be formatted as: "Q: <question>\\nA: <answer>"
        """

        response = structured_llm.invoke(prompt)
        dept_data = [item.model_dump() for item in response.qa_pairs]
        all_faqs.extend(dept_data)

    # Save to data/faqs.json
    output_path = os.path.join("data", "faqs.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_faqs, f, indent=2)

    print(f"\nSuccess! Total {len(all_faqs)} QA pairs saved to {output_path}")

if __name__ == "__main__":
    generate_all_data()