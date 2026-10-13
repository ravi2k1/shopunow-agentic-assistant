import uuid
import streamlit as st
from langchain_core.messages import HumanMessage

# Import compiled LangGraph workflow from agent.py
from agent import app, ALLOWED_DEPARTMENTS

# Page Configuration
st.set_page_config(
    page_title="ShopUNow AI Support Assistant",
    page_icon="🛍️",
    layout="wide"
)

# Session State Initialization
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "escalated" not in st.session_state:
    st.session_state.escalated = False

if "last_query" not in st.session_state:
    st.session_state.last_query = ""

# Sidebar
with st.sidebar:
    st.title("🛍️ ShopUNow Assistant")
    st.caption("Agentic RAG Assistant powered by LangGraph")
    st.divider()

    st.markdown("**Session Info**")
    st.code(f"Thread ID:\n{st.session_state.thread_id}", language="text")

    if st.button("🔄 Reset Conversation", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.escalated = False
        st.session_state.last_query = ""
        st.rerun()

    st.divider()
    st.markdown("**Supported Departments:**")
    for dept in ALLOWED_DEPARTMENTS:
        if dept != "Conversational":
            st.markdown(f"- `{dept}`")

# Header
st.title("ShopUNow Support Center")
st.subheader("How can we help you today?")
st.divider()

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Interactive Escalation Form
if st.session_state.escalated:
    st.warning("⚠️ Your query requires human support escalation.")
    with st.form("escalation_form"):
        st.subheader("Human Support Ticket Details")
        full_name = st.text_input("Full Name", placeholder="Name")
        email = st.text_input("Email Address", placeholder="email@example.com")
        phone = st.text_input("Phone Number", placeholder="Phone Number")
        issue_summary = st.text_area("Issue Summary", value=st.session_state.last_query, disabled=True)

        submit_button = st.form_submit_button("Submit Support Ticket")

        if submit_button:
            if not full_name or not email or not phone:
                st.error("Please fill out all contact fields before submitting.")
            else:
                confirmation_msg = (
                    f"Thank you **{full_name}**! Your support ticket for *'{st.session_state.last_query}'* "
                    f"has been successfully logged. Our team will reach out to you at **{email}** "
                    f"or **{phone}** within 24 business hours."
                )

                st.session_state.messages.append({"role": "assistant", "content": confirmation_msg})
                st.session_state.escalated = False
                st.success("Ticket Submitted Successfully!")
                st.rerun()

# Standard Chat Input Box
if not st.session_state.escalated:
    if prompt := st.chat_input("Type your question here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        st.session_state.last_query = prompt
        config = {"configurable": {"thread_id": st.session_state.thread_id}}

        # Explicit input state reset on every invocation
        input_state = {
            "messages": [HumanMessage(content=prompt)],
            "department": "",
            "sentiment": "neutral",
            "standalone_query": prompt,
            "context": "",
            "escalation_needed": False,
            "escalation_reason": ""
        }

        with st.chat_message("assistant"):
            with st.status("Agent processing query...", expanded=True) as status:
                st.write("🔍 Classifying intent & sentiment...")

                result = app.invoke(input_state, config=config)

                dept = result.get("department", "Unknown")
                sentiment = result.get("sentiment", "neutral")
                standalone_q = result.get("standalone_query", prompt)
                escalation_needed = result.get("escalation_needed", False)
                escalation_reason = result.get("escalation_reason", "")

                st.write(f"🏷️ **Department:** `{dept}` | **Sentiment:** `{sentiment}`")

                if standalone_q != prompt:
                    st.write(f"🧠 **Query Condenser:** Rephrased to *'{standalone_q}'*")

                if escalation_needed:
                    st.write(f"🚨 **Escalation Triggered:** {escalation_reason}")
                    status.update(label="Escalation Required", state="error", expanded=False)
                    st.session_state.escalated = True
                else:
                    st.write("📚 **Retrieval:** Fetched context using Hybrid Search (BM25 + Dense MMR)")
                    status.update(label="Response Generated", state="complete", expanded=False)
                    st.session_state.escalated = False

            if not escalation_needed:
                response_text = result["messages"][-1].content
                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
            else:
                st.rerun()