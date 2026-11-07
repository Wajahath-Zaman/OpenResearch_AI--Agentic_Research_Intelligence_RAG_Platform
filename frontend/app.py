import streamlit as st

from frontend.services.api_client import APIClient, APIError
from frontend.components.research_discovery import (
    render_research_discovery
)

from frontend.components.document_management import (
    render_document_management
)

from frontend.components.chat_interface import render_chat_interface

from frontend.components.paper_comparison import (
    render_paper_comparison
)

st.set_page_config(
    page_title="Open Research Intelligence Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

api = APIClient()

# Sidebar navigation
st.sidebar.title("🔬 Research Assistant")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Research Discovery",
        "Document Management",
        "Research Assistant",
        "Paper Comparison"
    ]
)

# Backend status
backend_connected = False

try:
    api.health_check()
    backend_connected = True
    st.sidebar.success("Backend Connected")
except APIError:
    st.sidebar.error("Backend Disconnected")


# Dashboard
if page == "Dashboard":

    st.title("🔬 Open Research Intelligence Assistant")

    st.caption(
        "Discover research papers, explore scientific literature, "
        "and generate evidence-based insights."
    )

    st.divider()

    st.subheader("Research Dashboard")

    if not backend_connected:
        st.warning(
            "Backend services are currently unavailable. "
            "Start FastAPI to access dashboard data."
        )

    else:
        try:
            with st.spinner("Loading dashboard metrics..."):
                documents = api.list_documents()

            documents = documents or []

            total_documents = len(documents)

            processed_documents = sum(
                1 for doc in documents
                if doc["status"] == "processed"
            )

            pending_documents = sum(
                1 for doc in documents
                if doc["status"] in ["discovered", "downloaded"]
            )

            failed_documents = sum(
                1 for doc in documents
                if doc["status"] == "failed"
            )

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Total Documents", total_documents)
            col2.metric("Processed", processed_documents)
            col3.metric("Pending", pending_documents)
            col4.metric("Failed", failed_documents)

            if total_documents == 0:
                st.info(
                    "No research documents found. "
                    "Start by discovering or uploading research papers."
                )

        except APIError as error:
            st.error(f"Unable to load dashboard: {error}")

    st.divider()

    st.subheader("Available Features")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info(
            "📚 **Research Discovery**\n\n"
            "Find scientific papers using OpenAlex."
        )

    with col2:
        st.info(
            "📄 **Document Management**\n\n"
            "Upload and process research PDFs."
        )

    with col3:
        st.info(
            "🤖 **Research Assistant**\n\n"
            "Ask questions and compare research papers."
        )

# Research Discovery
elif page == "Research Discovery":

    render_research_discovery(api)

elif page == "Document Management":

    render_document_management(api)

elif page == "Research Assistant":

    render_chat_interface(api)

elif page == "Paper Comparison":

    render_paper_comparison(api)