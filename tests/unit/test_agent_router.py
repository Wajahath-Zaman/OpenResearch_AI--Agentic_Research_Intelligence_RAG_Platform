import pytest

from agent.nodes.router import route_query


@pytest.mark.parametrize(
    "query",
    [
        "Find papers about Vision Transformers",
        "Find research on medical imaging",
        "Search papers about deep learning",
        "Discover papers about LLMs",
        "Show me recent papers on RAG",
        "What are the latest papers on AI agents?"
    ]
)
def test_router_selects_research_discovery(query):

    result = route_query({"query": query})

    assert result["route"] == "research"
    assert result["error"] is None


@pytest.mark.parametrize(
    "query",
    [
        "Explain the uploaded research paper",
        "What accuracy did Swin-Base achieve?",
        "Summarize the methodology",
        "Compare the findings in this document"
    ]
)
def test_router_selects_rag(query):

    result = route_query({"query": query})

    assert result["route"] == "rag"
    assert result["error"] is None


def test_router_rejects_empty_query():

    result = route_query({"query": "   "})

    assert result["route"] == ""
    assert result["error"] == "Query cannot be empty."