from unittest.mock import Mock

from agent.graph import build_agent_graph


def create_initial_state(query):

    return {
        "query": query,
        "route": "",
        "retrieved_chunks": [],
        "context": "",
        "research_results": [],
        "messages": [],
        "answer": "",
        "citations": [],
        "error": None
    }


def test_research_discovery_through_graph():

    mock_service = Mock()

    mock_service.discover.return_value = {
        "results": [
            {
                "title": "Vision Transformers for Medical Imaging",
                "authors": ["Author A", "Author B"],
                "publication_year": 2025,
                "doi": "10.1234/example"
            }
        ]
    }

    agent = build_agent_graph(
        discovery_service=mock_service
    )

    query = "Find recent papers about Vision Transformers"

    result = agent.invoke(create_initial_state(query))

    assert result["route"] == "research"
    assert len(result["research_results"]) == 1
    assert result["research_results"][0]["title"] == (
        "Vision Transformers for Medical Imaging"
    )
    assert "2025" in result["answer"]
    assert result["error"] is None

    mock_service.discover.assert_called_once_with(
        query=query,
        per_page=10
    )


def test_research_graph_handles_empty_results():

    mock_service = Mock()

    mock_service.discover.return_value = {
        "results": []
    }

    agent = build_agent_graph(
        discovery_service=mock_service
    )

    result = agent.invoke(
        create_initial_state(
            "Find research papers about an unknown topic"
        )
    )

    assert result["route"] == "research"
    assert result["research_results"] == []
    assert "No research papers found" in result["answer"]
    assert result["error"] is None


def test_research_graph_handles_api_failure():

    mock_service = Mock()

    mock_service.discover.side_effect = Exception(
        "OpenAlex API unavailable"
    )

    agent = build_agent_graph(
        discovery_service=mock_service
    )

    result = agent.invoke(
        create_initial_state(
            "Find recent papers about artificial intelligence"
        )
    )

    assert result["route"] == "research"
    assert result["research_results"] == []
    assert result["error"] == "OpenAlex API unavailable"