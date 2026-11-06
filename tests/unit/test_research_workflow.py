from unittest.mock import Mock

from agent.nodes.workflows import research_workflow


def test_research_workflow():

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

    state = {
        "query": "Find research papers about Vision Transformers"
    }

    result = research_workflow(
        state,
        discovery_service=mock_service
    )

    assert len(result["research_results"]) == 1

    assert (
        result["research_results"][0]["title"]
        == "Vision Transformers for Medical Imaging"
    )

    assert "2025" in result["answer"]
    assert "10.1234/example" in result["answer"]
    assert result["error"] is None

    mock_service.discover.assert_called_once_with(
        query=state["query"],
        per_page=10
    )


def test_research_workflow_handles_no_results():

    mock_service = Mock()

    mock_service.discover.return_value = {
        "results": []
    }

    state = {
        "query": "Find research papers about an unknown topic"
    }

    result = research_workflow(
        state,
        discovery_service=mock_service
    )

    assert result["research_results"] == []
    assert "No research papers found" in result["answer"]
    assert result["error"] is None


def test_research_workflow_handles_api_failure():

    mock_service = Mock()

    mock_service.discover.side_effect = Exception(
        "OpenAlex API unavailable"
    )

    state = {
        "query": "Find research papers about AI"
    }

    result = research_workflow(
        state,
        discovery_service=mock_service
    )

    assert result["research_results"] == []
    assert result["answer"] == ""
    assert result["error"] == "OpenAlex API unavailable"