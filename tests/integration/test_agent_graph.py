import pytest
from unittest.mock import Mock

from agent.graph import build_agent_graph


@pytest.fixture
def mock_generator():

    generator = Mock()

    generator.generate_answer.return_value = {
        "question": "What accuracy did Swin-Base achieve?",
        "answer": "Swin-Base achieved 98.80% accuracy.",
        "context": "Retrieved research context.",
        "retrieved_chunks": [
            {"text": "Swin-Base achieved 98.80% accuracy."}
        ],
        "citations": [
            {"citation_id": 1, "page_number": 5}
        ]
    }

    return generator


@pytest.fixture
def agent(mock_generator):

    return build_agent_graph(
        answer_generator=mock_generator
    )


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


def test_graph_routes_to_rag(agent, mock_generator):

    result = agent.invoke(
        create_initial_state(
            "What accuracy did Swin-Base achieve?"
        )
    )

    assert result["route"] == "rag"
    assert result["answer"] == "Swin-Base achieved 98.80% accuracy."
    assert len(result["retrieved_chunks"]) == 1
    assert len(result["citations"]) == 1
    assert result["error"] is None

    mock_generator.generate_answer.assert_called_once()


def test_graph_routes_to_research(agent, mock_generator):

    result = agent.invoke(
        create_initial_state(
            "Find recent papers about Vision Transformers"
        )
    )

    assert result["route"] == "research"
    assert "Found" in result['answer']
    assert "research papers" in result["answer"]
    assert len(result['research_results']) > 0

    mock_generator.generate_answer.assert_not_called()