from unittest.mock import Mock

from agent.graph import build_agent_graph


def create_initial_state(query):

    return {
        "query": query,
        "route": "",
        "retrieved_chunks": [],
        "context": "",
        "research_results": [],
        "document_ids": ["paper-a", "paper-b"],
        "comparison_results": [],
        "messages": [],
        "answer": "",
        "citations": [],
        "error": None
    }


def test_comparison_through_agent_graph():

    # Mock intent classifier
    classifier = Mock()
    classifier.classify.return_value = "comparison"

    # Mock comparison handler
    comparison_handler = Mock()
    comparison_handler.compare.return_value = {
        "comparison_results": [
            {
                "document_id": "paper-a",
                "analysis": "Paper A uses CNN architecture.",
                "citations": [{"page_number": 2}],
                "retrieved_chunks": [{"text": "CNN evidence"}]
            },
            {
                "document_id": "paper-b",
                "analysis": "Paper B uses Vision Transformer architecture.",
                "citations": [{"page_number": 5}],
                "retrieved_chunks": [{"text": "ViT evidence"}]
            }
        ],
        "comparison": "Paper A uses CNN, whereas Paper B uses Vision Transformers."
    }

    agent = build_agent_graph(
        intent_classifier=classifier,
        comparison_handler=comparison_handler
    )

    result = agent.invoke(
        create_initial_state(
            "Compare the methodologies of these two papers"
        )
    )

    assert result["route"] == "comparison"

    assert len(result["comparison_results"]) == 2

    assert "CNN" in result["answer"]
    assert "Vision Transformers" in result["answer"]
    assert "whereas" in result["answer"]

    assert result["error"] is None

    comparison_handler.compare.assert_called_once()


def test_comparison_failure_through_agent_graph():

    classifier = Mock()
    classifier.classify.return_value = "comparison"

    comparison_handler = Mock()
    comparison_handler.compare.side_effect = RuntimeError(
        "Retrieval service unavailable"
    )

    agent = build_agent_graph(
        intent_classifier=classifier,
        comparison_handler=comparison_handler
    )

    result = agent.invoke(
        create_initial_state("Compare these papers")
    )

    assert result["route"] == "comparison"
    assert result["error"] == "Retrieval service unavailable"
    assert result["answer"] == ""