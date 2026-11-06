from unittest.mock import Mock

from agent.nodes.workflows import rag_workflow


def test_rag_workflow():

    mock_generator = Mock()

    mock_generator.generate_answer.return_value = {
        "answer": "Swin-Base achieved 98.80% accuracy.",
        "context": "Retrieved research context.",
        "retrieved_chunks": [
            {"text": "Swin-Base achieved 98.80% accuracy."}
        ],
        "citations": [
            {"citation_id": 1, "page_number": 5}
        ]
    }

    state = {
        "query": "What accuracy did Swin-Base achieve?"
    }

    result = rag_workflow(
        state,
        answer_generator=mock_generator
    )

    assert result["answer"] == "Swin-Base achieved 98.80% accuracy."
    assert result["context"] == "Retrieved research context."
    assert len(result["retrieved_chunks"]) == 1
    assert len(result["citations"]) == 1
    assert result["error"] is None

    mock_generator.generate_answer.assert_called_once_with(
        question="What accuracy did Swin-Base achieve?",
        n_results=5
    )