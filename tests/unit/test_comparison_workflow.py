import pytest
from unittest.mock import Mock

from agent.nodes.comparison import PaperComparisonWorkflow


@pytest.fixture
def mock_generator():

    generator = Mock()

    generator.generate_answer.side_effect = [
        {
            "answer": "Paper A uses CNN architecture.",
            "citations": [{"page_number": 1}],
            "retrieved_chunks": [{"text": "CNN evidence"}]
        },
        {
            "answer": "Paper B uses Vision Transformer architecture.",
            "citations": [{"page_number": 2}],
            "retrieved_chunks": [{"text": "ViT evidence"}]
        }
    ]

    return generator


@pytest.fixture
def mock_llm():

    llm = Mock()
    llm.generate.return_value = "Structured comparison of Paper A and Paper B."

    return llm


def test_compare_two_papers(mock_generator, mock_llm):

    workflow = PaperComparisonWorkflow(mock_generator, mock_llm)

    result = workflow.compare(
        query="Compare their methodologies",
        document_ids=["paper-a", "paper-b"]
    )

    assert len(result["comparison_results"]) == 2

    assert "CNN architecture" in result["comparison_results"][0]["analysis"]
    assert "Vision Transformer" in result["comparison_results"][1]["analysis"]

    assert result["comparison"] == (
        "Structured comparison of Paper A and Paper B."
    )


def test_comparison_uses_document_filters(mock_generator, mock_llm):

    workflow = PaperComparisonWorkflow(mock_generator, mock_llm)

    workflow.compare(
        query="Compare their methodologies",
        document_ids=["paper-a", "paper-b"]
    )

    calls = mock_generator.generate_answer.call_args_list

    assert calls[0].kwargs["filters"] == {"document_id": "paper-a"}
    assert calls[1].kwargs["filters"] == {"document_id": "paper-b"}


def test_synthesis_calls_llm(mock_generator, mock_llm):

    workflow = PaperComparisonWorkflow(mock_generator, mock_llm)

    workflow.compare(
        query="Compare these papers",
        document_ids=["paper-a", "paper-b"]
    )

    mock_llm.generate.assert_called_once()

    messages = mock_llm.generate.call_args.kwargs["messages"]

    assert "Methodologies" in messages[0]["content"]
    assert "Paper A uses CNN architecture." in messages[1]["content"]
    assert "Paper B uses Vision Transformer architecture." in messages[1]["content"]


def test_comparison_requires_two_documents(mock_generator, mock_llm):

    workflow = PaperComparisonWorkflow(mock_generator, mock_llm)

    with pytest.raises(ValueError):
        workflow.compare("Compare papers", ["paper-a"])


def test_comparison_rejects_duplicate_documents(mock_generator, mock_llm):

    workflow = PaperComparisonWorkflow(mock_generator, mock_llm)

    with pytest.raises(ValueError):
        workflow.compare(
            "Compare papers",
            ["paper-a", "paper-a"]
        )


def test_comparison_rejects_empty_query(mock_generator, mock_llm):

    workflow = PaperComparisonWorkflow(mock_generator, mock_llm)

    with pytest.raises(ValueError):
        workflow.compare("", ["paper-a", "paper-b"])