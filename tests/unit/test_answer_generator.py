import pytest
from unittest.mock import MagicMock

from llm.answer_generator import RAGAnswerGenerator


@pytest.fixture
def context_injector():
    mock = MagicMock()

    mock.prepare.return_value = {
        "messages": [
            {"role": "system", "content": "Research assistant"},
            {"role": "user", "content": "Research context and question"}
        ],
        "context": "CNNs are used for image classification.",
        "retrieved_chunks": [
            {
                "id": "doc-1:0",
                "text": "CNNs are used for image classification.",
                "metadata": {
                    "document_id": "doc-1",
                    "page_number": 1
                }
            }
        ]
    }

    return mock


@pytest.fixture
def llm_client():
    mock = MagicMock()
    mock.generate.return_value = (
        "CNNs extract visual features and use them for classification."
    )
    return mock


@pytest.fixture
def answer_generator(context_injector, llm_client):
    return RAGAnswerGenerator(
        context_injector=context_injector,
        llm_client=llm_client
    )


def test_generate_answer_returns_expected_structure(answer_generator):
    result = answer_generator.generate_answer(
        "How are CNNs used?"
    )

    assert "question" in result
    assert "answer" in result
    assert "context" in result
    assert "retrieved_chunks" in result


def test_generate_answer_returns_llm_response(answer_generator):
    result = answer_generator.generate_answer(
        "How are CNNs used?"
    )

    assert "CNNs extract visual features" in result["answer"]


def test_generate_answer_calls_context_injector(
    answer_generator,
    context_injector
):
    question = "How are CNNs used?"

    answer_generator.generate_answer(question)

    context_injector.prepare.assert_called_once_with(
        question=question,
        n_results=None,
        filters=None
    )


def test_generate_answer_calls_llm_client(
    answer_generator,
    context_injector,
    llm_client
):
    answer_generator.generate_answer("How are CNNs used?")

    llm_client.generate.assert_called_once_with(
        messages=context_injector.prepare.return_value["messages"]
    )


def test_generate_answer_preserves_retrieved_chunks(answer_generator):
    result = answer_generator.generate_answer(
        "How are CNNs used?"
    )

    assert len(result["retrieved_chunks"]) == 1
    assert result["retrieved_chunks"][0]["metadata"]["page_number"] == 1


def test_generate_answer_passes_retrieval_parameters(
    answer_generator,
    context_injector
):
    answer_generator.generate_answer(
        question="How are CNNs used?",
        n_results=3,
        filters={"document_id": "doc-1"}
    )

    context_injector.prepare.assert_called_once_with(
        question="How are CNNs used?",
        n_results=3,
        filters={"document_id": "doc-1"}
    )


def test_empty_question_raises_error(answer_generator):
    with pytest.raises(ValueError):
        answer_generator.generate_answer("")


def test_invalid_question_type_raises_error(answer_generator):
    with pytest.raises(ValueError):
        answer_generator.generate_answer(None)