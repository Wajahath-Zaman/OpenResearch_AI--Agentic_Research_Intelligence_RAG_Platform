import pytest
from unittest.mock import MagicMock

from llm.context_injector import (
    RAGContextInjector,
    NoRetrievedContextError
)


@pytest.fixture
def retriever():
    mock = MagicMock()

    mock.search.return_value = [
        {
            "id": "doc-1:0",
            "text": "CNNs are used in image classification.",
            "metadata": {
                "document_id": "doc-1",
                "page_number": 1,
                "title": "Deep Learning Research"
            },
            "similarity": 0.92
        }
    ]

    return mock


@pytest.fixture
def context_assembler():
    mock = MagicMock()
    mock.assemble.return_value = (
        "Source 1: Deep Learning Research\n"
        "Page: 1\n"
        "CNNs are used in image classification."
    )
    return mock


@pytest.fixture
def prompt_builder():
    mock = MagicMock()

    mock.build_messages.return_value = [
        {"role": "system", "content": "Research assistant"},
        {"role": "user", "content": "Research context and question"}
    ]

    return mock


@pytest.fixture
def injector(retriever, context_assembler, prompt_builder):
    return RAGContextInjector(
        retriever=retriever,
        context_assembler=context_assembler,
        prompt_builder=prompt_builder
    )


def test_prepare_returns_expected_structure(injector):
    result = injector.prepare("How are CNNs used?")

    assert "messages" in result
    assert "context" in result
    assert "retrieved_chunks" in result


def test_prepare_retrieves_chunks(injector, retriever):
    injector.prepare("How are CNNs used?")

    retriever.search.assert_called_once()


def test_prepare_assembles_context(
    injector,
    context_assembler,
    retriever
):
    injector.prepare("How are CNNs used?")

    context_assembler.assemble.assert_called_once_with(
        retriever.search.return_value
    )


def test_prepare_builds_messages(
    injector,
    prompt_builder,
    context_assembler
):
    question = "How are CNNs used?"

    injector.prepare(question)

    prompt_builder.build_messages.assert_called_once_with(
        context=context_assembler.assemble.return_value,
        question=question
    )


def test_prepare_preserves_retrieved_chunks(injector, retriever):
    result = injector.prepare("How are CNNs used?")

    assert result["retrieved_chunks"] == retriever.search.return_value


def test_prepare_passes_retrieval_parameters(injector, retriever):
    injector.prepare(
        question="How are CNNs used?",
        n_results=3,
        filters={"document_id": "doc-1"}
    )

    retriever.search.assert_called_once_with(
        query="How are CNNs used?",
        n_results=3,
        filters={"document_id": "doc-1"}
    )


def test_empty_question_raises_error(injector):
    with pytest.raises(ValueError):
        injector.prepare("")


def test_invalid_question_type_raises_error(injector):
    with pytest.raises(ValueError):
        injector.prepare(None)


def test_no_retrieved_chunks_raises_error(injector, retriever):
    retriever.search.return_value = []

    with pytest.raises(NoRetrievedContextError):
        injector.prepare("What is deep learning?")