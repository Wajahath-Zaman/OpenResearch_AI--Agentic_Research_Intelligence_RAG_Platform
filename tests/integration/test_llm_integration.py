import pytest

from llm.answer_generator import RAGAnswerGenerator
from llm.client import LLMClient
from llm.context_injector import (
    RAGContextInjector,
    NoRetrievedContextError
)
from llm.fallback_provider import FallbackLLMProvider
from llm.retry_provider import RetryingLLMProvider

from llm.providers.base import (
    LLMProvider,
    LLMProviderError,
    RetryableLLMProviderError
)

from retrieval.context_assembler import ContextAssembler


class FakeRetriever:

    def search(self, query, n_results=None, filters=None):

        return [
            {
                "id": "doc-1:0",
                "text": "Retrieval-Augmented Generation combines "
                        "retrieval systems with language models.",
                "metadata": {
                    "document_id": "doc-1",
                    "chunk_index": 0,
                    "page_number": 1,
                    "title": "Introduction to RAG",
                    "source_path": "data/raw/rag.pdf"
                }
            }
        ]


class EmptyRetriever:

    def search(self, query, n_results=None, filters=None):
        return []


class FakeProvider(LLMProvider):

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def generate(self, messages, max_completion_tokens=1024):

        self.calls += 1

        if not self.responses:
            raise RuntimeError("No fake response configured.")

        response = self.responses.pop(0)

        if isinstance(response, Exception):
            raise response

        return response


@pytest.fixture
def context_injector():

    return RAGContextInjector(
        retriever=FakeRetriever(),
        context_assembler=ContextAssembler()
    )


@pytest.fixture
def sample_messages():

    return [
        {
            "role": "system",
            "content": "You are a research assistant."
        },
        {
            "role": "user",
            "content": "Explain RAG using the supplied context."
        }
    ]


def build_answer_generator(context_injector, primary, fallback):

    retrying_primary = RetryingLLMProvider(
        provider=primary,
        max_attempts=1,
        initial_delay=0,
        sleep_func=lambda _: None
    )

    retrying_fallback = RetryingLLMProvider(
        provider=fallback,
        max_attempts=1,
        initial_delay=0,
        sleep_func=lambda _: None
    )

    fallback_provider = FallbackLLMProvider(
        primary_provider=retrying_primary,
        fallback_provider=retrying_fallback
    )

    llm_client = LLMClient(provider=fallback_provider)

    return RAGAnswerGenerator(
        context_injector=context_injector,
        llm_client=llm_client
    )


def test_complete_rag_pipeline_with_groq(context_injector):

    groq = FakeProvider([
        "RAG combines information retrieval with generation."
    ])

    gemini = FakeProvider([
        "Gemini response."
    ])

    answer_generator = build_answer_generator(
        context_injector,
        groq,
        gemini
    )

    result = answer_generator.generate_answer(
        "What is RAG?"
    )

    assert result["answer"] == (
        "RAG combines information retrieval with generation."
    )

    assert groq.calls == 1
    assert gemini.calls == 0

    assert len(result["retrieved_chunks"]) == 1
    assert len(result["citations"]) == 1


def test_gemini_fallback_when_groq_fails(context_injector):

    groq = FakeProvider([
        LLMProviderError("Groq unavailable.")
    ])

    gemini = FakeProvider([
        "Gemini successfully generated the answer."
    ])

    answer_generator = build_answer_generator(
        context_injector,
        groq,
        gemini
    )

    result = answer_generator.generate_answer(
        "What is RAG?"
    )

    assert result["answer"] == (
        "Gemini successfully generated the answer."
    )

    assert groq.calls == 1
    assert gemini.calls == 1


def test_both_providers_failing(context_injector):

    groq = FakeProvider([
        LLMProviderError("Groq unavailable.")
    ])

    gemini = FakeProvider([
        LLMProviderError("Gemini unavailable.")
    ])

    answer_generator = build_answer_generator(
        context_injector,
        groq,
        gemini
    )

    with pytest.raises(LLMProviderError) as exc_info:

        answer_generator.generate_answer(
            "What is RAG?"
        )

    assert "Both primary and fallback LLM providers failed" in (
        str(exc_info.value)
    )


def test_retry_before_success(context_injector):

    groq = FakeProvider([
        RetryableLLMProviderError("Temporary failure."),
        "Successful response after retry."
    ])

    gemini = FakeProvider([
        "Gemini response."
    ])

    retrying_groq = RetryingLLMProvider(
        provider=groq,
        max_attempts=2,
        initial_delay=0,
        sleep_func=lambda _: None
    )

    retrying_gemini = RetryingLLMProvider(
        provider=gemini,
        max_attempts=1,
        initial_delay=0,
        sleep_func=lambda _: None
    )

    fallback = FallbackLLMProvider(
        primary_provider=retrying_groq,
        fallback_provider=retrying_gemini
    )

    answer_generator = RAGAnswerGenerator(
        context_injector=context_injector,
        llm_client=LLMClient(fallback)
    )

    result = answer_generator.generate_answer(
        "What is RAG?"
    )

    assert result["answer"] == (
        "Successful response after retry."
    )

    assert groq.calls == 2
    assert gemini.calls == 0


def test_no_context_prevents_llm_call():

    injector = RAGContextInjector(
        retriever=EmptyRetriever(),
        context_assembler=ContextAssembler()
    )

    groq = FakeProvider(["Unexpected response."])
    gemini = FakeProvider(["Unexpected response."])

    answer_generator = build_answer_generator(
        injector,
        groq,
        gemini
    )

    with pytest.raises(NoRetrievedContextError):

        answer_generator.generate_answer(
            "What is RAG?"
        )

    assert groq.calls == 0
    assert gemini.calls == 0