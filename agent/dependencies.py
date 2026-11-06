from config.settings import get_settings

from ingestion.embedding_generator import EmbeddingGenerator
from ingestion.discovery_service import ResearchDiscoveryService

from retrieval.vector_store import ChromaVectorStore
from retrieval.vector_retriever import VectorRetriever
from retrieval.context_assembler import ContextAssembler

from llm.context_injector import RAGContextInjector
from llm.answer_generator import RAGAnswerGenerator
from llm.client import LLMClient

from llm.providers.groq_provider import GroqProvider
from llm.providers.gemini_provider import GeminiProvider

from llm.retry_provider import RetryingLLMProvider
from llm.fallback_provider import FallbackLLMProvider

from agent.intent_classifier import IntentClassifier


def create_llm_client():

    settings = get_settings()

    groq_provider = RetryingLLMProvider(
        provider=GroqProvider(),
        max_attempts=3
    )

    if settings.gemini_api_key:

        gemini_provider = RetryingLLMProvider(
            provider=GeminiProvider(),
            max_attempts=3
        )

        provider = FallbackLLMProvider(
            primary_provider=groq_provider,
            fallback_provider=gemini_provider
        )

    else:
        provider = groq_provider

    return LLMClient(provider=provider)


def create_answer_generator():

    vector_store = ChromaVectorStore()
    embedding_generator = EmbeddingGenerator()

    retriever = VectorRetriever(
        embedding_generator=embedding_generator,
        vector_store=vector_store
    )

    context_assembler = ContextAssembler()

    context_injector = RAGContextInjector(
        retriever=retriever,
        context_assembler=context_assembler
    )

    return RAGAnswerGenerator(
        context_injector=context_injector,
        llm_client=create_llm_client()
    )


def create_research_discovery_service():

    return ResearchDiscoveryService()


def create_intent_classifier():

    return IntentClassifier(
        llm_client=create_llm_client()
    )