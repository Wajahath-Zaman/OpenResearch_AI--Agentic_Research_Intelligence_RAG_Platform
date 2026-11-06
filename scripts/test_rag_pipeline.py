from config.settings import get_settings

from ingestion.embedding_generator import EmbeddingGenerator

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


def main():

    settings = get_settings()

    # 1. Initialize vector database
    vector_store = ChromaVectorStore()

    total_chunks = vector_store.count()

    print(f"\nAvailable document chunks: {total_chunks}")

    if total_chunks == 0:
        print("No documents found. Ingest a PDF first.")
        return

    # 2. Initialize embedding model
    embedding_generator = EmbeddingGenerator()

    # 3. Initialize retriever
    retriever = VectorRetriever(
        embedding_generator=embedding_generator,
        vector_store=vector_store
    )

    # 4. Initialize context assembler
    context_assembler = ContextAssembler()

    # 5. Initialize RAG context injector
    context_injector = RAGContextInjector(
        retriever=retriever,
        context_assembler=context_assembler
    )

    # 6. Initialize Groq with retry
    groq_provider = RetryingLLMProvider(
        provider=GroqProvider(),
        max_attempts=3
    )

    # 7. Configure Gemini fallback if available
    if settings.gemini_api_key:

        gemini_provider = RetryingLLMProvider(
            provider=GeminiProvider(),
            max_attempts=3
        )

        provider = FallbackLLMProvider(
            primary_provider=groq_provider,
            fallback_provider=gemini_provider
        )

        print("Provider configuration: Groq + Gemini fallback")

    else:

        provider = groq_provider

        print("Provider configuration: Groq only")

    # 8. Initialize LLM client
    llm_client = LLMClient(provider=provider)

    # 9. Initialize answer generator
    answer_generator = RAGAnswerGenerator(
        context_injector=context_injector,
        llm_client=llm_client
    )

    # 10. Ask a question
    question = input("\nEnter your research question: ").strip()

    if not question:
        print("Question cannot be empty.")
        return

    print("\nGenerating answer...\n")

    result = answer_generator.generate_answer(
        question=question,
        n_results=5
    )

    # 11. Display answer
    print("=" * 60)
    print("GENERATED ANSWER")
    print("=" * 60)

    print(result["answer"])

    # 12. Display retrieved sources
    print("\n" + "=" * 60)
    print("CITATIONS")
    print("=" * 60)

    for citation in result["citations"]:

        print(
            f"\n[{citation['citation_id']}] "
            f"{citation.get('title') or 'Untitled'}"
        )

        print(f"Document ID: {citation['document_id']}")
        print(f"Page: {citation['page_number']}")

        if citation.get("source_path"):
            print(f"Source: {citation['source_path']}")

    print("\n" + "=" * 60)
    print("PIPELINE EXECUTION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()