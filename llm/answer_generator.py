from llm.client import LLMClient
from llm.context_injector import RAGContextInjector
from llm.citation_formatter import CitationFormatter

class RAGAnswerGenerator:

    def __init__(
        self,
        context_injector: RAGContextInjector,
        llm_client: LLMClient,
        citation_formatter: CitationFormatter | None = None
    ):
        self.context_injector = context_injector
        self.llm_client = llm_client
        self.citation_formatter = citation_formatter or CitationFormatter()

    def generate_answer(
        self,
        question: str,
        n_results: int | None = None,
        filters: dict | None = None
    ) -> dict:

        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must be a non-empty string.")

        # Step 1: Retrieve context and construct LLM messages
        rag_data = self.context_injector.prepare(
            question=question,
            n_results=n_results,
            filters=filters
        )

        # Step 2: Generate an answer using the LLM
        answer = self.llm_client.generate(
            messages=rag_data["messages"]
        )

        # Step 3: Format source citations
        citations = self.citation_formatter.format_sources(
            rag_data["retrieved_chunks"]
        )

        # Step 4: Return the answer and supporting information
        return {
            "question": question,
            "answer": answer,
            "context": rag_data["context"],
            "retrieved_chunks": rag_data["retrieved_chunks"],
            "citations": citations
        }