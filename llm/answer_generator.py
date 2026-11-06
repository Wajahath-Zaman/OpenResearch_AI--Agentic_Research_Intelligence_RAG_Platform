from llm.client import LLMClient
from llm.context_injector import RAGContextInjector
from llm.citation_formatter import CitationFormatter
from guardrails.output_guardrail import OutputGuardrail


class InvalidLLMOutputError(Exception):
    """Raised when an LLM response fails output validation."""


class RAGAnswerGenerator:

    def __init__(
        self,
        context_injector: RAGContextInjector,
        llm_client: LLMClient,
        citation_formatter: CitationFormatter | None = None,
        output_guardrail: OutputGuardrail | None = None
    ):
        self.context_injector = context_injector
        self.llm_client = llm_client
        self.citation_formatter = citation_formatter or CitationFormatter()
        self.output_guardrail = output_guardrail or OutputGuardrail()

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

        # Step 3: Validate the generated answer
        validation = self.output_guardrail.validate(answer)

        if not validation["valid"]:
            raise InvalidLLMOutputError(validation["reason"])

        # Step 4: Format source citations
        citations = self.citation_formatter.format_sources(
            rag_data["retrieved_chunks"]
        )

        # Step 5: Return the answer and supporting information
        return {
            "question": question,
            "answer": answer,
            "context": rag_data["context"],
            "retrieved_chunks": rag_data["retrieved_chunks"],
            "citations": citations
        }