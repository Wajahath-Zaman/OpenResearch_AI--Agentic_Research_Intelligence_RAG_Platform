from __future__ import annotations

from typing import TYPE_CHECKING

from llm.prompts.rag_prompt import RAGPromptBuilder
from guardrails.context_guardrail import ContextGuardrail

if TYPE_CHECKING:
    from retrieval.vector_retriever import VectorRetriever
    from retrieval.context_assembler import ContextAssembler


class NoRetrievedContextError(Exception):
    """Raised when retrieval returns no relevant chunks."""


class RAGContextInjector:

    def __init__(
        self,
        retriever: VectorRetriever,
        context_assembler: ContextAssembler,
        prompt_builder: type[RAGPromptBuilder] = RAGPromptBuilder,
        context_guardrail: ContextGuardrail | None = None
    ):
        self.retriever = retriever
        self.context_assembler = context_assembler
        self.prompt_builder = prompt_builder
        self.context_guardrail = context_guardrail or ContextGuardrail()

    def prepare(
        self,
        question: str,
        n_results: int | None = None,
        filters: dict | None = None
    ) -> dict:

        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must be a non-empty string.")

        # Step 1: Retrieve relevant research chunks
        retrieved_chunks = self.retriever.search(
            query=question,
            n_results=n_results,
            filters=filters
        )

        if not retrieved_chunks:
            raise NoRetrievedContextError(
                "No research context was retrieved for this question."
            )

        # Step 2: Inspect retrieved chunks for potential prompt injection
        guardrail_result = self.context_guardrail.inspect(retrieved_chunks)

        safe_chunks = guardrail_result["safe_chunks"]
        flagged_chunks = guardrail_result["flagged_chunks"]

        if not safe_chunks:
            raise NoRetrievedContextError(
                "All retrieved research chunks were flagged by the context guardrail."
            )

        # Step 3: Assemble only the safe chunks
        context = self.context_assembler.assemble(safe_chunks)

        # Step 4: Construct the RAG messages
        messages = self.prompt_builder.build_messages(
            context=context,
            question=question
        )

        return {
            "messages": messages,
            "context": context,
            "retrieved_chunks": safe_chunks,
            "flagged_chunks": flagged_chunks
        }