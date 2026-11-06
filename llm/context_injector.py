from __future__ import annotations

from typing import TYPE_CHECKING

from llm.prompts.rag_prompt import RAGPromptBuilder

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
        prompt_builder: type[RAGPromptBuilder] = RAGPromptBuilder
    ):
        self.retriever = retriever
        self.context_assembler = context_assembler
        self.prompt_builder = prompt_builder

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

        # Step 2: Assemble retrieved chunks into context
        context = self.context_assembler.assemble(retrieved_chunks)

        # Step 3: Construct the RAG messages
        messages = self.prompt_builder.build_messages(
            context=context,
            question=question
        )

        return {
            "messages": messages,
            "context": context,
            "retrieved_chunks": retrieved_chunks
        }