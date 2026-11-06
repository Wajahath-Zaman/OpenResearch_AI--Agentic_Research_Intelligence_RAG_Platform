from typing import TypedDict


class AgentState(TypedDict):

    # Original user query
    query: str

    # Selected workflow
    route: str

    # Retrieved document chunks
    retrieved_chunks: list[dict]

    # Assembled retrieval context
    context: str

    # Papers discovered through OpenAlex
    research_results: list[dict]

    # Messages passed to the LLM
    messages: list[dict]

    # Generated response
    answer: str

    # Source citations
    citations: list[dict]

    # Error information
    error: str | None

    # Documents selected for comparison
    document_ids: list[str]

    # Individual Paper Comparison Results
    comparison_results: list[dict]