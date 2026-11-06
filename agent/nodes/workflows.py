from agent.state import AgentState

from agent.dependencies import (
    create_answer_generator,
    create_research_discovery_service
)

from agent.nodes.comparison import PaperComparisonWorkflow

def rag_workflow(
    state: AgentState,
    answer_generator=None
) -> dict:

    try:

        if answer_generator is None:
            answer_generator = create_answer_generator()

        result = answer_generator.generate_answer(
            question=state["query"],
            n_results=5
        )

        return {
            "retrieved_chunks": result["retrieved_chunks"],
            "context": result["context"],
            "answer": result["answer"],
            "citations": result["citations"],
            "error": None
        }

    except Exception as exc:

        return {
            "error": str(exc),
            "answer": ""
        }


def research_workflow(
    state: AgentState,
    discovery_service=None
) -> dict:
    """
    Discover research papers using OpenAlex.
    """

    try:

        if discovery_service is None:
            discovery_service = create_research_discovery_service()

        result = discovery_service.discover(
            query=state["query"],
            per_page=10
        )

        papers = result["results"]

        if not papers:

            answer = (
                f"No research papers found for: {state['query']}"
            )

        else:

            paper_details = []

            for index, paper in enumerate(papers, start=1):

                authors = ", ".join(
                    paper.get("authors") or []
                ) or "Authors unavailable"

                year = paper.get("publication_year") or "Year unavailable"

                doi = paper.get("doi") or "DOI unavailable"

                paper_details.append(
                    f"{index}. {paper['title']}\n"
                    f"   Authors: {authors}\n"
                    f"   Year: {year}\n"
                    f"   DOI: {doi}"
                )

            answer = (
                f"Found {len(papers)} research papers "
                f"for: {state['query']}\n\n"
                + "\n\n".join(paper_details)
            )

        return {
            "research_results": papers,
            "answer": answer,
            "error": None
        }

    except Exception as exc:

        return {
            "research_results": [],
            "answer": "",
            "error": str(exc)
        }


def comparison_workflow(
    state: AgentState,
    comparison_handler=None
) -> dict:

    try:
        if comparison_handler is None:
            comparison_handler = PaperComparisonWorkflow()

        result = comparison_handler.compare(
            query=state["query"],
            document_ids=state["document_ids"]
        )

        return {
            "comparison_results": result["comparison_results"],
            "answer": result["comparison"],
            "citations": [
                citation
                for paper in result["comparison_results"]
                for citation in paper["citations"]
            ],
            "error": None
        }

    except Exception as exc:
        return {
            "comparison_results": [],
            "answer": "",
            "citations": [],
            "error": str(exc)
        }