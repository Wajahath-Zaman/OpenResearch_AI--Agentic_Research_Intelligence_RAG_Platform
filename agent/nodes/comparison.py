from agent.dependencies import create_answer_generator, create_llm_client


class PaperComparisonWorkflow:

    def __init__(
        self,
        answer_generator=None,
        llm_client=None
    ):
        self.answer_generator = (
            answer_generator or create_answer_generator()
        )

        self.llm_client = llm_client or create_llm_client()

    def compare(
        self,
        query: str,
        document_ids: list[str]
    ) -> dict:

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Comparison query cannot be empty.")

        if not isinstance(document_ids, list) or len(document_ids) < 2:
            raise ValueError(
                "At least two document IDs are required."
            )

        if len(set(document_ids)) != len(document_ids):
            raise ValueError("Duplicate document IDs are not allowed.")

        results = []

        # Stage 1: Analyse each paper independently.
        for index, document_id in enumerate(document_ids, start=1):

            result = self.answer_generator.generate_answer(
                question=(
                    f"Analyse this research paper in relation to: {query}\n"
                    "Extract its research objective, methodology, "
                    "dataset, model architecture, results and limitations. "
                    "Use only evidence from this paper."
                ),
                n_results=5,
                filters={"document_id": document_id}
            )

            results.append({
                "document_id": document_id,
                "paper_number": index,
                "analysis": result["answer"],
                "citations": result["citations"],
                "retrieved_chunks": result["retrieved_chunks"]
            })

        # Stage 2: Synthesize individual analyses.
        synthesis_messages = [
            {
                "role": "system",
                "content": """
                You are an academic research comparison assistant.

                Compare research papers using only the supplied analyses.

                Structure your response using these sections:

                1. Research Objectives
                2. Methodologies
                3. Datasets
                4. Model Architectures
                5. Results and Performance
                6. Limitations
                7. Overall Comparison

                Use a comparison table wherever appropriate.

                Do not invent missing information.
                If information is unavailable, explicitly state that.
                Distinguish reported findings from your interpretation.
                """
            },
            {
                "role": "user",
                "content": (
                    f"User's comparison request:\n{query}\n\n"
                    f"Paper analyses:\n{self._format_analyses(results)}"
                )
            }
        ]

        comparison = self.llm_client.generate(
            messages=synthesis_messages,
            max_completion_tokens=2048
        )

        return {
            "comparison_results": results,
            "comparison": comparison
        }

    @staticmethod
    def _format_analyses(results: list[dict]) -> str:

        sections = []

        for paper in results:

            sections.append(
                f"Paper {paper['paper_number']}\n"
                f"Document ID: {paper['document_id']}\n"
                f"Analysis:\n{paper['analysis']}"
            )

        return "\n\n".join(sections)