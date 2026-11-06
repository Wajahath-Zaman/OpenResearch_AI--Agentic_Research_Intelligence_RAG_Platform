from llm.client import LLMClient


class IntentClassifier:

    VALID_INTENTS = {"rag", "research", "comparison"}

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def classify(self, query: str) -> str:

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query cannot be empty.")

        messages = [
            {
                "role": "system",
                "content": """
                Classify the user's query into exactly one category:

                - rag: Questions about uploaded documents or their content.
                - research: Discovering or finding academic research papers.
                - comparison: Comparing two or more research papers.

                Return only one word:
                rag, research, or comparison.
                """
            },
            {
                "role": "user",
                "content": query
            }
        ]

        response = self.llm_client.generate(
            messages=messages,
            max_completion_tokens=10
        )

        intent = response.strip().lower()

        if intent not in self.VALID_INTENTS:
            raise ValueError(f"Invalid intent returned: {intent}")

        return intent