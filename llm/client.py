from llm.providers.base import LLMProvider


class LLMClient:

    def __init__(self, provider: LLMProvider):

        if not isinstance(provider, LLMProvider):
            raise TypeError(
                "Provider must implement the LLMProvider interface."
            )

        self.provider = provider

    def generate(
        self,
        messages: list[dict],
        max_completion_tokens: int = 1024
    ) -> str:
        """
        Generate a response through the configured LLM provider.
        """

        return self.provider.generate(
            messages=messages,
            max_completion_tokens=max_completion_tokens
        )