from llm.providers.base import LLMProvider, LLMProviderError


class FallbackLLMProvider(LLMProvider):

    def __init__(
        self,
        primary_provider: LLMProvider,
        fallback_provider: LLMProvider
    ):
        if not isinstance(primary_provider, LLMProvider):
            raise TypeError("Primary provider must implement LLMProvider.")

        if not isinstance(fallback_provider, LLMProvider):
            raise TypeError("Fallback provider must implement LLMProvider.")

        self.primary_provider = primary_provider
        self.fallback_provider = fallback_provider

    def generate(
        self,
        messages: list[dict],
        max_completion_tokens: int = 1024
    ) -> str:

        try:
            return self.primary_provider.generate(
                messages=messages,
                max_completion_tokens=max_completion_tokens
            )

        except LLMProviderError as primary_error:

            try:
                return self.fallback_provider.generate(
                    messages=messages,
                    max_completion_tokens=max_completion_tokens
                )

            except LLMProviderError as fallback_error:
                raise LLMProviderError(
                    "Both primary and fallback LLM providers failed. "
                    f"Primary error: {primary_error}. "
                    f"Fallback error: {fallback_error}."
                ) from fallback_error