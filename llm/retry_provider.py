import time

from llm.providers.base import (
    LLMProvider,
    LLMProviderError,
    RetryableLLMProviderError
)


class RetryingLLMProvider(LLMProvider):

    def __init__(
        self,
        provider: LLMProvider,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 8.0,
        sleep_func=None
    ):

        if not isinstance(provider, LLMProvider):
            raise TypeError("Provider must implement LLMProvider.")

        if (
            not isinstance(max_attempts, int)
            or isinstance(max_attempts, bool)
            or max_attempts < 1
        ):
            raise ValueError("Maximum attempts must be a positive integer.")

        if initial_delay < 0 or max_delay < 0:
            raise ValueError("Retry delays cannot be negative.")

        if initial_delay > max_delay:
            raise ValueError(
                "Initial delay cannot exceed maximum delay."
            )

        self.provider = provider
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.sleep_func = sleep_func or time.sleep

    def generate(
        self,
        messages: list[dict],
        max_completion_tokens: int = 1024
    ) -> str:

        for attempt in range(self.max_attempts):

            try:
                return self.provider.generate(
                    messages=messages,
                    max_completion_tokens=max_completion_tokens
                )

            except RetryableLLMProviderError:

                if attempt == self.max_attempts - 1:
                    raise

                delay = min(
                    self.initial_delay * (2 ** attempt),
                    self.max_delay
                )

                self.sleep_func(delay)

        raise LLMProviderError(
            "LLM provider failed after all retry attempts."
        )