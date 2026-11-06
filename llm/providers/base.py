from abc import ABC, abstractmethod


class LLMProviderError(Exception):
    """Base exception for LLM provider failures."""


class LLMProvider(ABC):
    """
    Abstract interface for LLM providers.
    """

    @abstractmethod
    def generate(
        self,
        messages: list[dict],
        max_completion_tokens: int = 1024
    ) -> str:
        """
        Generate a response from an LLM provider.
        """
        pass

class RetryableLLMProviderError(LLMProviderError):
    """Raised when an LLM provider encounters a temporary failure."""