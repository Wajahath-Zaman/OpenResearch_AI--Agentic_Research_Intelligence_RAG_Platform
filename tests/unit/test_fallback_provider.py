import pytest
from unittest.mock import MagicMock

from llm.fallback_provider import FallbackLLMProvider
from llm.providers.base import LLMProvider, LLMProviderError


class MockLLMProvider(LLMProvider):

    def __init__(self, response="Mock response.", error=None):
        self.response = response
        self.error = error

    def generate(
        self,
        messages: list[dict],
        max_completion_tokens: int = 1024
    ) -> str:

        if self.error:
            raise self.error

        return self.response


@pytest.fixture
def messages():
    return [
        {"role": "user", "content": "Explain RAG."}
    ]


@pytest.fixture
def primary_provider():
    return MockLLMProvider(response="Groq response.")


@pytest.fixture
def fallback_provider():
    return MockLLMProvider(response="Gemini response.")


@pytest.fixture
def fallback(primary_provider, fallback_provider):
    return FallbackLLMProvider(
        primary_provider=primary_provider,
        fallback_provider=fallback_provider
    )


def test_fallback_provider_implements_interface(fallback):
    assert isinstance(fallback, LLMProvider)


def test_primary_provider_success(fallback, messages):
    response = fallback.generate(messages)

    assert response == "Groq response."


def test_fallback_provider_not_called_on_primary_success(
    primary_provider,
    fallback_provider,
    messages
):
    fallback_provider.generate = MagicMock()

    fallback = FallbackLLMProvider(
        primary_provider,
        fallback_provider
    )

    fallback.generate(messages)

    fallback_provider.generate.assert_not_called()


def test_fallback_activates_when_primary_fails(
    messages,
    fallback_provider
):
    primary = MockLLMProvider(
        error=LLMProviderError("Groq unavailable.")
    )

    fallback = FallbackLLMProvider(
        primary,
        fallback_provider
    )

    response = fallback.generate(messages)

    assert response == "Gemini response."


def test_fallback_receives_same_messages(messages):
    primary = MockLLMProvider(
        error=LLMProviderError("Groq unavailable.")
    )

    secondary = MockLLMProvider()
    secondary.generate = MagicMock(return_value="Gemini response.")

    fallback = FallbackLLMProvider(primary, secondary)

    fallback.generate(messages)

    secondary.generate.assert_called_once_with(
        messages=messages,
        max_completion_tokens=1024
    )


def test_custom_token_limit_is_preserved(messages):
    primary = MockLLMProvider(
        error=LLMProviderError("Groq unavailable.")
    )

    secondary = MockLLMProvider()
    secondary.generate = MagicMock(return_value="Gemini response.")

    fallback = FallbackLLMProvider(primary, secondary)

    fallback.generate(messages, max_completion_tokens=512)

    secondary.generate.assert_called_once_with(
        messages=messages,
        max_completion_tokens=512
    )


def test_both_provider_failures_raise_error(messages):
    primary = MockLLMProvider(
        error=LLMProviderError("Groq unavailable.")
    )

    secondary = MockLLMProvider(
        error=LLMProviderError("Gemini unavailable.")
    )

    fallback = FallbackLLMProvider(primary, secondary)

    with pytest.raises(LLMProviderError) as exc_info:
        fallback.generate(messages)

    assert "Both primary and fallback" in str(exc_info.value)
    assert "Groq unavailable" in str(exc_info.value)
    assert "Gemini unavailable" in str(exc_info.value)


def test_invalid_input_does_not_trigger_fallback():
    primary = MockLLMProvider()
    secondary = MockLLMProvider()
    primary.generate = MagicMock(side_effect=ValueError("Invalid input."))
    secondary.generate = MagicMock()

    fallback = FallbackLLMProvider(primary, secondary)

    with pytest.raises(ValueError):
        fallback.generate([])

    secondary.generate.assert_not_called()


def test_invalid_primary_provider_raises_error(fallback_provider):
    with pytest.raises(TypeError):
        FallbackLLMProvider("Groq", fallback_provider)


def test_invalid_fallback_provider_raises_error(primary_provider):
    with pytest.raises(TypeError):
        FallbackLLMProvider(primary_provider, "Gemini")