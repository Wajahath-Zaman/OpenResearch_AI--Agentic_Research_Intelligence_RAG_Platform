import pytest
from unittest.mock import MagicMock

from llm.retry_provider import RetryingLLMProvider

from llm.providers.base import (
    LLMProvider,
    LLMProviderError,
    RetryableLLMProviderError
)


class MockLLMProvider(LLMProvider):

    def __init__(self, response="Mock response."):
        self.response = response

    def generate(
        self,
        messages: list[dict],
        max_completion_tokens: int = 1024
    ) -> str:
        return self.response


@pytest.fixture
def provider():
    return MockLLMProvider()


@pytest.fixture
def messages():
    return [{"role": "user", "content": "Explain RAG."}]


def test_retry_provider_implements_interface(provider):
    retry_provider = RetryingLLMProvider(provider)

    assert isinstance(retry_provider, LLMProvider)


def test_successful_request_returns_response(provider, messages):
    retry_provider = RetryingLLMProvider(provider)

    response = retry_provider.generate(messages)

    assert response == "Mock response."


def test_successful_request_does_not_retry(provider, messages):
    provider.generate = MagicMock(return_value="Success.")

    sleep = MagicMock()

    retry_provider = RetryingLLMProvider(
        provider,
        sleep_func=sleep
    )

    retry_provider.generate(messages)

    provider.generate.assert_called_once()
    sleep.assert_not_called()


def test_retry_occurs_after_transient_failure(provider, messages):
    provider.generate = MagicMock(
        side_effect=[
            RetryableLLMProviderError("Temporary failure."),
            "Success."
        ]
    )

    sleep = MagicMock()

    retry_provider = RetryingLLMProvider(
        provider,
        sleep_func=sleep
    )

    response = retry_provider.generate(messages)

    assert response == "Success."
    assert provider.generate.call_count == 2


def test_exponential_backoff_delays(provider, messages):
    provider.generate = MagicMock(
        side_effect=[
            RetryableLLMProviderError("Failure 1."),
            RetryableLLMProviderError("Failure 2."),
            "Success."
        ]
    )

    sleep = MagicMock()

    retry_provider = RetryingLLMProvider(
        provider,
        max_attempts=3,
        initial_delay=1,
        max_delay=8,
        sleep_func=sleep
    )

    retry_provider.generate(messages)

    assert [call.args[0] for call in sleep.call_args_list] == [1, 2]


def test_max_attempts_are_respected(provider, messages):
    provider.generate = MagicMock(
        side_effect=RetryableLLMProviderError("Temporary failure.")
    )

    retry_provider = RetryingLLMProvider(
        provider,
        max_attempts=3,
        sleep_func=MagicMock()
    )

    with pytest.raises(RetryableLLMProviderError):
        retry_provider.generate(messages)

    assert provider.generate.call_count == 3


def test_non_retryable_error_is_not_retried(provider, messages):
    provider.generate = MagicMock(
        side_effect=LLMProviderError("Invalid API key.")
    )

    sleep = MagicMock()

    retry_provider = RetryingLLMProvider(
        provider,
        sleep_func=sleep
    )

    with pytest.raises(LLMProviderError):
        retry_provider.generate(messages)

    provider.generate.assert_called_once()
    sleep.assert_not_called()


def test_invalid_provider_raises_error():
    with pytest.raises(TypeError):
        RetryingLLMProvider("Groq")


def test_invalid_max_attempts_raises_error(provider):
    with pytest.raises(ValueError):
        RetryingLLMProvider(provider, max_attempts=0)


def test_negative_delay_raises_error(provider):
    with pytest.raises(ValueError):
        RetryingLLMProvider(provider, initial_delay=-1)


def test_initial_delay_cannot_exceed_max_delay(provider):
    with pytest.raises(ValueError):
        RetryingLLMProvider(
            provider,
            initial_delay=10,
            max_delay=5
        )