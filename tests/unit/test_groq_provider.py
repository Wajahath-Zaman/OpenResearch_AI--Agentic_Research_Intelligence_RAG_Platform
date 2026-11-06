from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from llm.providers.groq_provider import (
    GroqProvider,
    LLMProviderError
)

import httpx

from groq import RateLimitError, AuthenticationError

from llm.providers.base import RetryableLLMProviderError

@pytest.fixture
def mock_client():

    return MagicMock()


@pytest.fixture
def provider(mock_client):

    return GroqProvider(client=mock_client)


@pytest.fixture
def sample_messages():

    return [
        {
            "role": "system",
            "content": "You are a helpful research assistant."
        },
        {
            "role": "user",
            "content": "Explain RAG."
        }
    ]


def configure_mock_response(mock_client, content):

    mock_client.chat.completions.create.return_value = (
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=content
                    )
                )
            ]
        )
    )


def test_groq_provider_initialization(provider):

    assert provider.client is not None


def test_generate_returns_response(
    provider,
    mock_client,
    sample_messages
):

    configure_mock_response(
        mock_client,
        "RAG combines retrieval with generation."
    )

    result = provider.generate(sample_messages)

    assert result == "RAG combines retrieval with generation."


def test_generate_calls_groq_api(
    provider,
    mock_client,
    sample_messages
):

    configure_mock_response(
        mock_client,
        "Generated response."
    )

    provider.generate(sample_messages)

    mock_client.chat.completions.create.assert_called_once()


def test_generate_uses_configured_model(
    provider,
    mock_client,
    sample_messages
):

    configure_mock_response(
        mock_client,
        "Generated response."
    )

    provider.generate(sample_messages)

    call_kwargs = (
        mock_client.chat.completions.create.call_args.kwargs
    )

    assert call_kwargs["model"] == provider.model


def test_generate_with_custom_token_limit(
    provider,
    mock_client,
    sample_messages
):

    configure_mock_response(
        mock_client,
        "Generated response."
    )

    provider.generate(
        sample_messages,
        max_completion_tokens=512
    )

    call_kwargs = (
        mock_client.chat.completions.create.call_args.kwargs
    )

    assert call_kwargs["max_completion_tokens"] == 512


def test_empty_messages_raise_error(provider):

    with pytest.raises(ValueError):

        provider.generate([])


def test_invalid_messages_type_raises_error(provider):

    with pytest.raises(TypeError):

        provider.generate("invalid")


def test_invalid_token_limit_raises_error(
    provider,
    sample_messages
):

    with pytest.raises(ValueError):

        provider.generate(
            sample_messages,
            max_completion_tokens=0
        )


def test_invalid_message_content_raises_error(provider):

    with pytest.raises(ValueError):

        provider.generate([
            {
                "role": "user",
                "content": None
            }
        ])


def test_empty_response_raises_error(
    provider,
    mock_client,
    sample_messages
):

    configure_mock_response(
        mock_client,
        ""
    )

    with pytest.raises(LLMProviderError):

        provider.generate(sample_messages)


def test_rate_limit_error_is_retryable(provider, sample_messages):
    request = httpx.Request(
        "POST",
        "https://api.groq.com"
    )

    response = httpx.Response(
        429,
        request=request
    )

    error = RateLimitError(
        "Rate limit exceeded.",
        response=response,
        body=None
    )

    provider.client.chat.completions.create.side_effect = error

    with pytest.raises(RetryableLLMProviderError):
        provider.generate(sample_messages)


def test_authentication_error_is_not_retryable(
    provider,
    sample_messages
):
    request = httpx.Request(
        "POST",
        "https://api.groq.com"
    )

    response = httpx.Response(
        401,
        request=request
    )

    error = AuthenticationError(
        "Invalid API key.",
        response=response,
        body=None
    )

    provider.client.chat.completions.create.side_effect = error

    with pytest.raises(LLMProviderError) as exc_info:
        provider.generate(sample_messages)

    assert not isinstance(
        exc_info.value,
        RetryableLLMProviderError
    )