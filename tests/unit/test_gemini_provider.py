import pytest
from unittest.mock import MagicMock

from llm.providers.gemini_provider import GeminiProvider
from llm.providers.base import LLMProvider, LLMProviderError


import httpx

from google.genai import errors

from llm.providers.base import RetryableLLMProviderError

@pytest.fixture
def mock_client():
    client = MagicMock()
    client.models.generate_content.return_value.text = "Gemini response."

    return client


@pytest.fixture
def provider(mock_client):
    return GeminiProvider(client=mock_client)


@pytest.fixture
def sample_messages():
    return [
        {
            "role": "system",
            "content": "You are a research assistant."
        },
        {
            "role": "user",
            "content": "Explain RAG."
        }
    ]


def test_gemini_provider_implements_interface(provider):
    assert isinstance(provider, LLMProvider)


def test_gemini_provider_generates_response(provider, sample_messages):
    response = provider.generate(sample_messages)

    assert response == "Gemini response."


def test_gemini_provider_strips_response(provider, sample_messages):
    provider.client.models.generate_content.return_value.text = (
        "  Gemini response.  "
    )

    response = provider.generate(sample_messages)

    assert response == "Gemini response."


def test_gemini_provider_passes_correct_model(
    provider,
    sample_messages
):
    provider.generate(sample_messages)

    call_kwargs = (
        provider.client.models.generate_content.call_args.kwargs
    )

    assert call_kwargs["model"] == provider.model


def test_gemini_provider_maps_assistant_role(
    provider,
    sample_messages
):
    messages = sample_messages + [
        {
            "role": "assistant",
            "content": "Previous response."
        }
    ]

    provider.generate(messages)

    call_kwargs = (
        provider.client.models.generate_content.call_args.kwargs
    )

    assert call_kwargs["contents"][1]["role"] == "model"


def test_gemini_provider_passes_system_instruction(
    provider,
    sample_messages
):
    provider.generate(sample_messages)

    call_kwargs = (
        provider.client.models.generate_content.call_args.kwargs
    )

    assert (
        call_kwargs["config"].system_instruction
        == "You are a research assistant."
    )


def test_gemini_provider_uses_custom_token_limit(
    provider,
    sample_messages
):
    provider.generate(
        sample_messages,
        max_completion_tokens=512
    )

    call_kwargs = (
        provider.client.models.generate_content.call_args.kwargs
    )

    assert call_kwargs["config"].max_output_tokens == 512


def test_empty_messages_raise_error(provider):
    with pytest.raises(ValueError):
        provider.generate([])


def test_invalid_token_limit_raises_error(
    provider,
    sample_messages
):
    with pytest.raises(ValueError):
        provider.generate(
            sample_messages,
            max_completion_tokens=0
        )


def test_empty_response_raises_error(provider, sample_messages):
    provider.client.models.generate_content.return_value.text = " "

    with pytest.raises(LLMProviderError):
        provider.generate(sample_messages)


def test_unsupported_role_raises_error(provider):
    messages = [
        {
            "role": "developer",
            "content": "Some instructions."
        }
    ]

    with pytest.raises(ValueError):
        provider.generate(messages)


def test_rate_limit_error_is_retryable(provider, sample_messages):
    error = errors.APIError(
        code=429,
        response_json={
            "error": {
                "message": "Rate limit exceeded."
            }
        }
    )

    provider.client.models.generate_content.side_effect = error

    with pytest.raises(RetryableLLMProviderError):
        provider.generate(sample_messages)


def test_authentication_error_is_not_retryable(
    provider,
    sample_messages
):
    error = errors.APIError(
        code=401,
        response_json={
            "error": {
                "message": "Invalid API key."
            }
        }
    )

    provider.client.models.generate_content.side_effect = error

    with pytest.raises(LLMProviderError) as exc_info:
        provider.generate(sample_messages)

    assert not isinstance(
        exc_info.value,
        RetryableLLMProviderError
    )


def test_network_error_is_retryable(provider, sample_messages):
    error = httpx.ConnectError("Connection unavailable.")

    provider.client.models.generate_content.side_effect = error

    with pytest.raises(RetryableLLMProviderError):
        provider.generate(sample_messages)