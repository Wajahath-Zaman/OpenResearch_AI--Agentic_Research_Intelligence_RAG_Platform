import pytest
from unittest.mock import MagicMock

from llm.client import LLMClient
from llm.providers.base import LLMProvider


class MockLLMProvider(LLMProvider):

    def generate(
        self,
        messages: list[dict],
        max_completion_tokens: int = 1024
    ) -> str:

        return "Mock LLM response."


@pytest.fixture
def provider():

    return MockLLMProvider()


@pytest.fixture
def client(provider):

    return LLMClient(provider=provider)


@pytest.fixture
def sample_messages():

    return [
        {
            "role": "user",
            "content": "Explain RAG."
        }
    ]


def test_llm_client_initialization(provider):

    client = LLMClient(provider)

    assert client.provider is provider


def test_llm_client_generates_response(client, sample_messages):

    response = client.generate(sample_messages)

    assert response == "Mock LLM response."


def test_llm_client_delegates_to_provider(
    provider,
    sample_messages
):

    provider.generate = MagicMock(
        return_value="Generated response."
    )

    client = LLMClient(provider)

    client.generate(sample_messages)

    provider.generate.assert_called_once_with(
        messages=sample_messages,
        max_completion_tokens=1024
    )


def test_llm_client_custom_token_limit(
    provider,
    sample_messages
):

    provider.generate = MagicMock(
        return_value="Generated response."
    )

    client = LLMClient(provider)

    client.generate(
        sample_messages,
        max_completion_tokens=512
    )

    provider.generate.assert_called_once_with(
        messages=sample_messages,
        max_completion_tokens=512
    )


def test_invalid_provider_raises_error():

    with pytest.raises(TypeError):

        LLMClient(provider="groq")


def test_provider_interface_requires_generate():

    with pytest.raises(TypeError):

        LLMProvider()