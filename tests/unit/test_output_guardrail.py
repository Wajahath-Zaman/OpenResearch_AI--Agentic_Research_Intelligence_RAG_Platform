import pytest

from guardrails.output_guardrail import OutputGuardrail


@pytest.fixture
def guardrail():
    return OutputGuardrail()


def test_accepts_valid_response(guardrail):
    result = guardrail.validate(
        "CNNs are commonly used for image classification."
    )

    assert result["valid"] is True
    assert result["reason"] is None


def test_rejects_empty_response(guardrail):
    result = guardrail.validate("")

    assert result["valid"] is False


def test_rejects_whitespace_response(guardrail):
    result = guardrail.validate("   ")

    assert result["valid"] is False


def test_rejects_non_string_response(guardrail):
    result = guardrail.validate(123)

    assert result["valid"] is False


def test_rejects_response_exceeding_maximum_length():
    guardrail = OutputGuardrail(max_length=100)

    result = guardrail.validate("A" * 101)

    assert result["valid"] is False
    assert "maximum allowed length" in result["reason"]


def test_accepts_response_at_maximum_length():
    guardrail = OutputGuardrail(max_length=100)

    result = guardrail.validate("A" * 100)

    assert result["valid"] is True


@pytest.mark.parametrize(
    "response",
    [
        "api_key: abc123secret",
        "secret_key = mysecret123",
        "password: example123",
        "Authorization: Bearer abcdefghijklmnopqrstuvwxyz123456"
    ]
)
def test_rejects_potential_credential_exposure(guardrail, response):
    result = guardrail.validate(response)

    assert result["valid"] is False
    assert "sensitive information" in result["reason"]


def test_accepts_normal_research_text(guardrail):
    response = (
        "The API uses an authentication mechanism. "
        "The model achieved 98.5% accuracy."
    )

    result = guardrail.validate(response)

    assert result["valid"] is True