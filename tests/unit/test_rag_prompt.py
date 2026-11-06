import pytest

from llm.prompts.rag_prompt import RAGPromptBuilder


@pytest.fixture
def sample_context():
    return "CNNs are widely used for medical image classification."


@pytest.fixture
def sample_question():
    return "How are CNNs used in medical imaging?"


def test_build_messages_returns_list(sample_context, sample_question):
    messages = RAGPromptBuilder.build_messages(
        sample_context,
        sample_question
    )

    assert isinstance(messages, list)


def test_build_messages_contains_two_messages(sample_context, sample_question):
    messages = RAGPromptBuilder.build_messages(
        sample_context,
        sample_question
    )

    assert len(messages) == 2


def test_message_roles_are_correct(sample_context, sample_question):
    messages = RAGPromptBuilder.build_messages(
        sample_context,
        sample_question
    )

    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


def test_context_is_included(sample_context, sample_question):
    messages = RAGPromptBuilder.build_messages(
        sample_context,
        sample_question
    )

    assert sample_context in messages[1]["content"]


def test_question_is_included(sample_context, sample_question):
    messages = RAGPromptBuilder.build_messages(
        sample_context,
        sample_question
    )

    assert sample_question in messages[1]["content"]


def test_empty_context_raises_error(sample_question):
    with pytest.raises(ValueError):
        RAGPromptBuilder.build_messages("", sample_question)


def test_empty_question_raises_error(sample_context):
    with pytest.raises(ValueError):
        RAGPromptBuilder.build_messages(sample_context, "")


def test_invalid_context_type_raises_error(sample_question):
    with pytest.raises(ValueError):
        RAGPromptBuilder.build_messages(None, sample_question)


def test_invalid_question_type_raises_error(sample_context):
    with pytest.raises(ValueError):
        RAGPromptBuilder.build_messages(sample_context, 123)