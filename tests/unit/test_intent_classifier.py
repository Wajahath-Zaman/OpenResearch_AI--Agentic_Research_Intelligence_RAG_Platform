import pytest
from unittest.mock import Mock

from agent.intent_classifier import IntentClassifier


@pytest.mark.parametrize(
    "query, expected",
    [
        ("Explain the uploaded paper", "rag"),
        ("Find recent papers about CNNs", "research"),
        ("Compare two papers about transformers", "comparison")
    ]
)
def test_intent_classification(query, expected):

    mock_client = Mock()
    mock_client.generate.return_value = expected

    classifier = IntentClassifier(mock_client)

    result = classifier.classify(query)

    assert result == expected
    mock_client.generate.assert_called_once()


def test_invalid_intent():

    mock_client = Mock()
    mock_client.generate.return_value = "unknown"

    classifier = IntentClassifier(mock_client)

    with pytest.raises(ValueError):
        classifier.classify("Explain transformers")


def test_empty_query():

    classifier = IntentClassifier(Mock())

    with pytest.raises(ValueError):
        classifier.classify(" ")