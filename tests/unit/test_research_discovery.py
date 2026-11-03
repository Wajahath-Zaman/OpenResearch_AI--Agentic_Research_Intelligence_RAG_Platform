import pytest 
import requests


from unittest.mock import Mock

from ingestion.research_discovery import (
    OpenAlexClient,
    OpenAlexAPIError
)


@pytest.fixture
def client():

    return OpenAlexClient()

def test_search_works_success(client):

    mock_response = Mock()

    mock_response.json.return_value = {
        "meta": {
            "count": 1
        },
        "results": [
            {
                "id": "https://openalex.org/W123",
                "title": "Attention Is All You Need."
            }
        ]
    }

    client.session.get = Mock(
        return_value=mock_response
    )

    result = client.search_works(
        query="transformers",
        per_page=10
    )

    assert len(result["results"]) == 1 

    assert (
        result["results"][0]['title'] == "Attention Is All You Need."
    )


def test_search_works_empty_query(client):

    with pytest.raises(ValueError):

        client.search_works("")


def test_search_works_invalid_per_page(client):

    with pytest.raises(ValueError):

        client.search_works(
            query="machine learning",
            per_page=101
        )

def test_search_works_invalid_page(client):

    with pytest.raises(ValueError):

        client.search_works(
            query="machine learning",
            page=0
        )


def test_search_works_http_error(client):

    mock_response = Mock()

    mock_response.raise_for_status.side_effect = (
        requests.HTTPError("404 Not Found")
    )

    client.session.get = Mock(
        return_value=mock_response
    )

    with pytest.raises(OpenAlexAPIError):

        client.search_works("transformers")


def test_search_works_connection_error(client):

    client.session.get = Mock(
        side_effect=requests.ConnectionError(
            "Connection failed"
        )
    )

    with pytest.raises(OpenAlexAPIError):

        client.search_works("transformers")


def test_search_works_invalid_response(client):

    mock_response = Mock()

    mock_response.json.return_value = {
        "meta": {},
        "results": "invalid"
    }

    client.session.get = Mock(
        return_value=mock_response
    )

    with pytest.raises(OpenAlexAPIError):

        client.search_works("transformers")