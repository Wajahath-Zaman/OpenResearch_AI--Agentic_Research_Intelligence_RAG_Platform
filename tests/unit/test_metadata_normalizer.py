import pytest

from ingestion.metadata_normalizer import (
    reconstruct_abstract,
    normalize_doi,
    normalize_openalex_id,
    normalize_work
)


def test_reconstruct_abstract():

    inverted_index = {
        "Attention": [0],
        "is": [1],
        "all": [2],
        "you": [3],
        "need": [4]
    }

    result = reconstruct_abstract(inverted_index)

    assert result == "Attention is all you need"


def test_reconstruct_abstract_unordered_index():

    inverted_index = {
        "need": [4],
        "Attention": [0],
        "you": [3],
        "all": [2],
        "is": [1]
    }

    result = reconstruct_abstract(inverted_index)

    assert result == "Attention is all you need"


def test_reconstruct_abstract_missing_data():

    assert reconstruct_abstract(None) is None
    assert reconstruct_abstract({}) is None


def test_normalize_doi():

    doi = "https://doi.org/10.48550/arXiv.1706.03762"

    result = normalize_doi(doi)

    assert result == "10.48550/arXiv.1706.03762"


def test_normalize_openalex_id():

    openalex_id = "https://openalex.org/W2741809807"

    result = normalize_openalex_id(openalex_id)

    assert result == "W2741809807"


def test_normalize_work():

    work = {
        "id": "https://openalex.org/W123",
        "title": "Attention Is All You Need",
        "publication_year": 2017,
        "doi": "https://doi.org/10.48550/example",
        "authorships": [
            {
                "author": {
                    "display_name": "Ashish Vaswani"
                }
            },
            {
                "author": {
                    "display_name": "Noam Shazeer"
                }
            }
        ],
        "abstract_inverted_index": {
            "Attention": [0],
            "is": [1],
            "all": [2],
            "you": [3],
            "need": [4]
        },
        "best_oa_location": {
            "landing_page_url": "https://example.com/paper"
        }
    }

    result = normalize_work(work)

    assert result["openalex_id"] == "W123"

    assert result["title"] == "Attention Is All You Need"

    assert result["authors"] == [
        "Ashish Vaswani",
        "Noam Shazeer"
    ]

    assert result["abstract"] == "Attention is all you need"

    assert result["publication_year"] == 2017

    assert result["doi"] == "10.48550/example"

    assert result["source_url"] == "https://example.com/paper"


def test_normalize_work_missing_optional_fields():

    work = {
        "id": "https://openalex.org/W456",
        "title": "Research Paper Without Optional Metadata"
    }

    result = normalize_work(work)

    assert result["openalex_id"] == "W456"
    assert result["authors"] == []
    assert result["abstract"] is None
    assert result["publication_year"] is None
    assert result["doi"] is None
    assert result["source_url"] is None


def test_normalize_work_missing_title():

    work = {
        "id": "https://openalex.org/W789"
    }

    with pytest.raises(ValueError):

        normalize_work(work)


def test_normalize_work_invalid_input():

    with pytest.raises(ValueError):

        normalize_work("invalid input")