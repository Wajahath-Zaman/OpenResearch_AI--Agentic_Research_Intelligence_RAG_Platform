import pytest

from ingestion.discovery_service import ResearchDiscoveryService
from ingestion.research_discovery import OpenAlexAPIError


class MockOpenAlexClient:

    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error

    def search_works(self, query, per_page=10, page=1):

        if self.error:
            raise self.error

        return self.response


@pytest.fixture
def sample_work():

    return {
        "id": "https://openalex.org/W123",
        "title": "Attention Is All You Need",
        "publication_year": 2017,
        "doi": "https://doi.org/10.48550/example",
        "authorships": [
            {"author": {"display_name": "Ashish Vaswani"}},
            {"author": {"display_name": "Noam Shazeer"}}
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


def test_successful_discovery(sample_work):

    client = MockOpenAlexClient(
        response={
            "meta": {"count": 100},
            "results": [sample_work]
        }
    )

    service = ResearchDiscoveryService(client)

    result = service.discover("transformer architecture")

    assert result["total_count"] == 100
    assert result["returned_count"] == 1
    assert result["skipped_count"] == 0
    assert result["results"][0]["title"] == "Attention Is All You Need"


def test_empty_search_results():

    client = MockOpenAlexClient(
        response={
            "meta": {"count": 0},
            "results": []
        }
    )

    service = ResearchDiscoveryService(client)

    result = service.discover("unknown research topic")

    assert result["returned_count"] == 0
    assert result["results"] == []
    assert result["skipped_count"] == 0


def test_malformed_paper_is_skipped(sample_work):

    invalid_work = {
        "id": "https://openalex.org/W456",
        "title": None
    }

    client = MockOpenAlexClient(
        response={
            "meta": {"count": 2},
            "results": [sample_work, invalid_work]
        }
    )

    service = ResearchDiscoveryService(client)

    result = service.discover("deep learning")

    assert result["returned_count"] == 1
    assert result["skipped_count"] == 1


def test_api_failure_is_propagated():

    client = MockOpenAlexClient(
        error=OpenAlexAPIError("OpenAlex unavailable")
    )

    service = ResearchDiscoveryService(client)

    with pytest.raises(OpenAlexAPIError):
        service.discover("machine learning")


def test_pagination_parameters(sample_work):

    client = MockOpenAlexClient(
        response={
            "meta": {"count": 100},
            "results": [sample_work]
        }
    )

    service = ResearchDiscoveryService(client)

    result = service.discover(
        query="artificial intelligence",
        per_page=20,
        page=3
    )

    assert result["page"] == 3
    assert result["per_page"] == 20


class MockDocumentRepository:

    def __init__(self):
        self.documents = []
        self.create_documents = []

    def get_document_by_doi(self, doi):

        for document in self.documents:

            if document.get("doi") == doi:
                return document

        return None

    def get_document_by_openalex_id(self, openalex_id):

        for document in self.documents:
            if document.get("openalex_id") == openalex_id:
                return document

        return None

    def create_document(self, **data):

        document = data.copy()

        self.documents.append(document)
        self.create_documents.append(document)

        return document


def test_discover_and_store_success(sample_work):

    client = MockOpenAlexClient(
        response={
            "meta": {"count": 1},
            "results": [sample_work]
        }
    )

    repository = MockDocumentRepository()

    service = ResearchDiscoveryService(
        client=client,
        repository=repository
    )

    result = service.discover_and_store("transformers")

    assert result["saved_count"] == 1
    assert result["duplicate_count"] == 0
    assert len(repository.create_documents) == 1



def test_duplicate_doi_is_not_inserted(sample_work):

    client = MockOpenAlexClient(
        response={
            "meta": {"count": 1},
            "results": [sample_work]
        }
    )

    repository = MockDocumentRepository()

    repository.documents.append({
        "doi": "10.48550/example",
        "openalex_id": "W999"
    })

    service = ResearchDiscoveryService(
        client=client,
        repository=repository
    )

    result = service.discover_and_store("transformers")

    assert result["saved_count"] == 0
    assert result["duplicate_count"] == 1

def test_duplicate_openalex_id_is_not_inserted(sample_work):

    sample_work["doi"] = None

    client = MockOpenAlexClient(
        response={
            "meta": {"count": 1},
            "results": [sample_work]
        }
    )

    repository = MockDocumentRepository()

    repository.documents.append({
        "doi": None,
        "openalex_id": "W123"
    })

    service = ResearchDiscoveryService(
        client=client,
        repository=repository
    )

    result = service.discover_and_store("transformers")

    assert result["saved_count"] == 0
    assert result["duplicate_count"] == 1

def test_duplicate_openalex_id_is_not_inserted(sample_work):

    sample_work["doi"] = None

    client = MockOpenAlexClient(
        response={
            "meta": {"count": 1},
            "results": [sample_work]
        }
    )

    repository = MockDocumentRepository()

    repository.documents.append({
        "doi": None,
        "openalex_id": "W123"
    })

    service = ResearchDiscoveryService(
        client=client,
        repository=repository
    )

    result = service.discover_and_store("transformers")

    assert result["saved_count"] == 0
    assert result["duplicate_count"] == 1

def test_repository_required_for_persistence(sample_work):

    client = MockOpenAlexClient(
        response={
            "meta": {"count": 1},
            "results": [sample_work]
        }
    )

    service = ResearchDiscoveryService(client=client)

    with pytest.raises(RuntimeError):
        service.discover_and_store("transformers")