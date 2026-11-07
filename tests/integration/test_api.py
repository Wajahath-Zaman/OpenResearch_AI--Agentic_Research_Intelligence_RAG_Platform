import pytest

from unittest.mock import Mock, patch
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/api/v1/health")

    assert response.status_code == 200


def test_list_documents(client):
    with patch(
        "app.api.routes.documents.DocumentRepository"
    ) as mock_repository:

        mock_repository.return_value.get_all_documents.return_value = []

        response = client.get("/api/v1/documents/")

    assert response.status_code == 200
    assert response.json() == []


def test_get_missing_document(client):
    from uuid import uuid4

    with patch(
        "app.api.routes.documents.DocumentRepository"
    ) as mock_repository:

        mock_repository.return_value.get_document_by_id.return_value = None

        response = client.get(
            f"/api/v1/documents/{uuid4()}"
        )

    assert response.status_code == 404


def test_research_discovery_validation(client):
    response = client.post(
        "/api/v1/research/discover",
        json={
            "query": "",
            "per_page": 10,
            "page": 1
        }
    )

    assert response.status_code == 422


def test_chat_validation(client):
    response = client.post(
        "/api/v1/chat/",
        json={"query": ""}
    )

    assert response.status_code == 422


def test_comparison_duplicate_document_ids(client):
    document_id = "0b189b6b-3d42-4163-9c34-455c3da8152d"

    response = client.post(
        "/api/v1/research/compare",
        json={
            "query": "Compare the methodologies.",
            "document_ids": [document_id, document_id]
        }
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Duplicate document IDs are not allowed."
    )


def test_comparison_invalid_document_id(client):
    response = client.post(
        "/api/v1/research/compare",
        json={
            "query": "Compare the methodologies.",
            "document_ids": ["invalid-id", "another-invalid-id"]
        }
    )

    assert response.status_code == 400