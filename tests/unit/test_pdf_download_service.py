import uuid

import pytest

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from database.models import DocumentStatus
from ingestion.pdf_download_service import (
    PDFDownloadService,
    PDFDownloadError
)


class MockResponse:

    def __init__(
        self,
        content=b"%PDF-1.4\nTest PDF content",
        content_type="application/pdf"
    ):
        self.content = content
        self.headers = {"Content-Type": content_type}
        self.closed = False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        yield self.content

    def close(self):
        self.closed = True


@pytest.fixture
def setup_service(tmp_path):

    document_id = uuid.uuid4()

    document = SimpleNamespace(
        id=document_id,
        pdf_url="https://example.com/paper.pdf",
        file_path=None,
        status=DocumentStatus.DISCOVERED
    )

    repository = Mock()
    repository.get_document_by_id.return_value = document

    repository.update_document_file_path.side_effect = (
        lambda document_id, file_path: setattr(
            document,
            "file_path",
            file_path
        ) or setattr(
            document,
            "status",
            DocumentStatus.DOWNLOADED
        ) or document
    )

    service = PDFDownloadService(repository=repository)
    service.raw_directory = tmp_path

    return service, repository, document


def test_successful_pdf_download(setup_service):

    service, repository, document = setup_service

    response = MockResponse()
    service.session.get = Mock(return_value=response)

    result = service.download_pdf(document.id)

    assert result["status"] == "downloaded"
    assert result["message"] == "PDF downloaded successfully."

    assert Path(result["file_path"]).exists()
    assert document.status == DocumentStatus.DOWNLOADED

    repository.update_document_file_path.assert_called_once()


def test_document_not_found(setup_service):

    service, repository, document = setup_service

    repository.get_document_by_id.return_value = None

    with pytest.raises(PDFDownloadError, match="Document not found"):
        service.download_pdf(document.id)


def test_missing_pdf_url(setup_service):

    service, repository, document = setup_service

    document.pdf_url = None

    with pytest.raises(
        PDFDownloadError,
        match="No openly accessible PDF"
    ):
        service.download_pdf(document.id)


def test_invalid_pdf_response(setup_service):

    service, repository, document = setup_service

    response = MockResponse(content=b"This is not a PDF")

    service.session.get = Mock(return_value=response)

    with pytest.raises(
        PDFDownloadError,
        match="not a valid PDF"
    ):
        service.download_pdf(document.id)

    assert list(service.raw_directory.iterdir()) == []


def test_already_downloaded_pdf(setup_service):

    service, repository, document = setup_service

    existing_file = service.raw_directory / "existing.pdf"
    existing_file.write_bytes(b"%PDF-1.4\nExisting PDF")

    document.file_path = str(existing_file)
    document.status = DocumentStatus.DOWNLOADED

    service.session.get = Mock()

    result = service.download_pdf(document.id)

    assert result["message"] == "PDF already downloaded."
    service.session.get.assert_not_called()
