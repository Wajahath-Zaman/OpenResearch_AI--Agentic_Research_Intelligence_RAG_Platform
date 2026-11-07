import uuid
import logging

from pathlib import Path

import requests

from config.settings import PROJECT_ROOT
from database.repositories import DocumentRepository
from database.models import DocumentStatus


logger = logging.getLogger(__name__)


class PDFDownloadError(Exception):
    """Raised when PDF downloading fails."""


class PDFDownloadService:

    def __init__(
        self,
        repository: DocumentRepository,
        timeout: int = 30,
        max_file_size: int = 20 * 1024 * 1024
    ):

        self.repository = repository
        self.timeout = timeout
        self.max_file_size = max_file_size

        self.raw_directory = PROJECT_ROOT / "data" / "raw"
        self.raw_directory.mkdir(parents=True, exist_ok=True)

        self.session = requests.Session()

    def download_pdf(self, document_id: uuid.UUID) -> dict:

        document = self.repository.get_document_by_id(document_id)

        if document is None:
            raise PDFDownloadError("Document not found.")

        # Check whether PDF is already downloaded
        if document.file_path:

            existing_path = Path(document.file_path)

            if existing_path.exists() and existing_path.is_file():
                return {
                    "document_id": str(document.id),
                    "file_path": str(existing_path),
                    "status": document.status.value,
                    "message": "PDF already downloaded."
                }

        # Check PDF availability
        if not document.pdf_url:
            raise PDFDownloadError(
                "No openly accessible PDF is available for this paper."
            )

        # Download PDF
        try:
            response = self.session.get(
                document.pdf_url,
                timeout=self.timeout,
                stream=True,
                headers={"User-Agent": "OpenResearchIntelligenceAssistant/1.0"}
            )

            response.raise_for_status()

            content_type = response.headers.get(
                "Content-Type", ""
            ).lower()

            if "application/pdf" not in content_type:
                raise PDFDownloadError(
                    "The provided URL did not return a PDF file."
                )

            content_length = response.headers.get("Content-Length")

            if content_length and int(content_length) > self.max_file_size:
                raise PDFDownloadError(
                    "PDF exceeds the maximum allowed size of 20 MB."
                )

            filename = f"{document.id}.pdf"
            file_path = self.raw_directory / filename

            downloaded_size = 0

            try:
                with open(file_path, "wb") as pdf_file:

                    for chunk in response.iter_content(chunk_size=8192):

                        if not chunk:
                            continue

                        downloaded_size += len(chunk)

                        if downloaded_size > self.max_file_size:
                            raise PDFDownloadError(
                                "PDF exceeds the maximum allowed size of 20 MB."
                            )

                        pdf_file.write(chunk)

                # Validate PDF signature
                with open(file_path, "rb") as pdf_file:
                    signature = pdf_file.read(5)

                if signature != b"%PDF-":
                    raise PDFDownloadError(
                        "Downloaded file is not a valid PDF."
                    )

            except Exception:
                file_path.unlink(missing_ok=True)
                raise

            # Update database
            updated_document = self.repository.update_document_file_path(
                document_id=document.id,
                file_path=str(file_path.resolve())
            )

            if updated_document is None:
                file_path.unlink(missing_ok=True)
                raise PDFDownloadError(
                    "Unable to update document information."
                )

            logger.info(
                "PDF downloaded successfully for document %s",
                document.id
            )

            return {
                "document_id": str(document.id),
                "file_path": updated_document.file_path,
                "status": updated_document.status.value,
                "file_size": downloaded_size,
                "message": "PDF downloaded successfully."
            }

        except requests.RequestException as error:

            raise PDFDownloadError(
                f"PDF download failed: {error}"
            ) from error

        finally:
            if "response" in locals():
                response.close()
