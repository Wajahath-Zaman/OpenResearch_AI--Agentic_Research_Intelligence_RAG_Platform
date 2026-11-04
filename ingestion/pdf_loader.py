from pathlib import Path

import pymupdf

from ingestion.document_schema import (
    ExtractedDocument,
    ExtractedDocumentMetadata,
    ExtractedPage
)

class PDFLoadError(Exception):
    """
    Raised When a PDF connot be loaded or processed.
    """


class PDFLoader:

    def load(self, pdf_path: str | Path) -> dict:

        pdf_path = Path(pdf_path)

        # Validate file existence
        if not pdf_path.exists():
            raise PDFLoadError(f"PDF file not found: {pdf_path}")

        # Validate file format
        if pdf_path.suffix.lower() != ".pdf":
            raise PDFLoadError("Unsupported file format. Only PDF files are accepted.")

        try:
            with pymupdf.open(pdf_path) as pdf:

                if pdf.is_encrypted:
                    raise PDFLoadError("Encrypted PDFs are not supported.")

                if len(pdf) == 0:
                    raise PDFLoadError("PDF contains no pages.")

                metadata = pdf.metadata or {}

                pages = []

                for page_number, page in enumerate(pdf, start=1):

                    pages.append({
                        "page_number": page_number,
                        "text": page.get_text("text").strip()
                    })

                return ExtractedDocument(
                        source_path=str(pdf_path.resolve()),
                        metadata=ExtractedDocumentMetadata(
                            title=metadata.get("title"),
                            author=metadata.get("author"),
                            total_pages=len(pages)
                        ),
                        pages=[
                            ExtractedPage(**page)
                            for page in pages
                        ]
                    ).model_dump()

        except PDFLoadError:
            raise

        except Exception as exc:
            raise PDFLoadError(
                f"Failed to load PDF: {exc}"
            ) from exc
