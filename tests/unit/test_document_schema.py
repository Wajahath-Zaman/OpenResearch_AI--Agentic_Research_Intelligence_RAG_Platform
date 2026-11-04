import pytest

from pydantic import ValidationError

from ingestion.document_schema import (
    ExtractedDocument,
    ExtractedDocumentMetadata,
    ExtractedPage
)


def test_valid_extracted_document():

    document = ExtractedDocument(
        source_path="sample.pdf",
        metadata=ExtractedDocumentMetadata(
            title="Sample Research Paper",
            author="Research Team",
            total_pages=2
        ),
        pages=[
            ExtractedPage(
                page_number=1,
                text="First page content."
            ),
            ExtractedPage(
                page_number=2,
                text="Second page content."
            )
        ]
    )

    assert document.metadata.total_pages == 2
    assert len(document.pages) == 2


def test_page_count_mismatch_raises_error():

    with pytest.raises(
        ValidationError,
        match="Metadata page count does not match"
    ):
        ExtractedDocument(
            source_path="sample.pdf",
            metadata=ExtractedDocumentMetadata(
                title="Sample Paper",
                total_pages=3
            ),
            pages=[
                ExtractedPage(
                    page_number=1,
                    text="Page one."
                ),
                ExtractedPage(
                    page_number=2,
                    text="Page two."
                )
            ]
        )


def test_invalid_page_number_raises_error():

    with pytest.raises(ValidationError):

        ExtractedPage(
            page_number=0,
            text="Invalid page."
        )