import pymupdf

from ingestion.pdf_loader import PDFLoader, PDFLoadError

import pytest


def create_sample_pdf(file_path):

    pdf = pymupdf.open()

    page1 = pdf.new_page()
    page1.insert_text((50,50), "This is page One.")

    page2 = pdf.new_page()
    page2.insert_text((50,50), "This is page two.")

    pdf.set_metadata({
        "title": "Sample Research Paper",
        "author": "Open Research Team"
    })

    pdf.save(file_path)
    pdf.close()


def test_pdf_loader_extracts_metadata(tmp_path):

    pdf_path = tmp_path / "sample.pdf"

    create_sample_pdf(pdf_path)

    loader = PDFLoader()

    result = loader.load(pdf_path)

    assert result['metadata']['title'] == "Sample Research Paper"
    assert result['metadata']['author'] == "Open Research Team"
    assert result['metadata']['total_pages'] == 2


def test_pdf_loader_extracts_page_text(tmp_path):

    pdf_path = tmp_path / "sample.pdf"

    create_sample_pdf(pdf_path)

    loader = PDFLoader()
    result = loader.load(pdf_path)

    assert len(result['pages']) == 2

    assert result['pages'][0]['page_number'] == 1
    assert "This is page One." in result['pages'][0]['text']

    assert result['pages'][1]["page_number"] == 2
    assert "This is page two." in result['pages'][1]["text"]



def test_missing_pdf_raises_error(tmp_path):

    pdf_path = tmp_path / "missing.pdf"

    loader = PDFLoader()

    with pytest.raises(PDFLoadError, match="PDF file not found"):
        loader.load(pdf_path)


def test_unsupported_file_format_raises_error(tmp_path):

    file_path = tmp_path / "sample.txt"
    file_path.write_text("This is not a PDF.")

    loader = PDFLoader()

    with pytest.raises(PDFLoadError, match="Unsupported file format"):
        loader.load(file_path)


def test_invalid_pdf_raises_error(tmp_path):

    pdf_path = tmp_path / "invalid.pdf"
    pdf_path.write_text("This is not a valid PDF document.")

    loader = PDFLoader()

    with pytest.raises(PDFLoadError, match="Failed to load PDF"):
        loader.load(pdf_path)
