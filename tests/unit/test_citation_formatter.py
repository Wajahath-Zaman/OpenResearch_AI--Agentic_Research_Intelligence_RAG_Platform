import pytest

from llm.citation_formatter import CitationFormatter


@pytest.fixture
def formatter():
    return CitationFormatter()


@pytest.fixture
def sample_chunks():
    return [
        {
            "id": "doc-1:0",
            "text": "CNNs extract image features.",
            "metadata": {
                "document_id": "doc-1",
                "page_number": 1,
                "title": "Deep Learning Research",
                "source_path": "paper1.pdf"
            }
        },
        {
            "id": "doc-1:1",
            "text": "CNNs perform classification.",
            "metadata": {
                "document_id": "doc-1",
                "page_number": 1,
                "title": "Deep Learning Research",
                "source_path": "paper1.pdf"
            }
        },
        {
            "id": "doc-2:0",
            "text": "CNNs are used in diagnosis.",
            "metadata": {
                "document_id": "doc-2",
                "page_number": 5,
                "title": "Medical Image Classification",
                "source_path": "paper2.pdf"
            }
        }
    ]


def test_format_sources_returns_list(formatter, sample_chunks):
    citations = formatter.format_sources(sample_chunks)

    assert isinstance(citations, list)


def test_format_sources_removes_duplicate_pages(formatter, sample_chunks):
    citations = formatter.format_sources(sample_chunks)

    assert len(citations) == 2


def test_citation_ids_are_sequential(formatter, sample_chunks):
    citations = formatter.format_sources(sample_chunks)

    assert [source["citation_id"] for source in citations] == [1, 2]


def test_document_ids_are_preserved(formatter, sample_chunks):
    citations = formatter.format_sources(sample_chunks)

    assert citations[0]["document_id"] == "doc-1"
    assert citations[1]["document_id"] == "doc-2"


def test_page_numbers_are_preserved(formatter, sample_chunks):
    citations = formatter.format_sources(sample_chunks)

    assert citations[0]["page_number"] == 1
    assert citations[1]["page_number"] == 5


def test_titles_are_preserved(formatter, sample_chunks):
    citations = formatter.format_sources(sample_chunks)

    assert citations[0]["title"] == "Deep Learning Research"


def test_empty_chunks_return_empty_citations(formatter):
    citations = formatter.format_sources([])

    assert citations == []