import pytest

from ingestion.chunking import TextChunker


@pytest.fixture
def chunker():
    return TextChunker(chunk_size=10, chunk_overlap=2)


def test_short_text_creates_single_chunk(chunker):

    text = "This is a short research paragraph."

    chunks = chunker.chunk_text(text)

    assert len(chunks) == 1
    assert chunks[0] == text


def test_long_text_creates_multiple_chunks(chunker):

    text = " ".join(f"word{i}" for i in range(25))

    chunks = chunker.chunk_text(text)

    assert len(chunks) == 3


def test_chunk_overlap_is_preserved(chunker):

    text = " ".join(f"word{i}" for i in range(15))

    chunks = chunker.chunk_text(text)

    first_chunk_words = chunks[0].split()
    second_chunk_words = chunks[1].split()

    assert first_chunk_words[-2:] == second_chunk_words[:2]


def test_empty_text_returns_empty_list(chunker):

    assert chunker.chunk_text("") == []


def test_document_chunking_preserves_page_numbers():

    chunker = TextChunker(chunk_size=5, chunk_overlap=1)

    document = {
        "metadata": {
            "title": "Sample Paper",
            "total_pages": 2
        },
        "pages": [
            {
                "page_number": 1,
                "text": " ".join(f"page1word{i}" for i in range(12))
            },
            {
                "page_number": 2,
                "text": " ".join(f"page2word{i}" for i in range(8))
            }
        ]
    }

    chunks = chunker.chunk_document(document)

    assert len(chunks) == 5

    assert all(chunk["page_number"] == 1 for chunk in chunks[:3])
    assert all(chunk["page_number"] == 2 for chunk in chunks[3:])

    assert chunks[0]["chunk_index"] == 0
    assert chunks[-1]["chunk_index"] == 4


def test_invalid_chunk_configuration():

    with pytest.raises(ValueError):
        TextChunker(chunk_size=0)

    with pytest.raises(ValueError):
        TextChunker(chunk_size=10, chunk_overlap=10)

    with pytest.raises(ValueError):
        TextChunker(chunk_size=10, chunk_overlap=-1)