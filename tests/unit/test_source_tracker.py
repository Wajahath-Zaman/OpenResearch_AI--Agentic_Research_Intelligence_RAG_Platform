import pytest

from retrieval.source_tracker import SourceTracker


@pytest.fixture
def tracker():
    return SourceTracker()


@pytest.fixture
def retrieved_chunks():

    return [
        {
            "text": "Machine learning improves diagnosis.",
            "metadata": {
                "document_id": "doc-001",
                "title": "AI in Healthcare",
                "page_number": 5,
                "source_path": "data/raw/paper1.pdf"
            }
        },
        {
            "text": "Deep learning supports diagnosis.",
            "metadata": {
                "document_id": "doc-001",
                "title": "AI in Healthcare",
                "page_number": 5,
                "source_path": "data/raw/paper1.pdf"
            }
        },
        {
            "text": "Neural networks identify abnormalities.",
            "metadata": {
                "document_id": "doc-001",
                "title": "AI in Healthcare",
                "page_number": 6,
                "source_path": "data/raw/paper1.pdf"
            }
        },
        {
            "text": "Robots use machine learning.",
            "metadata": {
                "document_id": "doc-002",
                "title": "Robotics Research",
                "page_number": 3,
                "source_path": "data/raw/paper2.pdf"
            }
        }
    ]


def test_extract_sources(tracker, retrieved_chunks):

    sources = tracker.extract_sources(retrieved_chunks)

    assert len(sources) == 3


def test_duplicate_sources_are_removed(tracker, retrieved_chunks):

    sources = tracker.extract_sources(retrieved_chunks)

    source_keys = [
        (source["document_id"], source["page_number"])
        for source in sources
    ]

    assert len(source_keys) == len(set(source_keys))


def test_multiple_pages_from_same_document_are_preserved(
    tracker,
    retrieved_chunks
):

    sources = tracker.extract_sources(retrieved_chunks)

    document_one_sources = [
        source for source in sources
        if source["document_id"] == "doc-001"
    ]

    assert len(document_one_sources) == 2


def test_source_metadata_is_preserved(tracker, retrieved_chunks):

    sources = tracker.extract_sources(retrieved_chunks)

    first_source = sources[0]

    assert first_source["document_id"] == "doc-001"
    assert first_source["title"] == "AI in Healthcare"
    assert first_source["page_number"] == 5


def test_empty_chunks_return_empty_sources(tracker):

    assert tracker.extract_sources([]) == []


def test_invalid_input_type_raises_error(tracker):

    with pytest.raises(TypeError):

        tracker.extract_sources("invalid")


def test_invalid_chunk_type_raises_error(tracker):

    with pytest.raises(TypeError):

        tracker.extract_sources(["invalid"])


def test_missing_document_id_is_skipped(tracker):

    chunks = [
        {
            "text": "Research content.",
            "metadata": {
                "page_number": 5
            }
        }
    ]

    assert tracker.extract_sources(chunks) == []


def test_missing_page_number_is_skipped(tracker):

    chunks = [
        {
            "text": "Research content.",
            "metadata": {
                "document_id": "doc-001"
            }
        }
    ]

    assert tracker.extract_sources(chunks) == []


def test_invalid_metadata_type_raises_error(tracker):

    chunks = [
        {
            "text": "Research content.",
            "metadata": "invalid"
        }
    ]

    with pytest.raises(TypeError):

        tracker.extract_sources(chunks)