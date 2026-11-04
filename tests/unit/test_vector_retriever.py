import pytest

from ingestion.embedding_generator import EmbeddingGenerator
from retrieval.vector_retriever import VectorRetriever
from retrieval.vector_store import ChromaVectorStore


class MockEmbeddingGenerator:

    def embed_texts(self, texts):

        return [
            [1.0, 0.0, 0.0]
            for _ in texts
        ]


@pytest.fixture
def retriever(tmp_path):

    vector_store = ChromaVectorStore(
        persist_directory=tmp_path / "chroma"
    )

    embedding_generator = MockEmbeddingGenerator()

    return VectorRetriever(
        embedding_generator=embedding_generator,
        vector_store=vector_store
    )


@pytest.fixture
def sample_chunks():

    return [
        {
            "chunk_index": 0,
            "page_number": 1,
            "text": "Machine learning research.",
            "word_count": 3,
            "embedding": [1.0, 0.0, 0.0]
        },
        {
            "chunk_index": 1,
            "page_number": 2,
            "text": "Deep learning research.",
            "word_count": 3,
            "embedding": [0.0, 1.0, 0.0]
        },
        {
            "chunk_index": 2,
            "page_number": 3,
            "text": "Natural language processing.",
            "word_count": 3,
            "embedding": [0.0, 0.0, 1.0]
        }
    ]


def test_search_returns_relevant_chunks(retriever, sample_chunks):

    retriever.vector_store.store_chunks(
        document_id="document-001",
        chunks=sample_chunks
    )

    results = retriever.search(
        query="Machine learning",
        n_results=2
    )

    assert len(results) == 2

    assert results[0]["text"] == "Machine learning research."

    assert results[0]["similarity"] == pytest.approx(1.0)


def test_search_returns_similarity_scores(retriever, sample_chunks):

    retriever.vector_store.store_chunks(
        document_id="document-001",
        chunks=sample_chunks
    )

    results = retriever.search(
        query="Machine learning",
        n_results=3
    )

    for result in results:

        assert "distance" in result
        assert "similarity" in result

        assert result["similarity"] == pytest.approx(
            1.0 - result["distance"]
        )


def test_search_respects_n_results(retriever, sample_chunks):

    retriever.vector_store.store_chunks(
        document_id="document-001",
        chunks=sample_chunks
    )

    results = retriever.search(
        query="Machine learning",
        n_results=1
    )

    assert len(results) == 1


def test_search_empty_collection_returns_empty_list(retriever):

    results = retriever.search("Machine learning")

    assert results == []


def test_empty_query_raises_error(retriever):

    with pytest.raises(ValueError):

        retriever.search("")


def test_whitespace_query_raises_error(retriever):

    with pytest.raises(ValueError):

        retriever.search("   ")


def test_invalid_query_type_raises_error(retriever):

    with pytest.raises(TypeError):

        retriever.search(123)


def test_invalid_n_results_raises_error(retriever):

    with pytest.raises(ValueError):

        retriever.search("Machine learning", n_results=0)


def test_invalid_n_results_type_raises_error(retriever):

    with pytest.raises(TypeError):

        retriever.search("Machine learning", n_results="five")


def test_filter_by_document_id(retriever, sample_chunks):

    retriever.vector_store.store_chunks(
        document_id="document-001",
        chunks=sample_chunks
    )

    retriever.vector_store.store_chunks(
        document_id="document-002",
        chunks=sample_chunks
    )

    results = retriever.search(
        query="Machine learning",
        filters={
            "document_id": "document-001"
        }
    )

    assert len(results) == 3

    assert all(
        result["metadata"]["document_id"] == "document-001"
        for result in results
    )


def test_filter_by_title(retriever, sample_chunks):

    retriever.vector_store.store_chunks(
        document_id="document-001",
        chunks=sample_chunks,
        title="Medical AI Research"
    )

    retriever.vector_store.store_chunks(
        document_id="document-002",
        chunks=sample_chunks,
        title="Robotics Research"
    )

    results = retriever.search(
        query="Machine learning",
        filters={
            "title": "Medical AI Research"
        }
    )

    assert len(results) == 3

    assert all(
        result["metadata"]["title"] == "Medical AI Research"
        for result in results
    )


def test_multiple_metadata_filters(retriever, sample_chunks):

    retriever.vector_store.store_chunks(
        document_id="document-001",
        chunks=sample_chunks
    )

    results = retriever.search(
        query="Machine learning",
        filters={
            "document_id": "document-001",
            "page_number": 2
        }
    )

    assert len(results) == 1

    assert results[0]["metadata"]["page_number"] == 2


def test_filter_with_no_matching_results(retriever, sample_chunks):

    retriever.vector_store.store_chunks(
        document_id="document-001",
        chunks=sample_chunks
    )

    results = retriever.search(
        query="Machine learning",
        filters={
            "document_id": "nonexistent-document"
        }
    )

    assert results == []


def test_invalid_filter_type_raises_error(retriever):

    with pytest.raises(TypeError):

        retriever.search(
            query="Machine learning",
            filters="document-001"
        )


def test_unsupported_metadata_field_raises_error(retriever):

    with pytest.raises(ValueError, match="Unsupported metadata filter"):

        retriever.search(
            query="Machine learning",
            filters={
                "publication_year": 2025
            }
        )


def test_empty_filter_behaves_like_unfiltered_search(
    retriever,
    sample_chunks
):

    retriever.vector_store.store_chunks(
        document_id="document-001",
        chunks=sample_chunks
    )

    results_without_filter = retriever.search(
        query="Machine learning"
    )

    results_with_empty_filter = retriever.search(
        query="Machine learning",
        filters={}
    )

    assert len(results_without_filter) == len(results_with_empty_filter)


def test_retriever_uses_default_top_k(retriever, sample_chunks):

    retriever.vector_store.store_chunks(
        document_id="document-001",
        chunks=sample_chunks
    )

    results = retriever.search("Machine learning")

    # The collection contains only three chunks.
    # Default K is five, so all three should be returned.
    assert len(results) == 3


def test_retriever_accepts_custom_top_k(retriever, sample_chunks):

    retriever.vector_store.store_chunks(
        document_id="document-001",
        chunks=sample_chunks
    )

    results = retriever.search(
        "Machine learning",
        n_results=2
    )

    assert len(results) == 2