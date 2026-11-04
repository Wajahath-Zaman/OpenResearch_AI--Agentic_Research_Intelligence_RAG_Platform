import pytest

from retrieval.vector_store import ChromaVectorStore


@pytest.fixture
def vector_store(tmp_path):

    return ChromaVectorStore(
        persist_directory=tmp_path / "chroma_test"
    )


@pytest.fixture
def sample_chunks():

    return [
        {
            "chunk_index": 0,
            "page_number": 1,
            "text": "Machine learning research.",
            "word_count": 3,
            "embedding": [0.1, 0.2, 0.3]
        },
        {
            "chunk_index": 1,
            "page_number": 2,
            "text": "Deep learning research.",
            "word_count": 3,
            "embedding": [0.4, 0.5, 0.6]
        }
    ]


def test_store_chunks(vector_store, sample_chunks):

    stored_count = vector_store.store_chunks(
        document_id="document-001",
        chunks=sample_chunks
    )

    assert stored_count == 2
    assert vector_store.count() == 2


def test_store_preserves_metadata(vector_store, sample_chunks):

    vector_store.store_chunks(
        document_id="document-001",
        chunks=sample_chunks,
        source_path="data/raw/sample.pdf",
        title="Sample Research Paper"
    )

    results = vector_store.get_document_chunks("document-001")

    assert len(results) == 2

    first_chunk = results[0]

    assert first_chunk["text"] == "Machine learning research."

    assert first_chunk["metadata"]["document_id"] == "document-001"
    assert first_chunk["metadata"]["page_number"] == 1
    assert first_chunk["metadata"]["source_path"] == "data/raw/sample.pdf"
    assert first_chunk["metadata"]["title"] == "Sample Research Paper"


def test_upsert_prevents_duplicate_chunks(vector_store, sample_chunks):

    vector_store.store_chunks("document-001", sample_chunks)

    vector_store.store_chunks("document-001", sample_chunks)

    assert vector_store.count() == 2


def test_retrieve_chunks_by_document_id(vector_store, sample_chunks):

    vector_store.store_chunks("document-001", sample_chunks)

    vector_store.store_chunks("document-002", sample_chunks)

    results = vector_store.get_document_chunks("document-001")

    assert len(results) == 2

    assert all(
        chunk["metadata"]["document_id"] == "document-001"
        for chunk in results
    )

    assert vector_store.count() == 4


def test_delete_document(vector_store, sample_chunks):

    vector_store.store_chunks("document-001", sample_chunks)

    deleted_count = vector_store.delete_document("document-001")

    assert deleted_count == 2
    assert vector_store.count() == 0


def test_empty_chunks_return_zero(vector_store):

    result = vector_store.store_chunks(
        document_id="document-001",
        chunks=[]
    )

    assert result == 0
    assert vector_store.count() == 0


def test_empty_document_id_raises_error(vector_store):

    with pytest.raises(ValueError):

        vector_store.store_chunks(
            document_id="",
            chunks=[]
        )


def test_persistence_across_instances(tmp_path, sample_chunks):

    storage_path = tmp_path / "persistent_chroma"

    store1 = ChromaVectorStore(
        persist_directory=storage_path
    )

    store1.store_chunks(
        document_id="document-001",
        chunks=sample_chunks
    )

    # Create another client using the same persistent directory
    store2 = ChromaVectorStore(
        persist_directory=storage_path
    )

    results = store2.get_document_chunks("document-001")

    assert len(results) == 2
    assert results[0]["metadata"]["document_id"] == "document-001"