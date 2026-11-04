import numpy as np
import pytest

from ingestion.embedding_generator import EmbeddingGenerator


class MockEmbeddingModel:

    def encode(
        self,
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    ):

        return np.array([
            [0.6, 0.8, 0.0] for _ in texts
        ])


@pytest.fixture
def embedding_generator():

    return EmbeddingGenerator(
        model=MockEmbeddingModel()
    )


def test_embed_texts_returns_vectors(embedding_generator):

    texts = [
        "Machine learning research.",
        "Deep learning applications."
    ]

    embeddings = embedding_generator.embed_texts(texts)

    assert len(embeddings) == 2
    assert len(embeddings[0]) == 3
    assert embeddings[0] == [0.6, 0.8, 0.0]


def test_empty_text_list_returns_empty_list(embedding_generator):

    result = embedding_generator.embed_texts([])

    assert result == []


def test_invalid_text_input_raises_error(embedding_generator):

    with pytest.raises(TypeError):
        embedding_generator.embed_texts("Machine learning")


def test_empty_string_raises_error(embedding_generator):

    with pytest.raises(ValueError):
        embedding_generator.embed_texts(["Valid text", ""])


def test_embed_chunks_preserves_metadata(embedding_generator):

    chunks = [
        {
            "chunk_index": 0,
            "page_number": 1,
            "text": "Machine learning research.",
            "word_count": 3
        },
        {
            "chunk_index": 1,
            "page_number": 2,
            "text": "Deep learning research.",
            "word_count": 3
        }
    ]

    result = embedding_generator.embed_chunks(chunks)

    assert len(result) == 2

    assert result[0]["chunk_index"] == 0
    assert result[0]["page_number"] == 1
    assert result[0]["text"] == "Machine learning research."
    assert len(result[0]["embedding"]) == 3

    assert result[1]["chunk_index"] == 1
    assert result[1]["page_number"] == 2

    # Original chunks must remain unchanged
    assert "embedding" not in chunks[0]


def test_empty_chunks_return_empty_list(embedding_generator):

    result = embedding_generator.embed_chunks([])

    assert result == []