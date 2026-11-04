import pytest

from retrieval.query_encoder import QueryEncoder


class MockEmbeddingGenerator:

    def embed_texts(self, texts):

        return [
            [0.1, 0.2, 0.3]
            for _ in texts
        ]


@pytest.fixture
def query_encoder():

    return QueryEncoder(
        embedding_generator=MockEmbeddingGenerator()
    )


def test_encode_query_returns_embedding(query_encoder):

    query = "Machine learning in healthcare."

    result = query_encoder.encode_query(query)

    assert isinstance(result, list)
    assert len(result) == 3
    assert result == [0.1, 0.2, 0.3]


def test_query_is_stripped_before_encoding(query_encoder):

    result = query_encoder.encode_query(
        "  Machine learning research.  "
    )

    assert result == [0.1, 0.2, 0.3]


def test_empty_query_raises_error(query_encoder):

    with pytest.raises(ValueError, match="Query cannot be empty"):

        query_encoder.encode_query("")


def test_whitespace_query_raises_error(query_encoder):

    with pytest.raises(ValueError, match="Query cannot be empty"):

        query_encoder.encode_query("   ")


def test_invalid_query_type_raises_error(query_encoder):

    with pytest.raises(TypeError, match="Query must be a string"):

        query_encoder.encode_query(123)