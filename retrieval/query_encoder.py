from ingestion.embedding_generator import EmbeddingGenerator

class QueryEncoder:

    def __init__(self, embedding_generator=None):

        self.embedding_generator = (
            embedding_generator or EmbeddingGenerator()
        )

    def encode_query(self, query: str) -> list[float]:
        """
        Convert a user query into a numerical embedding
        """

        if not isinstance(query, str):
            raise TypeError("Query must be a string.")

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        embeddings = self.embedding_generator.embed_texts(
            [query.strip()]
        )

        return embeddings[0]