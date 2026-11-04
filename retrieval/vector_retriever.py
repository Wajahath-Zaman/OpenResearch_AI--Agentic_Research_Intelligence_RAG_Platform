from ingestion.embedding_generator import EmbeddingGenerator
from retrieval.vector_store import ChromaVectorStore

from retrieval.retrieval_config import RetrievalConfig


class VectorRetriever:

    def __init__(
            self,
            embedding_generator=None,
            vector_store=None,
            retrieval_config=None
        ):

            self.embedding_generator = (
                embedding_generator or EmbeddingGenerator()
            )

            self.vector_store = vector_store or ChromaVectorStore()

            self.retrieval_config = (
                retrieval_config or RetrievalConfig()
            )

    def search(
            self,
            query: str,
            n_results: int | None = None,
            filters: dict | None = None
        ) -> list[dict]:
        """
        Retrieve the most semantically similar document chunks.
        """

        if not isinstance(query, str):
            raise TypeError("Query must be a string.")

        if not query.strip():
            raise ValueError("Query cannot be empty.")


        top_k = self.retrieval_config.resolve_k(n_results)

        # Generate query embedding
        query_embedding = self.embedding_generator.embed_texts(
            [query.strip()]
        )[0]

        metadata_filter = self._build_metadata_filter(filters)

        # Access ChromaDB collection
        collection = self.vector_store.collection

        # Handle empty collection
        total_chunks = collection.count()

        if total_chunks == 0:
            return []

        # Retrieve nearest neighbours
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, total_chunks),
            where=metadata_filter,
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

        retrieved_chunks = []

        for index, chunk_id in enumerate(results["ids"][0]):

            distance = results["distances"][0][index]

            similarity = 1.0 - distance

            retrieved_chunks.append({
                "id": chunk_id,
                "text": results["documents"][0][index],
                "metadata": results["metadatas"][0][index],
                "distance": distance,
                "similarity": similarity
            })

        return retrieved_chunks

    def _build_metadata_filter(self, filters: dict | None) -> dict | None:
        """
        Build a valid ChromaDB metadata filter.
        """

        if filters is None:
            return None

        if not isinstance(filters, dict):
            raise TypeError("Metadata filters must be a dictionary.")

        if not filters:
            return None

        allowed_fields = {
            "document_id",
            "chunk_index",
            "page_number",
            "word_count",
            "source_path",
            "title"
        }

        conditions = []

        for field, value in filters.items():

            if field not in allowed_fields:
                raise ValueError(f"Unsupported metadata filter: {field}")

            if not isinstance(value, (str, int, float, bool)):
                raise ValueError(f"Invalid filter value for: {field}")

            conditions.append({
                field: {"$eq": value}
            })

        if len(conditions) == 1:
            return conditions[0]

        return {"$and": conditions}