from sentence_transformers import SentenceTransformer

from config.settings import get_settings


class EmbeddingGenerator:

    def __init__(self, model=None):

        settings = get_settings()

        self.model_name = settings.embedding_model

        # Allow model injection for testing
        self.model = model or SentenceTransformer(
            self.model_name,
            device="cpu"
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Convert a list of text strings into numerical embeddings.
        """

        if not isinstance(texts, list):
            raise TypeError("Input must be a list of strings.")

        if not texts:
            return []

        if not all(isinstance(text, str) and text.strip() for text in texts):
            raise ValueError("All input texts must be non-empty strings.")

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        return embeddings.tolist()

    def embed_chunks(self, chunks: list[dict]) -> list[dict]:
        """
        Generate embeddings while preserving chunk metadata.
        """

        if not isinstance(chunks, list):
            raise TypeError("Chunks must be provided as a list.")

        if not chunks:
            return []

        texts = [chunk["text"] for chunk in chunks]

        embeddings = self.embed_texts(texts)

        embedded_chunks = []

        for chunk, embedding in zip(chunks, embeddings):

            embedded_chunk = chunk.copy()
            embedded_chunk["embedding"] = embedding

            embedded_chunks.append(embedded_chunk)

        return embedded_chunks