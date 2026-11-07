import asyncio

from ragas.embeddings.base import BaseRagasEmbedding
from ingestion.embedding_generator import EmbeddingGenerator


class RagasEmbeddingAdapter(BaseRagasEmbedding):

    def __init__(self, embedding_generator=None):
        super().__init__()

        self.embedding_generator = (
            embedding_generator or EmbeddingGenerator()
        )

    def embed_text(self, text: str, **kwargs) -> list[float]:
        return self.embedding_generator.embed_texts([text])[0]

    async def aembed_text(self, text: str, **kwargs) -> list[float]:
        return await asyncio.to_thread(
            self.embed_text,
            text
        )

    def embed_texts(
        self,
        texts: list[str],
        **kwargs
    ) -> list[list[float]]:
        return self.embedding_generator.embed_texts(texts)

    async def aembed_texts(
        self,
        texts: list[str],
        **kwargs
    ) -> list[list[float]]:
        return await asyncio.to_thread(
            self.embed_texts,
            texts
        )