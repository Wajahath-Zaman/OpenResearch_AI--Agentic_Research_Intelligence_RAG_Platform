from pathlib import Path

import chromadb

from config.settings import get_settings


class ChromaVectorStore:

    def __init__(
        self,
        persist_directory: str | Path | None = None,
        collection_name: str = "research_chunks",
        client=None
    ):

        settings = get_settings()

        self.persist_directory = Path(
            persist_directory or settings.chroma_persist_directory
        )

        self.persist_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        self.client = client or chromadb.PersistentClient(
            path=str(self.persist_directory)
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def store_chunks(
        self,
        document_id: str,
        chunks: list[dict],
        source_path: str | None = None,
        title: str | None = None
    ) -> int:
        """
        Store embedded document chunks in ChromaDB.
        """

        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError("Document ID cannot be empty.")

        if not isinstance(chunks, list):
            raise TypeError("Chunks must be provided as a list.")

        if not chunks:
            return 0

        ids = []
        documents = []
        embeddings = []
        metadatas = []

        for chunk in chunks:

            if not isinstance(chunk, dict):
                raise TypeError("Each chunk must be a dictionary.")

            required_fields = {
                "chunk_index",
                "page_number",
                "text",
                "embedding",
                "word_count"
            }

            if not required_fields.issubset(chunk):
                raise ValueError("Chunk is missing required fields.")

            if not isinstance(chunk["text"], str) or not chunk["text"].strip():
                raise ValueError("Chunk text cannot be empty.")

            if not chunk["embedding"]:
                raise ValueError("Chunk embedding cannot be empty.")

            chunk_id = f"{document_id}:{chunk['chunk_index']}"

            metadata = {
                "document_id": document_id,
                "chunk_index": chunk["chunk_index"],
                "page_number": chunk["page_number"],
                "word_count": chunk["word_count"]
            }

            if source_path:
                metadata["source_path"] = source_path

            if title:
                metadata["title"] = title

            ids.append(chunk_id)
            documents.append(chunk["text"])
            embeddings.append(chunk["embedding"])
            metadatas.append(metadata)

        # Upsert makes repeated storage of the same chunk IDs idempotent.
        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

        return len(ids)

    def get_document_chunks(self, document_id: str) -> list[dict]:
        """
        Retrieve all stored chunks belonging to a document.
        """

        if not document_id or not document_id.strip():
            raise ValueError("Document ID cannot be empty.")

        results = self.collection.get(
            where={"document_id": document_id},
            include=["documents", "metadatas"]
        )

        chunks = []

        for index, chunk_id in enumerate(results["ids"]):

            chunks.append({
                "id": chunk_id,
                "text": results["documents"][index],
                "metadata": results["metadatas"][index]
            })

        return chunks

    def delete_document(self, document_id: str) -> int:
        """
        Delete all chunks associated with a document.
        """

        existing_chunks = self.get_document_chunks(document_id)

        if not existing_chunks:
            return 0

        self.collection.delete(
            where={"document_id": document_id}
        )

        return len(existing_chunks)

    def count(self) -> int:
        """
        Return the total number of stored chunks.
        """

        return self.collection.count()