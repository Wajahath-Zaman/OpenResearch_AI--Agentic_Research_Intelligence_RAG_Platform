class SourceTracker:

    def extract_sources(self, retrieved_chunks: list[dict]) -> list[dict]:
        """
        Extract unique source references from retrieved chunks.
        """

        if not isinstance(retrieved_chunks, list):
            raise TypeError("Retrieved chunks must be a list.")

        sources = []
        seen_sources = set()

        for chunk in retrieved_chunks:

            if not isinstance(chunk, dict):
                raise TypeError("Each retrieved chunk must be a dictionary.")

            metadata = chunk.get("metadata", {})

            if not isinstance(metadata, dict):
                raise TypeError("Chunk metadata must be a dictionary.")

            document_id = metadata.get("document_id")
            page_number = metadata.get("page_number")

            if not document_id or page_number is None:
                continue

            source_key = (document_id, page_number)

            if source_key in seen_sources:
                continue

            seen_sources.add(source_key)

            sources.append({
                "document_id": document_id,
                "title": metadata.get("title", "Unknown Document"),
                "page_number": page_number,
                "source_path": metadata.get("source_path")
            })

        return sources