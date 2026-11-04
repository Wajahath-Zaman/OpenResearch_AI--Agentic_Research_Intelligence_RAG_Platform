

class TextChunker:

    def __init__(self, chunk_size: int = 180, chunk_overlap: int = 30):

        if chunk_size <= 0:
            raise ValueError("Chunk size must be greater than zero.")

        if chunk_overlap < 0:
            raise ValueError("Chunk overlap cannot be negative.")

        if chunk_overlap >= chunk_size:
            raise ValueError("Chunk overlap must be smaller than chunk size.")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str) -> list[str]:
        """
        Split text into overlapping word-based chunks.
        """

        words = text.split()

        if not words:
            return []

        chunks = []
        start = 0

        while start < len(words):

            end = start + self.chunk_size

            chunk = " ".join(words[start:end])

            chunks.append(chunk)

            if end >= len(words):
                break

            start = end - self.chunk_overlap

        return chunks

    def chunk_document(self, document: dict) -> list[dict]:
        """
        Generate chunks while preserving document and page metadata.
        """

        chunks = []
        chunk_index = 0

        for page in document["pages"]:

            page_chunks = self.chunk_text(page["text"])

            for chunk_text in page_chunks:

                chunks.append({
                    "chunk_index": chunk_index,
                    "page_number": page["page_number"],
                    "text": chunk_text,
                    "word_count": len(chunk_text.split())
                })

                chunk_index += 1

        return chunks