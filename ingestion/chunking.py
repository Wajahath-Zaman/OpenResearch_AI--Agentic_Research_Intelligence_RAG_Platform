

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
        Used for oversized paragraphs.
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

    def chunk_paragraphs(self, text: str) -> list[str]:
        """
        Combine paragraphs into chunks while preserving
        paragraph boundaries wherever possible.
        """

        paragraphs = [
            paragraph.strip()
            for paragraph in text.split("\n\n")
            if paragraph.strip()
        ]

        if not paragraphs:
            return []

        chunks = []
        current_words = []

        for paragraph in paragraphs:

            paragraph_words = paragraph.split()

            # Split oversized paragraphs separately
            if len(paragraph_words) > self.chunk_size:

                if current_words:
                    chunks.append(" ".join(current_words))
                    current_words = []

                chunks.extend(self.chunk_text(paragraph))
                continue

            # Add paragraph if it fits
            if len(current_words) + len(paragraph_words) <= self.chunk_size:

                current_words.extend(paragraph_words)

            else:
                # Save current chunk
                chunks.append(" ".join(current_words))

                # Preserve overlap between chunks
                overlap_words = (
                    current_words[-self.chunk_overlap:]
                    if self.chunk_overlap > 0
                    else []
                )

                # Ensure the next chunk does not exceed chunk_size
                if len(overlap_words) + len(paragraph_words) <= self.chunk_size:
                    current_words = overlap_words + paragraph_words
                else:
                    current_words = paragraph_words

        if current_words:
            chunks.append(" ".join(current_words))

        return chunks

    def chunk_document(self, document: dict) -> list[dict]:
        """
        Generate paragraph-aware chunks while preserving
        document and page metadata.
        """

        chunks = []
        chunk_index = 0

        for page in document["pages"]:

            page_chunks = self.chunk_paragraphs(page["text"])

            for chunk_text in page_chunks:

                chunks.append({
                    "chunk_index": chunk_index,
                    "page_number": page["page_number"],
                    "text": chunk_text,
                    "word_count": len(chunk_text.split())
                })

                chunk_index += 1

        return chunks