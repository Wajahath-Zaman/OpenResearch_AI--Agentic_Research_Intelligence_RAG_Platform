import re
import unicodedata


class TextPreprocessor:

    def preprocess_text(self, text: str) -> str:
        """
        Clean extracted PDF text while preserving paragraph structure.
        """

        if not isinstance(text, str):
            raise TypeError("Input text must be a string.")

        # Normalize Unicode characters
        text = unicodedata.normalize("NFC", text)

        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Remove null characters
        text = text.replace("\x00", "")

        # Fix words split by line-break hyphenation
        text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)

        # Process individual lines
        lines = []

        for line in text.split("\n"):

            # Remove leading and trailing whitespace
            line = line.strip()

            # Normalize multiple spaces and tabs
            line = re.sub(r"[ \t]+", " ", line)

            lines.append(line)

        # Preserve paragraph breaks but remove excessive blank lines
        text = "\n".join(lines)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    def preprocess_document(self, document: dict) -> dict:
        """
        Preprocess all pages of an extracted document.
        """

        processed_document = document.copy()

        processed_pages = []

        for page in document["pages"]:

            processed_page = page.copy()

            processed_page["text"] = self.preprocess_text(
                page["text"]
            )

            processed_pages.append(processed_page)

        processed_document["pages"] = processed_pages

        return processed_document