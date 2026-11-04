import pytest

from ingestion.preprocessing import TextPreprocessor


@pytest.fixture
def preprocessor():
    return TextPreprocessor()


def test_removes_extra_spaces(preprocessor):

    text = "This   is    a     research paper."

    result = preprocessor.preprocess_text(text)

    assert result == "This is a research paper."


def test_removes_null_characters(preprocessor):

    text = "Research\x00 paper"

    result = preprocessor.preprocess_text(text)

    assert result == "Research paper"


def test_fixes_hyphenated_line_breaks(preprocessor):

    text = "This is an inter-\nnational research paper."

    result = preprocessor.preprocess_text(text)

    assert result == "This is an international research paper."


def test_preserves_paragraph_structure(preprocessor):

    text = "First paragraph.\n\nSecond paragraph."

    result = preprocessor.preprocess_text(text)

    assert result == "First paragraph.\n\nSecond paragraph."


def test_preserves_page_numbers_and_metadata(preprocessor):

    document = {
        "source_path": "sample.pdf",
        "metadata": {
            "title": "Research Paper",
            "author": "Researcher",
            "total_pages": 2
        },
        "pages": [
            {
                "page_number": 1,
                "text": "First   page."
            },
            {
                "page_number": 2,
                "text": "Second   page."
            }
        ]
    }

    result = preprocessor.preprocess_document(document)

    assert result["metadata"] == document["metadata"]
    assert result["source_path"] == document["source_path"]

    assert result["pages"][0]["page_number"] == 1
    assert result["pages"][0]["text"] == "First page."

    assert result["pages"][1]["page_number"] == 2
    assert result["pages"][1]["text"] == "Second page."

    # Original document must remain unchanged
    assert document["pages"][0]["text"] == "First   page."