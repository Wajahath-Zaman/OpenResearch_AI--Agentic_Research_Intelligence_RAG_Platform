import pytest

from retrieval.context_assembler import ContextAssembler


@pytest.fixture
def assembler():

    return ContextAssembler()


@pytest.fixture
def sample_chunks():

    return [
        {
            "text": "Machine learning improves medical diagnosis.",
            "metadata": {
                "title": "AI in Healthcare",
                "page_number": 5
            }
        },
        {
            "text": "Deep learning models identify abnormalities.",
            "metadata": {
                "title": "Deep Learning Applications",
                "page_number": 12
            }
        }
    ]


def test_assemble_context(assembler, sample_chunks):

    context = assembler.assemble(sample_chunks)

    assert "Machine learning improves medical diagnosis." in context

    assert "Deep learning models identify abnormalities." in context


def test_context_preserves_source_information(assembler, sample_chunks):

    context = assembler.assemble(sample_chunks)

    assert "AI in Healthcare" in context
    assert "Page: 5" in context

    assert "Deep Learning Applications" in context
    assert "Page: 12" in context


def test_context_preserves_retrieval_order(assembler, sample_chunks):

    context = assembler.assemble(sample_chunks)

    first_position = context.index("Machine learning")
    second_position = context.index("Deep learning")

    assert first_position < second_position


def test_context_contains_source_identifiers(assembler, sample_chunks):

    context = assembler.assemble(sample_chunks)

    assert "--- Source 1 ---" in context
    assert "--- Source 2 ---" in context


def test_empty_chunks_return_empty_context(assembler):

    context = assembler.assemble([])

    assert context == ""


def test_invalid_input_type_raises_error(assembler):

    with pytest.raises(TypeError):

        assembler.assemble("invalid input")


def test_invalid_chunk_type_raises_error(assembler):

    with pytest.raises(TypeError):

        assembler.assemble(["invalid chunk"])


def test_missing_chunk_text_raises_error(assembler):

    with pytest.raises(ValueError):

        assembler.assemble([
            {
                "metadata": {
                    "title": "Research Paper"
                }
            }
        ])


def test_invalid_metadata_type_raises_error(assembler):

    with pytest.raises(TypeError):

        assembler.assemble([
            {
                "text": "Research content.",
                "metadata": "invalid"
            }
        ])