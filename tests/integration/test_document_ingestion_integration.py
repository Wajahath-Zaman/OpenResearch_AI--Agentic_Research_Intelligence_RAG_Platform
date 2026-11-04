import math

import fitz
import numpy as np
import pytest

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database.base import Base
from database.repositories import DocumentRepository
from database.models import Document, DocumentStatus

from ingestion.ingestion_service import DocumentIngestionService
from ingestion.chunking import TextChunker
from ingestion.embedding_generator import EmbeddingGenerator
from retrieval.vector_store import ChromaVectorStore


class MockEmbeddingModel:

    def encode(
        self,
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    ):

        vector = np.full(384, 1 / math.sqrt(384))

        return np.array([
            vector.copy() for _ in texts
        ])


@pytest.fixture
def ingestion_environment(tmp_path):

    database_path = tmp_path / "test.db"

    engine = create_engine(
        f"sqlite:///{database_path}",
        connect_args={"check_same_thread": False}
    )

    Base.metadata.create_all(engine)

    TestingSessionLocal = sessionmaker(
        bind=engine,
        expire_on_commit=False
    )

    session = TestingSessionLocal()

    repository = DocumentRepository(session)

    pdf_path = tmp_path / "research_paper.pdf"

    pdf = fitz.open()

    page1 = pdf.new_page()
    page1.insert_text(
        (50, 50),
        " ".join(f"research{i}" for i in range(12))
    )

    page2 = pdf.new_page()
    page2.insert_text(
        (50, 50),
        " ".join(f"machinelearning{i}" for i in range(8))
    )

    pdf.set_metadata({
        "title": "Integration Research Paper",
        "author": "Research Team"
    })

    pdf.save(pdf_path)
    pdf.close()

    document = repository.create_document(
        title="Integration Research Paper",
        authors=["Research Team"],
        file_path=str(pdf_path)
    )

    vector_store = ChromaVectorStore(
        persist_directory=tmp_path / "chroma"
    )

    embedding_generator = EmbeddingGenerator(
        model=MockEmbeddingModel()
    )

    service = DocumentIngestionService(
        repository=repository,
        chunker=TextChunker(
            chunk_size=10,
            chunk_overlap=2
        ),
        embedding_generator=embedding_generator,
        vector_store=vector_store
    )

    yield {
        "service": service,
        "repository": repository,
        "vector_store": vector_store,
        "document": document
    }

    session.close()
    engine.dispose()


def test_complete_document_ingestion(ingestion_environment):

    service = ingestion_environment["service"]
    repository = ingestion_environment["repository"]
    vector_store = ingestion_environment["vector_store"]
    document = ingestion_environment["document"]

    result = service.ingest_document(document.id)

    # Verify ingestion response
    assert result["document_id"] == str(document.id)
    assert result["status"] == DocumentStatus.PROCESSED.value
    assert result["total_chunks"] == 3

    # Verify SQLite status
    updated_document = repository.get_document_by_id(document.id)

    assert updated_document.status == DocumentStatus.PROCESSED

    # Verify ChromaDB storage
    stored_chunks = vector_store.get_document_chunks(
        str(document.id)
    )

    assert len(stored_chunks) == 3

    # Verify page-level references
    page_numbers = {
        chunk["metadata"]["page_number"]
        for chunk in stored_chunks
    }

    assert page_numbers == {1, 2}

    # Verify embeddings and text
    assert all(
        len(chunk["text"].split()) > 0
        for chunk in stored_chunks
    )