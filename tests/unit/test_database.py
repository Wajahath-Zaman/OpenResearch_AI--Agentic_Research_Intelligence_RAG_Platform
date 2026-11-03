from sqlalchemy import text

from database.connection import engine, SessionLocal
from database.base import Base

import uuid
import pytest

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from database.base import Base
from database.models import Document, DocumentStatus


from database.repositories import DocumentRepository


def test_database_engine_initialization():
    assert engine is not None


def test_database_connection():
    with engine.connect() as connection:
        result = connection.execute(text("SELECT 1"))

        assert result.scalar() == 1

def test_session_creation():
    db = SessionLocal()

    try:
        assert db is not None

    finally:
        db.close()


def test_base_initialization():
    assert Base is not None


# Testing the database engine

@pytest.fixture
def db_session():

    test_engine = create_engine(
        "sqlite:///:memory:",
        connect_args = {"check_same_thread": False},
        poolclass=StaticPool
    )

    Base.metadata.create_all(bind=test_engine)


    testingSessionLocal = sessionmaker(
        bind=test_engine,
        autoflush=False,
        autocommit=False
    )

    session = testingSessionLocal()

    try:
        yield session

    finally:
        session.close()
        Base.metadata.drop_all(bind=test_engine)
        test_engine.dispose()



def test_document_table_exists():
    assert "documents" in Base.metadata.tables


def test_document_creation(db_session):

    document  = Document(
        title="Attention is all you need",
        authors=["Ashish Vaswani", "Noam Shazeer"],
        publication_year = 2017,
        doi="10.48550/arXiv.1706.03762"
    )

    db_session.add(document)
    db_session.commit()
    db_session.refresh(document)

    assert document.id is not None
    assert isinstance(document.id, uuid.UUID)

    assert document.title == 'Attention is all you need'

    assert document.publication_year == 2017

    assert document.status == DocumentStatus.DISCOVERED


def test_document_unique_doi(db_session):

    document1 = Document(
        title="Research Paper One",
        doi="10.1234/example"
    )

    document2 = Document(
        title="Research Paper Two",
        doi="10.1234/example"
    )

    db_session.add(document1)
    db_session.commit()

    db_session.add(document2)

    with pytest.raises(Exception):
        db_session.commit()

    db_session.rollback()



def test_document_optional_fields(db_session):

    document = Document(
        title="Research Paper Without DOI"
    )


    db_session.add(document)
    db_session.commit()

    assert document.doi is None
    assert document.abstract is None
    assert document.file_path is None



# Testing Repositories

def test_create_document(db_session):

    repository = DocumentRepository(db_session)

    document = repository.create_document(
        title="Attention is All you Need.",
        authors=["Ashish Vaswani"],
        publication_year=2017,
        doi="10.48550/example1"
    )

    assert document.id is not None
    assert document.title == "Attention is All you Need."


def test_get_document_by_id(db_session):

    repository = DocumentRepository(db_session)

    created_document = repository.create_document(
        title = "Research Paper One"
    )

    retrieved_document = repository.get_document_by_id(
        created_document.id
    )

    assert retrieved_document is not None
    assert retrieved_document.id == created_document.id


def test_get_document_by_doi(db_session): 

    repository = DocumentRepository(db_session)

    repository.create_document(
        title="Research Paper Two",
        doi="10.1234/example2"
    )

    document = repository.get_document_by_doi(
        "10.1234/example2"
    )

    assert document is not None
    assert document.title == "Research Paper Two"


def test_get_all_documents(db_session):

    repository = DocumentRepository(db_session)

    repository.create_document(title="Paper One")
    repository.create_document(title="Paper Two")
    repository.create_document(title="Paper Three")

    documents = repository.get_all_documents()

    assert len(documents) == 3


def test_get_document_by_status(db_session):

    repository = DocumentRepository(db_session)

    document = repository.create_document(
        title="Paper One"
    )

    repository.update_document_status(
        document.id,
        DocumentStatus.DOWNLOADED
    )

    documents = repository.get_document_by_status(
        DocumentStatus.DOWNLOADED
    )

    assert len(documents) == 1
    assert documents[0].id == document.id


def test_update_document_status(db_session):

    repository = DocumentRepository(db_session)

    document = repository.create_document(
        title="Paper One"
    )

    updated_document = repository.update_document_status(
        document.id,
        DocumentStatus.PROCESSED
    )

    assert updated_document.status == DocumentStatus.PROCESSED



def test_delete_document(db_session):

    repository = DocumentRepository(db_session)

    document = repository.create_document(
        title='Paper To Delete'
    )

    result = repository.delete_document(document.id)

    assert result is True

    retrieved_document = repository.get_document_by_id(
        document.id
    )

    assert retrieved_document is None

def test_get_nonexistent_document(db_session):

    repository = DocumentRepository(db_session)

    document = repository.get_document_by_id(uuid.uuid4())

    assert document is None


def test_delete_nonexistent_document(db_session):

    repository = DocumentRepository(db_session)

    result = repository.delete_document(uuid.uuid4())

    assert result is False


def test_transaction_rollback(db_session):

    repository = DocumentRepository(db_session)

    # Insert a document with a DOI
    repository.create_document(
        title="Original Research Paper",
        doi="10.1234/duplicate"
    )

    # Attempt To insert another document with the same DOI

    with pytest.raises(Exception):
        repository.create_document(
            title="Duplicate Research paper",
            doi="10.1234/duplicate"
        )

        # Verify that the session is usable after rollback
        document = repository.create_document(
            title="Another Research Paper",
            doi="10.1234/unique"
        )

        assert document.id is not None
        assert document.title == "Another Research Paper"