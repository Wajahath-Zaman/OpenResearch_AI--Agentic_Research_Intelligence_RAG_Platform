import pytest

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database.base import Base
from database.models import Document
from database.repositories import DocumentRepository

from ingestion.research_discovery import OpenAlexClient
from ingestion.discovery_service import ResearchDiscoveryService


@pytest.fixture
def db_session():

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False}
    )

    Base.metadata.create_all(bind=engine)

    TestingSessionLocal = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False
    )

    session = TestingSessionLocal()

    yield session

    session.close()
    engine.dispose()


@pytest.fixture
def discovery_service(db_session):

    client = OpenAlexClient()

    repository = DocumentRepository(db_session)

    return ResearchDiscoveryService(
        client=client,
        repository=repository
    )


@pytest.mark.integration
def test_real_openalex_discovery(discovery_service):

    result = discovery_service.discover(
        query="transformer neural networks",
        per_page=5
    )

    assert result["returned_count"] > 0
    assert len(result["results"]) > 0

    for paper in result["results"]:

        assert paper["title"]
        assert isinstance(paper["authors"], list)


@pytest.mark.integration
def test_real_discovery_and_database_persistence(discovery_service, db_session):

    result = discovery_service.discover_and_store(
        query="transformer neural networks",
        per_page=5
    )

    assert result["saved_count"] > 0

    documents = (
        db_session.query(Document)
        .all()
    )

    assert len(documents) == result["saved_count"]


@pytest.mark.integration
def test_duplicate_prevention_with_real_database(discovery_service, db_session):

    first_result = discovery_service.discover_and_store(
        query="transformer neural networks",
        per_page=5
    )

    assert first_result["saved_count"] > 0

    initial_count = (
        db_session.query(Document)
        .count()
    )

    second_result = discovery_service.discover_and_store(
        query="transformer neural networks",
        per_page=5
    )

    final_count = (
        db_session.query(Document)
        .count()
    )

    assert final_count == initial_count
    assert second_result["duplicate_count"] > 0