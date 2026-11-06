from database.connection import SessionLocal
from database.repositories import DocumentRepository
from ingestion.ingestion_service import DocumentIngestionService
from retrieval.vector_store import ChromaVectorStore


def main():
    session = SessionLocal()

    try:
        repository = DocumentRepository(session)

        doi = "10.59543/jidmis.v3i.17135"

        document = repository.get_document_by_doi(doi)

        if not document:
            raise ValueError("Document not found in the database.")

        print(f"Document: {document.title}")
        print(f"Document ID: {document.id}")
        print("\nStarting ingestion pipeline...\n")

        ingestion_service = DocumentIngestionService(
            repository=repository
        )

        ingestion_service.ingest_document(document.id)

        session.refresh(document)

        print("\nIngestion completed successfully!")
        print(f"Document status: {document.status.value}")

        vector_store = ChromaVectorStore()

        print(f"Total chunks in ChromaDB: {vector_store.count()}")

    finally:
        session.close()


if __name__ == "__main__":
    main()