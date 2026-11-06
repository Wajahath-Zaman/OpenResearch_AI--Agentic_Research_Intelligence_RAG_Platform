from pathlib import Path

from config.settings import PROJECT_ROOT
from database.connection import SessionLocal
from database.repositories import DocumentRepository
from database.models import DocumentStatus


def main():

    pdf_path = PROJECT_ROOT / "data" / "raw" / "paper1.pdf"

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    session = SessionLocal()

    try:
        repository = DocumentRepository(session)

        doi = "10.59543/jidmis.v3i.17135"

        # Avoid registering the same paper twice.
        existing = repository.get_document_by_doi(doi)

        if existing:
            print("Document already registered.")
            print(f"Document ID: {existing.id}")
            return

        document = repository.create_document(
            title=(
                "A Comparative Analysis of Transformer Architectures "
                "for Automated Lung Cancer Detection in CT Images"
            ),
            authors=[
                "Baris Okmen",
                "Yigitcan Cakmak",
                "Ishak Pacal"
            ],
            abstract=(
                "A comparative analysis of Swin-Base, ViT-Base, "
                "DeiT-Base, and BEiT-Base architectures for "
                "automated lung cancer classification using CT images."
            ),
            publication_year=2026,
            doi=doi,
            openalex_id=None,
            source_url="https://doi.org/10.59543/jidmis.v3i.17135",
            file_path=str(pdf_path.resolve()),
            status=DocumentStatus.DISCOVERED
        )

        print("\nDocument successfully registered!")
        print(f"Document ID: {document.id}")
        print(f"Title: {document.title}")
        print(f"Status: {document.status}")

    finally:
        session.close()


if __name__ == "__main__":
    main()