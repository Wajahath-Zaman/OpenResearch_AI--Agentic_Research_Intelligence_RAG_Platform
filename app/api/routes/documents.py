from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.api.schemas.document import DocumentResponse
from database.repositories import DocumentRepository
from app.api.dependencies.database import get_db


from pathlib import Path
from uuid import uuid4

from fastapi import File, UploadFile

from config.settings import PROJECT_ROOT
from database.models import DocumentStatus

from ingestion.ingestion_service import DocumentIngestionService
from app.api.schemas.document import DocumentIngestionResponse

from ingestion.pdf_download_service import (
    PDFDownloadService,
    PDFDownloadError
)

router = APIRouter(prefix="/documents", tags=["Documents"])

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB


@router.post(
    "/upload",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED
)
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    # Validate filename and extension
    if not file.filename or Path(file.filename).suffix.lower() != ".pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed."
        )

    # Read and validate file size
    contents = await file.read(MAX_FILE_SIZE + 1)

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="File exceeds the maximum allowed size of 20 MB."
        )

    # Validate PDF signature
    if not contents.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=400,
            detail="Invalid PDF file."
        )

    # Create storage directory
    upload_directory = PROJECT_ROOT / "data" / "raw"
    upload_directory.mkdir(parents=True, exist_ok=True)

    # Generate a unique filename
    file_path = upload_directory / f"{uuid4()}.pdf"

    try:
        file_path.write_bytes(contents)

        repository = DocumentRepository(db)

        document = repository.create_document(
            title=Path(file.filename).stem[:500],
            authors=[],
            abstract=None,
            publication_year=None,
            doi=None,
            openalex_id=None,
            source_url=None,
            file_path=str(file_path.resolve()),
            status=DocumentStatus.DISCOVERED
        )

    except Exception:
        file_path.unlink(missing_ok=True)
        raise

    return document

@router.get("/", response_model=list[DocumentResponse])
def list_documents(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=500),
    db: Session = Depends(get_db)
):
    repository = DocumentRepository(db)

    return repository.get_all_documents(
        skip=skip,
        limit=limit
    )


@router.get("/{document_id}", response_model=DocumentResponse)
def get_document(
    document_id: UUID,
    db: Session = Depends(get_db)
):
    repository = DocumentRepository(db)

    document = repository.get_document_by_id(document_id)

    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found."
        )

    return document


@router.post(
    "/{document_id}/download",
    status_code=status.HTTP_200_OK
)
def download_document(
    document_id: UUID,
    db: Session = Depends(get_db)
):
    repository = DocumentRepository(db)

    download_service = PDFDownloadService(
        repository=repository
    )

    try:
        result = download_service.download_pdf(document_id)
        return result

    except PDFDownloadError as error:

        message = str(error)

        if message == "Document not found.":
            raise HTTPException(
                status_code=404,
                detail=message
            )

        if "No openly accessible PDF" in message:
            raise HTTPException(
                status_code=404,
                detail=message
            )

        raise HTTPException(
            status_code=502,
            detail=message
        )


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(
    document_id: UUID,
    db: Session = Depends(get_db)
):
    repository = DocumentRepository(db)

    deleted = repository.delete_document(document_id)

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found."
        )

    return None

@router.post(
    "/{document_id}/ingest",
    response_model=DocumentIngestionResponse,
    status_code=status.HTTP_200_OK
)
def ingest_document(
    document_id: UUID,
    db: Session = Depends(get_db)
):
    repository = DocumentRepository(db)

    ingestion_service = DocumentIngestionService(
        repository=repository
    )

    try:
        result = ingestion_service.ingest_document(document_id)
        return result

    except ValueError as error:
        if str(error) == "Document not found in database.":
            raise HTTPException(
                status_code=404,
                detail=str(error)
            )

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Document ingestion failed."
        )