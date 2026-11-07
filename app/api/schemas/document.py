from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from database.models import DocumentStatus


class DocumentResponse(BaseModel):

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    title: str
    authors: list
    abstract: str | None
    publication_year: int | None
    doi: str | None
    openalex_id: str | None
    source_url: str | None
    pdf_url: str | None = None
    file_path: str | None
    status: DocumentStatus
    created_at: datetime
    updated_at: datetime

class DocumentIngestionResponse(BaseModel):
    document_id: UUID
    title: str
    total_chunks: int
    status: DocumentStatus