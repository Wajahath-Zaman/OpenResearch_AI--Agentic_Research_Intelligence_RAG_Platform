from pydantic import BaseModel, Field
from app.api.schemas.document import DocumentResponse


class ResearchDiscoveryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    per_page: int = Field(default=10, ge=1, le=100)
    page: int = Field(default=1, ge=1)
    save_to_database: bool = True


class ResearchDiscoveryResponse(BaseModel):
    query: str
    total_count: int
    page: int
    per_page: int
    returned_count: int
    skipped_count: int
    saved_count: int
    duplicate_count: int
    unidentifiable_count: int
    results: list[dict] = Field(default_factory=list)
    saved_documents: list[DocumentResponse] = Field(default_factory=list)
    duplicate_documents: list[DocumentResponse] = Field(default_factory=list)
    unidentifiable_documents: list[DocumentResponse] = Field(default_factory=list)