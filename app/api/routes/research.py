from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.dependencies.database import get_db
from app.api.schemas.research import (
    ResearchDiscoveryRequest,
    ResearchDiscoveryResponse
)

from database.repositories import DocumentRepository
from ingestion.discovery_service import ResearchDiscoveryService


router = APIRouter(prefix="/research", tags=["Research"])


@router.post(
    "/discover",
    response_model=ResearchDiscoveryResponse
)
def discover_research(
    request: ResearchDiscoveryRequest,
    db: Session = Depends(get_db)
):
    repository = DocumentRepository(db)

    service = ResearchDiscoveryService(
        repository=repository
    )

    try:
        if request.save_to_database:
            result = service.discover_and_store(
                query=request.query,
                per_page=request.per_page,
                page=request.page
            )

        else:
            result = service.discover(
                query=request.query,
                per_page=request.per_page,
                page=request.page
            )

            result.update({
                "saved_count": 0,
                "duplicate_count": 0,
                "unidentifiable_count": 0,
                "saved_documents": []
            })

        return result

    except Exception:
        raise HTTPException(
            status_code=502,
            detail="Research discovery service failed."
        )