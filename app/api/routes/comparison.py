from uuid import UUID

from fastapi import APIRouter, HTTPException

from app.api.schemas.comparison import (
    PaperComparisonRequest,
    PaperComparisonResponse
)

from agent.nodes.comparison import PaperComparisonWorkflow


router = APIRouter(
    prefix="/research",
    tags=["Research"]
)


@router.post(
    "/compare",
    response_model=PaperComparisonResponse
)
def compare_papers(request: PaperComparisonRequest):

    # Validate document UUIDs
    try:
        document_ids = [
            str(UUID(document_id))
            for document_id in request.document_ids
        ]
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="One or more document IDs are invalid."
        )

    try:
        workflow = PaperComparisonWorkflow()

        result = workflow.compare(
            query=request.query,
            document_ids=document_ids
        )

        return {
            "comparison": result["comparison"],
            "comparison_results": result["comparison_results"]
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Paper comparison failed."
        )