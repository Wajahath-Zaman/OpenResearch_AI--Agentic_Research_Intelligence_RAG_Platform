from fastapi import APIRouter, HTTPException

from app.api.schemas.chat import ChatRequest, ChatResponse

from agent.graph import build_agent_graph

from agent.dependencies import (
    create_answer_generator,
    create_research_discovery_service,
    create_intent_classifier
)


router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("/", response_model=ChatResponse)
def chat(request: ChatRequest):

    try:
        agent = build_agent_graph(
            answer_generator=create_answer_generator(),
            discovery_service=create_research_discovery_service(),
            intent_classifier=create_intent_classifier()
        )

        initial_state = {
            "query": request.query,
            "route": "",
            "retrieved_chunks": [],
            "context": "",
            "research_results": [],
            "messages": [],
            "answer": "",
            "citations": [],
            "error": None,
            "document_ids": request.document_ids,
            "comparison_results": []
        }

        result = agent.invoke(initial_state)

        if result.get("error"):
            raise HTTPException(
                status_code=500,
                detail=result["error"]
            )

        return {
            "query": request.query,
            "route": result["route"],
            "answer": result["answer"],
            "citations": result.get("citations", []),
            "research_results": result.get("research_results", []),
            "comparison_results": result.get("comparison_results", []),
            "error": None
        }

    except HTTPException:
        raise

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="The research assistant encountered an error."
        )