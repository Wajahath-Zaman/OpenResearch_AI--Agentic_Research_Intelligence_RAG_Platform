from fastapi import APIRouter

from app.api.routes import health, documents, research, chat, comparison

api_router = APIRouter()

api_router.include_router(health.router, tags=["Health"])
api_router.include_router(documents.router)
api_router.include_router(research.router)
api_router.include_router(chat.router)
api_router.include_router(comparison.router)