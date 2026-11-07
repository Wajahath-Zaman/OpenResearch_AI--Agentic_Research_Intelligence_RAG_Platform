from fastapi import FastAPI

from config.settings import get_settings
from app.api.router import api_router

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "An AI-powered research assistant for discovering, "
        "analysing, and comparing scientific research papers."
    ),
    debug=settings.debug
)

app.include_router(
    api_router,
    prefix=settings.api_prefix
)