from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict



PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):

    # Application
    app_name: str = "Open Research Intelligence Assistant"
    app_version: str = "1.0.0"
    app_environment: str = "development"
    debug: bool = False
    # Above code is config settings that we can toggle.

    # API
    api_prefix: str = "/api/v1"

    # Database
    database_url: str = (
        f"sqlite:///{(PROJECT_ROOT / 'data' / 'app.db').as_posix()}"
    )

    # LLM Providers
    groq_api_key: str | None = None
    groq_model: str = "openai/gpt-oss-20b"

    gemini_api_key: str | None = None
    gemini_model: str = "gemini-3.1-flash-lite"

    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Vector Database
    chroma_persist_directory: str = str(
        PROJECT_ROOT / "data" / "chroma"
    )

    # MLflow
    mlflow_tracking_uri: str = str(
        PROJECT_ROOT / "mlruns"
    )

    # Logging
    log_level: str = "INFO"


    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

@lru_cache
def get_settings() -> Settings:
    return Settings()