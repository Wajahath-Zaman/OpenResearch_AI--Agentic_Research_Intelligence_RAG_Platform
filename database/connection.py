from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.orm import sessionmaker

from config.settings import get_settings

settings = get_settings()

# Ensure the SQLite database directory exists
database_url = make_url(settings.database_url)

if database_url.get_backend_name() == "sqlite":
    database_path = database_url.database

    if database_path and database_path != ":memory:":
        Path(database_path).parent.mkdir(
            parents=True,
            exist_ok=True
        )

# Creating Database Engine
engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False}
    if database_url.get_backend_name() == "sqlite"
    else {}
)

# Creating Session Factory
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False
)

# Database Dependency
def get_db():
    db = SessionLocal()

    try:
        yield db

    finally:
        db.close()