from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from config.settings import get_settings



settings = get_settings()

# Creating Database Engine
engine = create_engine(
    settings.database_url,
    connect_args={'check_same_thread': False} # Request Handling
    if settings.database_url.startswith("sqlite")
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