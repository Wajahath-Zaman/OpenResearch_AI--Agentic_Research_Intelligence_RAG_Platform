from database.base import Base
from database.connection import engine
from database import models


def main():

    print("Initializing database...")

    Base.metadata.create_all(bind=engine)

    print("Database tables initialized successfully!")

    print("\nAvailable tables:")

    for table in Base.metadata.sorted_tables:
        print(f"- {table.name}")


if __name__ == "__main__":
    main()