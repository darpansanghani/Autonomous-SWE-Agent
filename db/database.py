from sqlmodel import create_engine, SQLModel, Session
import os
import sys

# Hack: we need to find settings correctly regardless of where script runs from
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.settings import settings

# Import models so SQLModel knows about them before create_all
from db.models import Run, RunLog, FileChange, TestResult

# sqlite requires this exact format
DATABASE_URL = f"sqlite:///{os.path.abspath(settings.db_path)}"

engine = create_engine(DATABASE_URL, echo=False)

def init_db():
    """Create all tables in the SQLite database."""
    SQLModel.metadata.create_all(engine)

def get_session() -> Session:
    """Yields a database session."""
    return Session(engine)
