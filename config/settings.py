from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
from pathlib import Path

class Settings(BaseSettings):
    # LLM Providers
    openai_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    nvidia_nim_api_key: Optional[str] = None

    # Models
    planner_model: str = "gemini/gemini-1.5-pro"
    code_writer_model: str = "gpt-4o"
    reviewer_model: str = "gpt-4o"
    rag_query_model: str = "nvidia_nim/meta/llama-3.1-8b-instruct"
    pr_writer_model: str = "gemini/gemini-1.5-flash"

    # GitHub
    github_token: Optional[str] = None
    github_username: Optional[str] = None

    # Paths
    qdrant_path: str = "./data/qdrant"
    db_path: str = "./data/agent.db"
    sandbox_work_dir: str = "./data/runs"
    
    # Limits
    sandbox_timeout_seconds: int = 120
    max_retries: int = 3
    max_cost_usd: float = 5.00

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

# Global singleton
settings = Settings()

# Ensure directories exist
Path(settings.qdrant_path).mkdir(parents=True, exist_ok=True)
Path(settings.db_path).parent.mkdir(parents=True, exist_ok=True)
Path(settings.sandbox_work_dir).mkdir(parents=True, exist_ok=True)
