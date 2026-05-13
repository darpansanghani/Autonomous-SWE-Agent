from sqlmodel import SQLModel, Field
from datetime import datetime
from typing import Optional
import uuid

class Run(SQLModel, table=True):
    """One agent run = one GitHub issue attempt."""
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8], primary_key=True)
    repo_url: str
    issue_number: int
    issue_title: str
    status: str                     # "running" | "success" | "failed" | "human_review"
    plan_json: Optional[str] = None # serialized Plan
    total_cost_usd: float = 0.0
    retry_count: int = 0
    pr_url: Optional[str] = None
    duration_seconds: Optional[float] = None
    primary_model: str              # which LLM was used for Code Writer
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

class RunLog(SQLModel, table=True):
    """Individual log entries for a run (persists the live log)."""
    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: str = Field(foreign_key="run.id")
    timestamp: datetime
    event: str                      
    emoji: str
    data_json: Optional[str] = None # serialized event data dict
    message: str                    # human-readable log line

class FileChange(SQLModel, table=True):
    """Files changed in a run (for diff viewer)."""
    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: str = Field(foreign_key="run.id")
    file_path: str
    action: str                     # "create" | "modify" | "delete"
    unified_diff: Optional[str] = None
    full_content: Optional[str] = None

class TestResult(SQLModel, table=True):
    """Test execution results."""
    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: str = Field(foreign_key="run.id")
    attempt: int                    # 1, 2, 3 (which retry loop)
    command_run: str
    exit_code: int
    total: int
    passed: int
    failed: int
    duration_seconds: float
    failures_json: Optional[str] = None  # serialized List[TestFailure]
    stdout: Optional[str] = None
    stderr: Optional[str] = None
