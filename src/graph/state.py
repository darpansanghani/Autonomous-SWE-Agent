from typing import TypedDict, List, Optional, Any, Literal
from dataclasses import dataclass
from datetime import datetime

from src.github_client.issue_reader import GitHubIssue

# We define the basic data structures here to avoid circular imports.
# Full Pydantic models for agent outputs will live in their respective agent files.

class AgentState(TypedDict):
    """The central state dictionary that flows through the LangGraph workflow."""
    
    # ── Input ──────────────────────────────
    issue: GitHubIssue
    repo_url: str
    repo_local_path: str
    target_languages: List[str]
    
    # ── Planning ────────────────────────────
    plan: Optional[Any]            # Plan (Pydantic model)
    current_task_index: int
    
    # ── RAG ─────────────────────────────────
    repo_map: Optional[dict]       # JSON structural map
    code_context: List[Any]        # List[CodeChunk]
    qdrant_collection: Optional[str]
    language_profile: Optional[Any] # RepoLanguageProfile
    
    # ── Code Generation ─────────────────────
    code_changes: List[Any]        # List[FileOperation]
    applied_patch: bool
    
    # ── Testing ─────────────────────────────
    test_results: Optional[Any]    # TestResults
    
    # ── Review ──────────────────────────────
    review: Optional[Any]          # ReviewDecision
    retry_count: int
    feedback_history: List[str]    # accumulates feedback from Reviewer
    
    # ── Output ──────────────────────────────
    pr_url: Optional[str]
    status: Literal["running", "success", "failed", "human_review"]
    total_cost_usd: float
    run_id: str
