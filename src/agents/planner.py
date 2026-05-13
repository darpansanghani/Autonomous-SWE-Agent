import json
from typing import List, Literal, Optional
from pydantic import BaseModel

from src.llm.router import agent_completion
from src.github_client.issue_reader import GitHubIssue

class SubTask(BaseModel):
    id: int
    description: str
    type: Literal["search", "write", "test", "docs"]
    depends_on: List[int]
    target_files: List[str]
    acceptance_criteria: str

class Plan(BaseModel):
    issue_summary: str
    root_cause_hypothesis: Optional[str]
    affected_languages: List[str]
    sub_tasks: List[SubTask]
    files_likely_involved: List[str]
    complexity: Literal["trivial", "low", "medium", "high"]
    estimated_retries_budget: int
    test_strategy: str

PROMPT = """You are a senior software engineer analyzing a GitHub issue.
You have access to the repository structure below.

REPO MAP:
{repo_map}

ISSUE:
Title: {issue.title}
Body: {issue.body}
Labels: {issue.labels}
Comments: {issue.comments}

YOUR TASK:
1. Summarize what needs to be done.
2. Break the work into ordered sub-tasks.
3. For each sub-task, specify what type of work it is, which files are involved, and acceptance criteria.
4. Estimate complexity (trivial/low/medium/high).

RULES:
- Always include a "test" sub-task.
- Order tasks by dependency (search before write).
- Be specific about file paths using the repo map.
"""

async def create_plan(issue: GitHubIssue, repo_map: dict) -> Plan:
    """Parses issue and repo map into a structured Plan."""
    
    # ensure repo map string isn't too huge for the prompt
    repo_map_str = json.dumps(repo_map, indent=2)
    if len(repo_map_str) > 100000:
        repo_map_str = repo_map_str[:100000] + "\n... (truncated)"
        
    messages = [
        {"role": "system", "content": "You are a senior technical lead."},
        {"role": "user", "content": PROMPT.format(
            repo_map=repo_map_str,
            issue=issue
        )}
    ]
    
    # Let litellm handle parsing JSON matching the Pydantic schema
    response = await agent_completion(
        agent_name="planner",
        messages=messages,
        response_format={"type": "json_object"}
    )
    
    # Parse the raw JSON into our Pydantic model
    raw_json = response.choices[0].message.content
    try:
        return Plan.model_validate_json(raw_json)
    except Exception as e:
        # fallback if LLM gave malformed json
        raise ValueError(f"Failed to parse LLM plan: {e}\nRaw output: {raw_json}")
