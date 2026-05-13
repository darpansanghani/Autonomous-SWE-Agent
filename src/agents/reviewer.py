import json
from typing import List, Literal, Optional
from pydantic import BaseModel

from src.llm.router import agent_completion
from src.github_client.issue_reader import GitHubIssue

class ReviewCriterion(BaseModel):
    name: str
    score: int          # 1-5
    weight: float       # percentage
    feedback: str       # specific reasoning

class ReviewDecision(BaseModel):
    criteria: List[ReviewCriterion]
    weighted_score: float
    verdict: Literal["approve", "revise", "reject"]
    overall_feedback: str
    retry_suggestion: Optional[str]
    security_flags: List[str]

PROMPT = """You are a senior code reviewer evaluating changes made by an AI agent.

ORIGINAL ISSUE:
{issue_summary}

CODE CHANGES:
{diffs}

TEST RESULTS:
- Passed: {test_passed}/{test_total}
- Failures: {test_failures}

PREVIOUS FEEDBACK:
{history}

EVALUATE the changes on these criteria (score each 1-5):
1. tests_pass (30%): Do all tests pass?
2. requirements_met (25%): Does the code fully address the issue?
3. no_regressions (20%): Did existing tests still pass?
4. style_consistency (15%): Does code match repo conventions?
5. security (10%): Any dangerous patterns (eval, exec, hardcoded secrets)?

If verdict is "revise", provide a SPECIFIC retry_suggestion telling the Code Writer exactly what to fix.
"""

async def review_changes(
    issue: GitHubIssue,
    code_changes: List[Any], # List[FileOperation]
    test_results: Any,       # TestResults
    feedback_history: List[str],
    retry_count: int
) -> ReviewDecision:
    """Evaluates code changes and decides if we should merge, retry, or give up."""
    
    diffs = "\n".join([c.unified_diff or c.full_content or "" for c in code_changes])
    history = "\n".join(feedback_history) if feedback_history else "None"
    
    failures = ""
    if test_results and test_results.failures_json:
        failures = test_results.failures_json
        
    messages = [
        {"role": "system", "content": "You are a senior reviewer."},
        {"role": "user", "content": PROMPT.format(
            issue_summary=issue.title,
            diffs=diffs,
            test_passed=test_results.passed if test_results else 0,
            test_total=test_results.total if test_results else 0,
            test_failures=failures,
            history=history
        )}
    ]
    
    response = await agent_completion(
        agent_name="reviewer",
        messages=messages,
        response_format={"type": "json_object"}
    )
    
    raw = response.choices[0].message.content
    decision = ReviewDecision.model_validate_json(raw)
    
    # Hard overrides
    if decision.security_flags:
        decision.verdict = "reject"
        decision.overall_feedback = f"SECURITY REJECT: {decision.security_flags}"
        
    return decision
