import json
from typing import List, Literal, Optional, Any
from pydantic import BaseModel

from src.llm.router import agent_completion
from src.github_client.issue_reader import GitHubIssue

class ReviewCriterion(BaseModel):
    name: str
    score: int          # 1-5
    weight: float       # percentage weight (e.g. 0.3)
    feedback: str       # specific evaluation details

class ReviewDecision(BaseModel):
    criteria: List[ReviewCriterion]
    weighted_score: float             # scale of 1-10
    verdict: Literal["approve", "revise", "reject"]
    overall_feedback: str
    retry_suggestion: Optional[str] = None
    security_flags: List[str] = []

PROMPT = """You are a senior codebase reviewer evaluating code changes made by an autonomous software agent.

ORIGINAL ISSUE DETAILS:
{issue_summary}

PROPOSED CODE CHANGES:
{diffs}

SANDBOX TEST RUN RESULTS:
- Total Ran: {test_total}
- Passed: {test_passed}
- Failed: {test_failed}
- Failure Diagnostic Logs: {test_failures}

PREVIOUS RETRY ATTEMPTS HISTORY:
{history}

EVALUATE the changes on these five criteria (score each 1-5):
1. tests_pass (weight 0.30): Do all tests pass? Were tests successfully run?
2. requirements_met (weight 0.25): Does the code completely address the issue requirements?
3. no_regressions (weight 0.20): Did existing parts of the codebase remain unharmed?
4. style_consistency (weight 0.15): Does the code strictly follow established clean code practices?
5. security (weight 0.10): Is the code free of risk?

CRITICAL SECURITY RED FLAGS (Immediately flag in "security_flags" and set verdict to "reject"):
- Use of os.system() or subprocess calls without parameterization (shell=True)
- eval() or exec() execution blocks
- Hardcoded API keys, database credentials, passwords, or tokens
- Raw SQL string concatenations (SQL injection vulnerability)
- Path traversal vulnerabilities (opening files without resolve check)

VERDICT GUIDELINES:
- If any security flags are found -> verdict: "reject"
- If all tests pass and weighted score is >= 7.0 -> verdict: "approve"
- If tests fail, or weighted score is < 7.0, and we have retries remaining -> verdict: "revise" (provide a clear, actionable instruction in "retry_suggestion")
- If retry count has reached maximum capacity and we still fail -> verdict: "reject"

Return a JSON object conforming exactly to the ReviewDecision schema:
{{
  "criteria": [
    {{"name": "tests_pass", "score": 5, "weight": 0.30, "feedback": "..."}}
  ],
  "weighted_score": 8.5,
  "verdict": "approve",
  "overall_feedback": "...",
  "retry_suggestion": null,
  "security_flags": []
}}
"""

async def review_changes(
    issue: GitHubIssue,
    code_changes: List[Any],
    test_results: Any,
    feedback_history: List[str],
    retry_count: int
) -> ReviewDecision:
    """Evaluates code changes and decides if we should merge, retry, or give up."""
    
    diffs = "\n".join([getattr(c, "unified_diff", "") or getattr(c, "full_content", "") or "" for c in code_changes])
    history = "\n".join([f"Attempt {i+1}: {f}" for i, f in enumerate(feedback_history)]) if feedback_history else "No previous attempts."
    
    failures = ""
    test_failed = 0
    test_passed = 0
    test_total = 0
    
    if test_results:
        test_passed = test_results.passed
        test_failed = test_results.failed
        test_total = test_results.total
        if test_results.failures:
            failures = "\n".join([f"- {f.test_name}: {f.error_message} (File: {f.file_path}:{f.line_number})" for f in test_results.failures])
        else:
            failures = test_results.stderr or "No diagnostic log."
        
    messages = [
        {"role": "system", "content": "You are a professional, senior software reviewer grading AI contributions."},
        {"role": "user", "content": PROMPT.format(
            issue_summary=f"Title: {issue.title}\nBody: {issue.body}",
            diffs=diffs,
            test_passed=test_passed,
            test_failed=test_failed,
            test_total=test_total,
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
    data = json.loads(raw)
    
    # Calculate weighted score explicitly to prevent LLM math errors
    criteria = []
    total_weight = 0.0
    weighted_sum = 0.0
    
    for c in data.get("criteria", []):
        criterion = ReviewCriterion(**c)
        criteria.append(criterion)
        weighted_sum += criterion.score * criterion.weight
        total_weight += criterion.weight
        
    calculated_score = (weighted_sum / total_weight) * 2.0 if total_weight > 0 else 0.0
    
    # Override LLM outputs with calculations
    data["weighted_score"] = round(calculated_score, 1)
    data["criteria"] = [c.model_dump() for c in criteria]
    
    decision = ReviewDecision(**data)
    
    # Assert security rejects
    if decision.security_flags:
        decision.verdict = "reject"
        decision.overall_feedback = f"SECURITY AUDIT REJECTED: {', '.join(decision.security_flags)}"
        
    return decision
