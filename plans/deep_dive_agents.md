# Deep Dive: Agent Designs

---

## 1. 🧠 Planner Agent (`agents/planner.py`)

### Purpose
Parse a GitHub issue and produce a structured, ordered plan that other agents can execute step-by-step.

### Input
```python
@dataclass
class GitHubIssue:
    number: int
    title: str
    body: str
    labels: List[str]         # ["bug", "enhancement"]
    comments: List[str]       # clarifying comments from maintainers
    author: str
    created_at: datetime
```

### Output (Pydantic — forces structured LLM output)
```python
class SubTask(BaseModel):
    id: int
    description: str
    type: Literal["search", "write", "test", "docs"]
    depends_on: List[int]             # e.g. [1] means "do task 1 first"
    target_files: List[str]           # estimated file paths from repo map
    acceptance_criteria: str          # how to know this sub-task is done

class Plan(BaseModel):
    issue_summary: str                # one-line summary of what to do
    root_cause_hypothesis: Optional[str]  # for bugs: what's likely wrong
    affected_languages: List[str]
    sub_tasks: List[SubTask]
    files_likely_involved: List[str]
    complexity: Literal["trivial", "low", "medium", "high"]
    estimated_retries_budget: int     # trivial=1, low=1, medium=2, high=3
    test_strategy: str                # "add new test" or "existing tests cover it"
```

### System Prompt Strategy
```text
You are a senior software engineer analyzing a GitHub issue.
You have access to the repository structure below.

REPO MAP:
{repo_map_json}

ISSUE:
Title: {issue.title}
Body: {issue.body}
Labels: {issue.labels}

YOUR TASK:
1. Summarize what needs to be done in one sentence.
2. If this is a bug, hypothesize the root cause.
3. Break the work into ordered sub-tasks.
4. For each sub-task, specify:
   - What type of work it is (search/write/test/docs)
   - Which files are likely involved
   - Clear acceptance criteria
5. Estimate complexity (trivial/low/medium/high).

RULES:
- Always include a "test" sub-task — either write new tests or verify existing ones pass.
- Order tasks by dependency — search before write, write before test.
- Be specific about file paths using the repo map above.

Return your plan as JSON matching the Plan schema.
```

### Decision Logic
```
Issue has label "bug"?
  → Include root_cause_hypothesis
  → First sub-task is always "search" to find the buggy code
  → test_strategy = "add regression test"

Issue has label "enhancement" or "feature"?
  → First sub-task is "search" to find where to add new code
  → Include a "docs" sub-task if README is involved
  → test_strategy = "add new test for new feature"

Issue body mentions specific file names?
  → Put those in files_likely_involved directly
  → Cross-reference with repo map to validate they exist
```

---

## 2. ✍️ Code Writer Agent (`agents/code_writer.py`)

### Purpose
Generate actual code changes (as unified diffs) based on a sub-task + retrieved context.

### Architecture: Tool-Use Agent
The Code Writer is NOT a simple "one-shot prompt → get code" agent. It has **tools** that it can call mid-generation to gather more information:

```python
TOOLS = [
    {
        "name": "read_file",
        "description": "Read the full content of a file in the repository",
        "parameters": {
            "file_path": {"type": "string", "description": "Relative path from repo root"}
        }
    },
    {
        "name": "search_codebase",
        "description": "Search the codebase for relevant code using natural language",
        "parameters": {
            "query": {"type": "string", "description": "What to search for"}
        }
    },
    {
        "name": "list_directory",
        "description": "List all files in a directory",
        "parameters": {
            "dir_path": {"type": "string", "description": "Relative directory path"}
        }
    },
    {
        "name": "submit_changes",
        "description": "Submit the final code changes when done",
        "parameters": {
            "operations": {
                "type": "array",
                "items": {
                    "action": "create | modify | delete",
                    "file_path": "string",
                    "unified_diff": "string (for modify)",
                    "full_content": "string (for create)"
                }
            },
            "explanation": "string — why these changes solve the issue"
        }
    }
]
```

### Agent Loop (ReAct Pattern)
```
1. LLM receives: sub-task + code context + tools
2. LLM thinks: "I need to see the full users.py file"
3. LLM calls: read_file("backend/routes/users.py")
4. System returns: file content
5. LLM thinks: "I also need to see how products.py handles missing items"
6. LLM calls: read_file("backend/routes/products.py")
7. System returns: file content
8. LLM thinks: "Now I know the pattern. I'll add a None check."
9. LLM calls: submit_changes([{action: "modify", file_path: "backend/routes/users.py", unified_diff: "..."}])
10. Loop ends → return CodeChanges
```

### System Prompt
```text
You are an expert software engineer making changes to a codebase.

TASK:
{sub_task.description}

ACCEPTANCE CRITERIA:
{sub_task.acceptance_criteria}

CODE CONTEXT (from codebase search):
{formatted_code_chunks}

REPO CONVENTIONS (auto-detected):
- Indentation: {indent_style} ({indent_size} spaces)
- Import style: {import_style}
- Naming: {naming_convention}
- Test location: {test_directory}

INSTRUCTIONS:
1. Use the tools to explore the codebase if you need more context.
2. When ready, call submit_changes with your modifications.
3. Output changes as UNIFIED DIFFS for modifications.
4. For new files, provide the full content.
5. Match the repo's existing coding style exactly.
6. Include docstrings/comments matching the repo's documentation style.

RULES:
- Never modify files not related to the task.
- Never add dependencies not already in the project.
- Keep changes minimal and focused.
- If creating a new function, follow the naming pattern of existing functions.
```

### Diff Generation Format
```diff
--- a/backend/routes/users.py
+++ b/backend/routes/users.py
@@ -15,6 +15,8 @@
 @router.get("/users/{user_id}")
 def get_user(user_id: int, db: Session = Depends(get_db)):
     user = db.query(User).filter(User.id == user_id).first()
+    if user is None:
+        raise HTTPException(status_code=404, detail="User not found")
     return user
```

### Convention Auto-Detection
Before the Code Writer runs, we analyze the repo to extract coding conventions:

```python
def detect_conventions(repo_path: str) -> RepoConventions:
    sample_files = get_random_source_files(repo_path, n=5)

    return RepoConventions(
        indent_style=detect_indent(sample_files),       # "spaces" | "tabs"
        indent_size=detect_indent_size(sample_files),    # 2 | 4
        quote_style=detect_quotes(sample_files),         # "single" | "double"
        import_style=detect_import_style(sample_files),  # "absolute" | "relative"
        naming_convention=detect_naming(sample_files),   # "snake_case" | "camelCase"
        has_type_hints=detect_type_hints(sample_files),  # True/False (Python)
        docstring_style=detect_docstring_style(sample_files),  # "google" | "numpy" | "sphinx"
        line_ending=detect_line_ending(sample_files),    # "LF" | "CRLF"
        max_line_length=detect_max_line_length(sample_files),  # 80 | 100 | 120
    )
```

---

## 3. 🧪 Test Runner Agent (`sandbox/test_runner.py`)

### Purpose
Execute tests in a safe subprocess and return structured results.

### Execution Flow (detailed)

```python
async def run_tests(
    repo_path: str,
    language_profile: RepoLanguageProfile,
    code_changes: List[FileOperation],
    timeout: int = 120
) -> TestResults:
    """Complete test execution pipeline."""

    # 1. Create isolated work directory
    work_dir = create_work_dir(repo_path)  # copies repo to temp dir

    # 2. Apply code patches
    for op in code_changes:
        apply_operation(work_dir, op)

    # 3. Install dependencies (if not cached)
    await install_dependencies(work_dir, language_profile)

    # 4. Detect and run test command
    test_cmd = get_test_command(language_profile)
    result = await execute_subprocess(test_cmd, work_dir, timeout)

    # 5. Parse output into structured results
    test_results = parse_test_output(
        result.stdout,
        result.stderr,
        result.exit_code,
        language_profile.test_framework
    )

    # 6. Cleanup
    cleanup_work_dir(work_dir)

    return test_results
```

### Test Output Parsers

**Pytest parser** (`result_parser.py`):
```python
def parse_pytest_output(stdout: str, stderr: str, exit_code: int) -> TestResults:
    """
    Parse pytest output like:
    ===== 18 passed, 2 failed in 3.45s =====

    FAILED tests/test_users.py::test_get_missing_user - AssertionError: ...
    """
    # Extract summary line
    summary_match = re.search(
        r"(\d+) passed(?:, (\d+) failed)?(?:, (\d+) error)?",
        stdout
    )

    # Extract individual failures
    failures = []
    failure_blocks = re.findall(
        r"FAILED ([\w/\.]+)::(\w+) - (\w+): (.+)",
        stdout + stderr
    )
    for file_path, test_name, error_type, message in failure_blocks:
        failures.append(TestFailure(
            test_name=test_name,
            file_path=file_path,
            error_type=error_type,
            error_message=message.strip(),
            line_number=extract_line_number(stdout, test_name)
        ))

    return TestResults(
        command_run="pytest --tb=short -q",
        exit_code=exit_code,
        passed=int(summary_match.group(1)) if summary_match else 0,
        failed=int(summary_match.group(2) or 0) if summary_match else 0,
        total=...,
        failures=failures,
        stdout=stdout[-2000:],   # last 2000 chars (avoid huge outputs)
        stderr=stderr[-1000:]
    )
```

**Jest parser** (similar pattern for JavaScript)
**go test parser** (similar pattern for Go)

### Safety controls
```python
async def execute_subprocess(cmd: List[str], cwd: str, timeout: int) -> SubprocessResult:
    env = os.environ.copy()
    env["HTTP_PROXY"] = ""          # block outbound network
    env["HTTPS_PROXY"] = ""
    env["NO_PROXY"] = "*"

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env
    )

    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        proc.kill()
        return SubprocessResult(exit_code=-1, stdout="", stderr="TIMEOUT after {timeout}s")

    return SubprocessResult(
        exit_code=proc.returncode,
        stdout=stdout.decode(),
        stderr=stderr.decode()
    )
```

---

## 4. 🔍 Reviewer Agent (`agents/reviewer.py`)

### Purpose
Quality gate that decides: approve the code for PR, send it back for revision, or reject entirely.

### Scoring Checklist

```python
class ReviewCriterion(BaseModel):
    name: str
    score: int              # 1-5
    weight: float           # percentage weight
    feedback: str           # specific feedback for this criterion

class ReviewDecision(BaseModel):
    criteria: List[ReviewCriterion]
    weighted_score: float   # calculated: sum(score * weight) / sum(weights), scaled to 10
    verdict: Literal["approve", "revise", "reject"]
    overall_feedback: str
    retry_suggestion: Optional[str]   # specific instruction for Code Writer on retry
    security_flags: List[str]         # any dangerous patterns detected
```

### Scoring Criteria with Weights

| Criterion | Weight | What LLM Checks |
|-----------|--------|-----------------|
| `tests_pass` | 30% | Did all tests pass? Any new failures? |
| `requirements_met` | 25% | Does the code actually address the issue? |
| `no_regressions` | 20% | Did any previously passing tests break? |
| `style_consistency` | 15% | Does it match repo conventions? |
| `security` | 10% | Any `eval()`, `exec()`, hardcoded secrets, SQL injection? |

### Verdict Logic
```python
def decide_verdict(
    weighted_score: float,
    test_results: TestResults,
    retry_count: int,
    max_retries: int
) -> str:
    # Hard reject: security issues found
    if security_flags:
        return "reject"

    # Hard reject: more tests fail now than before
    if test_results.new_failures > 0:
        if retry_count >= max_retries:
            return "reject"
        return "revise"

    # Score-based
    if weighted_score >= 7.0:
        return "approve"
    elif retry_count < max_retries:
        return "revise"
    else:
        return "reject"  # exhausted retries
```

### System Prompt
```text
You are a senior code reviewer evaluating changes made by an AI agent.

ORIGINAL ISSUE:
{issue_summary}

CODE CHANGES:
{unified_diffs}

TEST RESULTS:
- Total: {test_results.total}
- Passed: {test_results.passed}
- Failed: {test_results.failed}
- Failures: {formatted_failures}

PREVIOUS FEEDBACK (retry #{retry_count}):
{feedback_history}

EVALUATE the changes on these criteria (score each 1-5):
1. tests_pass (30%): Do all tests pass? Were relevant tests added?
2. requirements_met (25%): Does the code fully address the issue?
3. no_regressions (20%): Did existing tests still pass?
4. style_consistency (15%): Does code match repo conventions?
5. security (10%): Any dangerous patterns (eval, exec, hardcoded secrets)?

ALSO CHECK for these security red flags:
- os.system(), subprocess.call() with shell=True
- eval(), exec()
- Hardcoded API keys, passwords, tokens
- SQL string concatenation (SQL injection risk)
- Unrestricted file path operations (path traversal)

If verdict is "revise", provide a SPECIFIC retry_suggestion telling
the Code Writer exactly what to fix. Don't be vague.

Return JSON matching the ReviewDecision schema.
```

### Feedback Accumulation (Retry Loop)
```python
# In LangGraph state, feedback_history grows each retry:
feedback_history = [
    # Retry 1:
    "The middleware is only applied to /api/v1 routes. Apply it to ALL route groups.",
    # Retry 2:
    "Middleware now covers all routes, but the RateLimiter class is missing a docstring. Also, the test_rate_limit_exceeded test asserts status 429 but the middleware returns 503."
]

# The Code Writer sees ALL previous feedback, preventing repeated mistakes
```

---

## 5. 📤 PR Agent (`agents/pr_agent.py`)

### Purpose
Create a polished, professional GitHub Pull Request.

### Step-by-Step Flow

```python
async def create_pull_request(
    repo_url: str,
    repo_path: str,
    issue: GitHubIssue,
    code_changes: List[FileOperation],
    test_results: TestResults,
    plan: Plan,
    total_cost: float
) -> str:
    """Create branch, commit, push, and open PR. Returns PR URL."""

    # 1. Create branch
    branch_name = f"agent/issue-{issue.number}-{slugify(issue.title)[:30]}"
    create_branch(repo_path, branch_name)

    # 2. Apply changes and commit
    for i, task in enumerate(plan.sub_tasks):
        task_changes = get_changes_for_task(code_changes, task)
        if task_changes:
            apply_changes(repo_path, task_changes)
            commit(
                repo_path,
                message=f"feat(#{issue.number}): {task.description}\n\nSub-task {task.id}/{len(plan.sub_tasks)}"
            )

    # 3. Push to remote
    push(repo_path, branch_name)

    # 4. Generate PR description via LLM
    pr_body = await generate_pr_body(issue, code_changes, test_results, plan, total_cost)

    # 5. Open PR via GitHub API
    pr = github.create_pull(
        title=f"fix: {issue.title} [agent]",
        body=pr_body,
        head=branch_name,
        base="main"
    )

    # 6. Add labels
    pr.add_labels(["automated", "ai-generated"])

    return pr.html_url
```

### PR Body Template
```markdown
## Summary
{ai_generated_summary}

## Issue
Closes #{issue_number}

## Changes Made
| File | Action | Description |
|------|--------|-------------|
| `backend/routes/users.py` | Modified | Added None check, return 404 |
| `backend/tests/test_users.py` | Modified | Added test for missing user |

## Test Results
✅ **{passed}/{total}** tests passed

{failure_details_if_any}

## Agent Metrics
| Metric | Value |
|--------|-------|
| Sub-tasks | {n_subtasks} |
| Retries | {retry_count}/{max_retries} |
| LLM Cost | ${total_cost:.2f} |
| Duration | {duration}s |
| Model | {primary_model} |

---
🤖 *Generated by [Autonomous SWE Agent](repo_link)*
```

### Commit Strategy
| Strategy | When |
|----------|------|
| One commit per sub-task | Default — shows clean history |
| Single squash commit | If only 1 sub-task in the plan |
| Fixup commits on retry | If code was revised, amend the original commit |

### Error Handling
```python
# If PR creation fails:
try:
    pr = github.create_pull(...)
except GithubException as e:
    if "A pull request already exists" in str(e):
        # Find existing PR and update it
        existing_pr = find_existing_pr(branch_name)
        existing_pr.edit(body=pr_body)
        return existing_pr.html_url
    elif "Repository was archived" in str(e):
        return "ERROR: Repository is archived, cannot create PR"
    else:
        raise
```
