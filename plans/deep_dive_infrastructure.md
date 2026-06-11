# Deep Dive: Infrastructure — LangGraph, LLM Router, GitHub, DB & UI

---

## 1. ⚙️ LangGraph Workflow (`graph/workflow.py`)

### Full Graph Definition

```python
from langgraph.graph import StateGraph, END

def build_agent_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # ── Register Nodes ──
    graph.add_node("parse_issue",     parse_issue_node)
    graph.add_node("clone_repo",      clone_repo_node)
    graph.add_node("index_codebase",  index_codebase_node)
    graph.add_node("plan",            planner_node)
    graph.add_node("search_code",     search_code_node)
    graph.add_node("write_code",      code_writer_node)
    graph.add_node("apply_patch",     apply_patch_node)
    graph.add_node("run_tests",       test_runner_node)
    graph.add_node("review",          reviewer_node)
    graph.add_node("create_pr",       pr_agent_node)
    graph.add_node("report_failure",  failure_report_node)

    # ── Linear Flow ──
    graph.set_entry_point("parse_issue")
    graph.add_edge("parse_issue",    "clone_repo")
    graph.add_edge("clone_repo",     "index_codebase")
    graph.add_edge("index_codebase", "plan")
    graph.add_edge("plan",           "search_code")
    graph.add_edge("search_code",    "write_code")
    graph.add_edge("write_code",     "apply_patch")
    graph.add_edge("apply_patch",    "run_tests")
    graph.add_edge("run_tests",      "review")

    # ── Conditional Branch (the retry loop) ──
    graph.add_conditional_edges(
        "review",
        route_after_review,    # function that decides next node
        {
            "approve":  "create_pr",
            "revise":   "search_code",    # back to search → write → test → review
            "reject":   "report_failure",
        }
    )

    # ── Terminal Edges ──
    graph.add_edge("create_pr",       END)
    graph.add_edge("report_failure",  END)

    return graph.compile()
```

### Conditional Edge Logic (`edges.py`)

```python
def route_after_review(state: AgentState) -> str:
    review = state["review"]
    retry_count = state["retry_count"]
    max_retries = state["plan"].estimated_retries_budget

    if review.verdict == "approve":
        return "approve"
    elif review.verdict == "revise" and retry_count < max_retries:
        return "revise"
    else:
        return "reject"
```

### Node Implementations (`nodes.py`)

Each node is a function: `(AgentState) -> dict` returning partial state updates.

```python
async def parse_issue_node(state: AgentState) -> dict:
    """Fetch and parse the GitHub issue."""
    issue = await github_client.fetch_issue(state["repo_url"], state["issue_number"])
    log_event("issue_parsed", {"title": issue.title})
    return {"issue": issue, "status": "running"}

async def clone_repo_node(state: AgentState) -> dict:
    """Clone the repository to a local temp directory."""
    local_path = await github_client.clone_repo(state["repo_url"])
    log_event("repo_cloned", {"path": local_path})
    return {"repo_local_path": local_path}

async def index_codebase_node(state: AgentState) -> dict:
    """Index the codebase: detect languages, parse AST, embed, store."""
    index_result = await indexer.index_repository(
        state["repo_local_path"],
        extract_repo_name(state["repo_url"])
    )
    log_event("codebase_indexed", {"chunks": index_result.total_chunks})
    return {
        "repo_map": index_result.repo_map,
        "target_languages": list(index_result.profile.languages.keys()),
        "qdrant_collection": index_result.collection_name,
        "language_profile": index_result.profile
    }

async def planner_node(state: AgentState) -> dict:
    """Generate execution plan from issue + repo map."""
    plan = await planner.create_plan(state["issue"], state["repo_map"])
    log_event("plan_created", {"tasks": len(plan.sub_tasks), "complexity": plan.complexity})
    return {
        "plan": plan,
        "current_task_index": 0,
        "retry_count": 0,
        "feedback_history": []
    }

async def search_code_node(state: AgentState) -> dict:
    """Retrieve relevant code chunks for current sub-task."""
    current_task = state["plan"].sub_tasks[state["current_task_index"]]
    chunks = await retriever.retrieve(
        query=current_task.description,
        collection=state["qdrant_collection"],
        repo_map=state["repo_map"]
    )
    log_event("code_retrieved", {"chunks": len(chunks)})
    return {"code_context": chunks}

async def code_writer_node(state: AgentState) -> dict:
    """Generate code changes using tool-use agent."""
    current_task = state["plan"].sub_tasks[state["current_task_index"]]
    changes = await code_writer.generate_changes(
        sub_task=current_task,
        code_context=state["code_context"],
        repo_path=state["repo_local_path"],
        feedback_history=state["feedback_history"]
    )
    log_event("code_generated", {"files_changed": len(changes.operations)})
    return {"code_changes": changes.operations}

async def apply_patch_node(state: AgentState) -> dict:
    """Apply generated diffs to the repo."""
    success = patch_applier.apply_all(state["repo_local_path"], state["code_changes"])
    log_event("patch_applied", {"success": success})
    return {"applied_patch": success}

async def test_runner_node(state: AgentState) -> dict:
    """Run tests in subprocess sandbox."""
    results = await test_runner.run_tests(
        repo_path=state["repo_local_path"],
        language_profile=state["language_profile"],
        code_changes=state["code_changes"]
    )
    log_event("tests_run", {"passed": results.passed, "failed": results.failed})
    return {"test_results": results}

async def reviewer_node(state: AgentState) -> dict:
    """Review code quality and decide verdict."""
    review = await reviewer.review_changes(
        issue=state["issue"],
        code_changes=state["code_changes"],
        test_results=state["test_results"],
        feedback_history=state["feedback_history"],
        retry_count=state["retry_count"]
    )
    log_event("review_done", {"score": review.weighted_score, "verdict": review.verdict})

    updated = {"review": review}
    if review.verdict == "revise":
        updated["retry_count"] = state["retry_count"] + 1
        updated["feedback_history"] = state["feedback_history"] + [review.retry_suggestion]
        # Revert patch before retry
        patch_applier.revert_all(state["repo_local_path"], state["code_changes"])
    return updated

async def pr_agent_node(state: AgentState) -> dict:
    """Create branch, commit, push, open PR."""
    pr_url = await pr_agent.create_pull_request(
        repo_url=state["repo_url"],
        repo_path=state["repo_local_path"],
        issue=state["issue"],
        code_changes=state["code_changes"],
        test_results=state["test_results"],
        plan=state["plan"],
        total_cost=state["total_cost_usd"]
    )
    log_event("pr_created", {"url": pr_url})
    return {"pr_url": pr_url, "status": "success"}

async def failure_report_node(state: AgentState) -> dict:
    """Post a comment on the issue explaining failure."""
    await github_client.post_issue_comment(
        repo_url=state["repo_url"],
        issue_number=state["issue"].number,
        body=format_failure_report(state)
    )
    log_event("failure_reported")
    return {"status": "failed"}
```

### The `log_event` System (bridges backend → UI)

```python
import queue
import threading

# Global log queue — nodes write, UI reads
_log_queue: queue.Queue = queue.Queue()
_log_lock = threading.Lock()

def log_event(event_type: str, data: dict = None):
    """Called by every graph node to emit progress events."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "event": event_type,
        "data": data or {},
        "emoji": EVENT_EMOJIS.get(event_type, "ℹ️")
    }
    _log_queue.put(entry)

    # Also persist to DB for history
    db.save_log_entry(entry)

EVENT_EMOJIS = {
    "issue_parsed": "✅",
    "repo_cloned": "✅",
    "codebase_indexed": "✅",
    "plan_created": "✅",
    "code_retrieved": "✅",
    "code_generated": "✍️",
    "patch_applied": "✅",
    "tests_run": "🧪",
    "review_done": "🔍",
    "pr_created": "🎉",
    "failure_reported": "❌",
}
```

---

## 2. 🔀 LLM Router (`llm/router.py`)

### LiteLLM Configuration

```python
import litellm

litellm.set_verbose = False

# Model aliases — one place to change all assignments
MODEL_CONFIG = {
    "planner": {
        "primary": settings.PLANNER_MODEL,      # "gemini/gemini-1.5-pro"
        "fallback": "gpt-4o",
        "temperature": 0.2,
        "max_tokens": 4096,
    },
    "code_writer": {
        "primary": settings.CODE_WRITER_MODEL,   # "gpt-4o"
        "fallback": "gemini/gemini-1.5-pro",
        "temperature": 0.1,                      # low = more deterministic code
        "max_tokens": 8192,
    },
    "reviewer": {
        "primary": settings.REVIEWER_MODEL,      # "gpt-4o"
        "fallback": "gemini/gemini-1.5-pro",
        "temperature": 0.1,
        "max_tokens": 4096,
    },
    "rag_reranker": {
        "primary": settings.RAG_QUERY_MODEL,     # "nvidia_nim/meta/llama-3.1-8b-instruct"
        "fallback": "gemini/gemini-1.5-flash",
        "temperature": 0.0,
        "max_tokens": 2048,
    },
    "pr_writer": {
        "primary": settings.PR_WRITER_MODEL,     # "gemini/gemini-1.5-flash"
        "fallback": "gpt-4o-mini",
        "temperature": 0.3,
        "max_tokens": 2048,
    },
}
```

### Completion wrapper with fallback

```python
async def agent_completion(
    agent_name: str,
    messages: List[dict],
    response_format: dict = None,
    tools: List[dict] = None
) -> LLMResponse:
    """Call LLM with automatic fallback + cost tracking."""
    config = MODEL_CONFIG[agent_name]

    for model in [config["primary"], config["fallback"]]:
        try:
            response = await litellm.acompletion(
                model=model,
                messages=messages,
                temperature=config["temperature"],
                max_tokens=config["max_tokens"],
                response_format=response_format,
                tools=tools,
            )
            # Track cost
            cost_tracker.record(
                agent=agent_name,
                model=model,
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                cost_usd=litellm.completion_cost(response)
            )
            return response

        except Exception as e:
            log_event("llm_fallback", {"agent": agent_name, "failed_model": model, "error": str(e)})
            continue

    raise LLMError(f"All models failed for agent {agent_name}")
```

### Cost Tracker (`llm/cost_tracker.py`)

```python
class CostTracker:
    def __init__(self):
        self.entries: List[CostEntry] = []

    def record(self, agent: str, model: str, input_tokens: int, output_tokens: int, cost_usd: float):
        self.entries.append(CostEntry(
            agent=agent, model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            timestamp=datetime.now()
        ))

    @property
    def total_cost(self) -> float:
        return sum(e.cost_usd for e in self.entries)

    def is_over_budget(self, budget: float) -> bool:
        return self.total_cost >= budget

    def summary(self) -> dict:
        """Per-agent cost breakdown."""
        by_agent = defaultdict(float)
        for e in self.entries:
            by_agent[e.agent] += e.cost_usd
        return dict(by_agent)
```

**Kill switch**: Before every LLM call, check `cost_tracker.is_over_budget(settings.MAX_COST_USD)`. If True, abort the run immediately.

---

## 3. 🐙 GitHub Client (`github_client/`)

### `repo_manager.py` — Clone, Branch, Commit, Push

```python
from git import Repo as GitRepo

class RepoManager:
    def __init__(self, github_token: str):
        self.token = github_token

    def clone(self, repo_url: str) -> str:
        """Clone to temp directory. Returns local path."""
        # Inject token into URL for private repos
        auth_url = repo_url.replace(
            "https://github.com/",
            f"https://{self.token}@github.com/"
        )
        local_path = Path(settings.SANDBOX_WORK_DIR) / f"run_{uuid4().hex[:8]}"
        GitRepo.clone_from(auth_url, local_path, depth=1)  # shallow clone
        return str(local_path)

    def create_branch(self, repo_path: str, branch_name: str):
        repo = GitRepo(repo_path)
        repo.git.checkout("-b", branch_name)

    def commit(self, repo_path: str, message: str):
        repo = GitRepo(repo_path)
        repo.git.add("--all")
        repo.git.commit("-m", message, "--author", "SWE Agent <agent@swe-agent.dev>")

    def push(self, repo_path: str, branch_name: str):
        repo = GitRepo(repo_path)
        repo.git.push("origin", branch_name)
```

### `issue_reader.py` — Fetch & Parse Issues

```python
from github import Github

class IssueReader:
    def __init__(self, token: str):
        self.gh = Github(token)

    def fetch_issue(self, repo_url: str, issue_number: int) -> GitHubIssue:
        owner, name = parse_repo_url(repo_url)  # "user/repo" → ("user", "repo")
        repo = self.gh.get_repo(f"{owner}/{name}")
        issue = repo.get_issue(issue_number)

        return GitHubIssue(
            number=issue.number,
            title=issue.title,
            body=issue.body or "",
            labels=[l.name for l in issue.labels],
            comments=[c.body for c in issue.get_comments()],
            author=issue.user.login,
            created_at=issue.created_at
        )

    def parse_issue_from_url(self, issue_url: str) -> tuple:
        """Extract repo_url and issue_number from a full issue URL.
        'https://github.com/user/repo/issues/5' → ('https://github.com/user/repo', 5)
        """
        parts = issue_url.rstrip("/").split("/")
        issue_number = int(parts[-1])
        repo_url = "/".join(parts[:-2])
        return repo_url, issue_number
```

---

## 4. 💾 Database Schema (`db/models.py`)

### SQLModel Tables

```python
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
    status: str                     # "running" | "success" | "failed"
    plan_json: Optional[str]        # serialized Plan
    total_cost_usd: float = 0.0
    retry_count: int = 0
    pr_url: Optional[str]
    duration_seconds: Optional[float]
    primary_model: str              # which LLM was used
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime]

class RunLog(SQLModel, table=True):
    """Individual log entries for a run (feeds the live log UI)."""
    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: str = Field(foreign_key="run.id")
    timestamp: datetime
    event: str                      # "issue_parsed", "tests_run", etc.
    emoji: str
    data_json: Optional[str]        # serialized event data
    message: str                    # human-readable log line

class FileChange(SQLModel, table=True):
    """Files changed in a run (for diff viewer)."""
    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: str = Field(foreign_key="run.id")
    file_path: str
    action: str                     # "create" | "modify" | "delete"
    unified_diff: Optional[str]     # the actual diff
    full_content: Optional[str]     # for created files

class TestResult(SQLModel, table=True):
    """Test execution results."""
    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: str = Field(foreign_key="run.id")
    attempt: int                    # 1, 2, 3 (which retry)
    command_run: str
    exit_code: int
    total: int
    passed: int
    failed: int
    duration_seconds: float
    failures_json: Optional[str]    # serialized List[TestFailure]
    stdout: Optional[str]
    stderr: Optional[str]
```

### DB Init (`db/database.py`)

```python
from sqlmodel import create_engine, SQLModel, Session

DATABASE_URL = f"sqlite:///{settings.DB_PATH}"  # ./data/agent.db
engine = create_engine(DATABASE_URL)

def init_db():
    SQLModel.metadata.create_all(engine)

def get_session() -> Session:
    return Session(engine)
```

---

## 5. 🖥️ Streamlit UI Internals (`ui/app.py`)

### Main App Structure

```python
import streamlit as st
import threading

st.set_page_config(page_title="🤖 Autonomous SWE Agent", layout="wide")

# ── Tab Navigation ─────────────────────────
tab1, tab2, tab3 = st.tabs(["🚀 Run Agent", "📜 Run History", "📝 Diff Viewer"])

with tab1:
    render_run_agent_view()

with tab2:
    render_history_view()

with tab3:
    render_diff_viewer()
```

### Background Thread Execution

```python
def render_run_agent_view():
    issue_url = st.text_input("GitHub Issue URL", placeholder="https://github.com/user/repo/issues/1")
    model_choice = st.selectbox("LLM Profile", ["GPT-4o", "Gemini 1.5 Pro", "Mixed (recommended)"])
    cost_budget = st.slider("Cost Budget ($)", 0.5, 10.0, 5.0)

    if st.button("▶ Run Agent", type="primary"):
        # Initialize session state for this run
        st.session_state["run_active"] = True
        st.session_state["run_logs"] = []
        st.session_state["run_result"] = None

        # Start agent in background thread
        thread = threading.Thread(
            target=run_agent_background,
            args=(issue_url, model_choice, cost_budget),
            daemon=True
        )
        thread.start()

    # ── Live Log Display ─────────────────
    if st.session_state.get("run_active"):
        log_container = st.container()
        status_placeholder = st.empty()

        # Poll for new log entries
        while st.session_state.get("run_active"):
            # Read new entries from the log queue
            new_logs = drain_log_queue()
            st.session_state["run_logs"].extend(new_logs)

            # Re-render all logs
            with log_container:
                for log in st.session_state["run_logs"]:
                    st.markdown(f"{log['emoji']} **[{log['timestamp']}]** {log['message']}")

            # Check if run completed
            if st.session_state.get("run_result"):
                result = st.session_state["run_result"]
                if result["status"] == "success":
                    st.success(f"🎉 PR created: [{result['pr_url']}]({result['pr_url']})")
                else:
                    st.error(f"❌ Agent failed after {result['retries']} retries")
                st.session_state["run_active"] = False
                break

            time.sleep(0.5)  # poll interval

        # Cost badge
        if st.session_state.get("run_result"):
            col1, col2 = st.columns(2)
            col1.metric("💰 Total Cost", f"${result['cost']:.2f}")
            col2.metric("⏱ Duration", f"{result['duration']:.0f}s")
```

### Background Runner Function

```python
def run_agent_background(issue_url: str, model_choice: str, cost_budget: float):
    """Runs in a daemon thread — executes the LangGraph pipeline."""
    try:
        graph = build_agent_graph()
        repo_url, issue_number = parse_issue_url(issue_url)

        initial_state = {
            "repo_url": repo_url,
            "issue_number": issue_number,
            "status": "running",
            "retry_count": 0,
            "feedback_history": [],
            "total_cost_usd": 0.0,
            "run_id": uuid4().hex[:8],
        }

        # Execute graph — all log_event() calls will push to the queue
        final_state = graph.invoke(initial_state)

        st.session_state["run_result"] = {
            "status": final_state["status"],
            "pr_url": final_state.get("pr_url"),
            "cost": final_state["total_cost_usd"],
            "retries": final_state["retry_count"],
            "duration": (datetime.now() - start_time).total_seconds()
        }
    except Exception as e:
        st.session_state["run_result"] = {"status": "failed", "error": str(e)}
```

### History View (reads from SQLite)

```python
def render_history_view():
    runs = db.get_all_runs()  # SELECT * FROM run ORDER BY created_at DESC

    if not runs:
        st.info("No runs yet. Go to 'Run Agent' tab to start one!")
        return

    # Build dataframe for display
    df = pd.DataFrame([{
        "Run ID": r.id,
        "Issue": f"#{r.issue_number}: {r.issue_title}",
        "Status": "✅" if r.status == "success" else "❌",
        "Cost": f"${r.total_cost_usd:.2f}",
        "PR": r.pr_url or "—",
        "Duration": f"{r.duration_seconds:.0f}s" if r.duration_seconds else "—",
    } for r in runs])

    st.dataframe(df, use_container_width=True)

    # Click a row to see details
    selected_run_id = st.selectbox("View details for run:", [r.id for r in runs])
    if selected_run_id:
        render_run_details(selected_run_id)
```

### Diff Viewer (syntax-highlighted)

```python
def render_diff_viewer():
    runs = db.get_all_runs()
    selected_run = st.selectbox("Select Run", [f"{r.id} — {r.issue_title}" for r in runs])

    if selected_run:
        run_id = selected_run.split(" — ")[0]
        file_changes = db.get_file_changes(run_id)

        for change in file_changes:
            with st.expander(f"{'📝' if change.action == 'modify' else '🆕'} {change.file_path}"):
                if change.unified_diff:
                    # Render colored diff
                    st.code(change.unified_diff, language="diff")
                elif change.full_content:
                    # New file — show full content
                    lang = detect_language_from_extension(change.file_path)
                    st.code(change.full_content, language=lang)
```
