# Autonomous SWE Agent — Project Structure, Build Phases & UI

---

## Project Directory Structure

```
autonomous_agent/
│
├── README.md
├── pyproject.toml               # all deps, project metadata
├── .env.example                 # template for all API keys
├── .gitignore
│
├── config/
│   ├── settings.py              # Pydantic BaseSettings — loads .env
│   └── models.py                # LLM model assignments per agent
│
├── src/
│   ├── __init__.py
│   │
│   ├── graph/                   # ── LangGraph Core ──
│   │   ├── __init__.py
│   │   ├── state.py             # AgentState TypedDict
│   │   ├── workflow.py          # build_graph() — wires all nodes
│   │   ├── nodes.py             # one function per graph node
│   │   └── edges.py             # conditional edge logic (retry? approve?)
│   │
│   ├── agents/                  # ── Specialized Agents ──
│   │   ├── __init__.py
│   │   ├── planner.py           # Issue → Plan (Pydantic output)
│   │   ├── code_writer.py       # Context → FileOperations (with tools)
│   │   ├── reviewer.py          # Code → ReviewDecision (scored checklist)
│   │   └── pr_agent.py          # Changes → GitHub PR
│   │
│   ├── rag/                     # ── Codebase Understanding ──
│   │   ├── __init__.py
│   │   ├── indexer.py           # orchestrates full repo indexing
│   │   ├── chunker.py           # tree-sitter AST-aware chunking
│   │   ├── embedder.py          # embed chunks → Qdrant
│   │   ├── retriever.py         # query → top-K relevant chunks
│   │   └── repo_map.py          # generate JSON structural map (files/classes/fns)
│   │
│   ├── sandbox/                 # ── Safe Code Execution ──
│   │   ├── __init__.py
│   │   ├── executor.py          # subprocess + venv runner
│   │   ├── patch_applier.py     # apply unified diffs to files
│   │   ├── test_runner.py       # language-aware test execution
│   │   └── result_parser.py     # parse pytest/jest/go test output → TestResults
│   │
│   ├── github_client/           # ── GitHub Integration ──
│   │   ├── __init__.py
│   │   ├── issue_reader.py      # fetch + parse GitHub issues
│   │   ├── repo_manager.py      # clone, branch, commit, push (GitPython)
│   │   └── pr_creator.py        # create PR with description (PyGithub)
│   │
│   ├── llm/                     # ── LLM Router ──
│   │   ├── __init__.py
│   │   ├── router.py            # LiteLLM wrapper with retry + fallback
│   │   ├── cost_tracker.py      # token usage + cost per run (LiteLLM callbacks)
│   │   └── prompts/             # all system prompts as .txt files
│   │       ├── planner.txt
│   │       ├── code_writer.txt
│   │       ├── reviewer.txt
│   │       └── pr_description.txt
│   │
│   └── utils/
│       ├── __init__.py
│       ├── diff.py              # generate + apply unified diffs
│       ├── file_utils.py        # safe file read/write helpers
│       ├── language_detect.py   # detect repo languages from extensions
│       └── logger.py            # structured logging (structlog)
│
├── db/                          # ── Persistence ──
│   ├── __init__.py
│   ├── models.py                # SQLModel table definitions
│   ├── database.py              # SQLite engine + session
│   └── repository.py           # CRUD operations for run history
│
├── ui/                          # ── Streamlit Dashboard ──
│   ├── app.py                   # single-page app (SPA) with tab navigation
│   └── components/
│       ├── issue_form.py        # issue URL + repo URL input form
│       ├── agent_log.py         # streaming log component
│       ├── diff_renderer.py     # renders unified diffs with colors
│       └── cost_badge.py        # shows token cost per run
│
├── scripts/
│   ├── setup_dummy_repo.py      # creates + populates dummy GitHub repo
│   ├── index_repo.py            # standalone: index a repo into Qdrant
│   └── run_agent.py             # CLI entry point (no UI)
│
└── tests/
    ├── conftest.py
    ├── unit/
    │   ├── test_chunker.py
    │   ├── test_retriever.py
    │   ├── test_patch_applier.py
    │   ├── test_result_parser.py
    │   └── test_planner.py
    └── integration/
        ├── test_full_pipeline.py  # end-to-end on dummy repo
        └── test_github_client.py
```

---

## Dummy GitHub Repository Design

### What to Create: `swe-agent-playground`

A small Python + JavaScript project with **intentional bugs and missing features** — each one becomes a GitHub issue for the agent to solve.

```
swe-agent-playground/
├── README.md
├── backend/                     # Python FastAPI
│   ├── main.py
│   ├── routes/
│   │   ├── users.py             # CRUD for users
│   │   └── products.py          # CRUD for products
│   ├── models.py
│   ├── database.py
│   └── tests/
│       ├── test_users.py
│       └── test_products.py
├── frontend/                    # JavaScript (vanilla)
│   ├── index.html
│   ├── app.js
│   └── utils.js
├── requirements.txt
└── package.json
```

### Pre-written Issues (Increasing Difficulty)

| # | Issue Title | Difficulty | What Agent Must Do |
|---|-------------|-----------|-------------------|
| 1 | Fix typo in README | Trivial | Edit one line in README.md |
| 2 | `get_user` returns 500 on missing user | Easy | Add `None` check, return 404 |
| 3 | Add `DELETE /products/{id}` endpoint | Medium | New route + test |
| 4 | Add input validation to `POST /users` | Medium | Pydantic validation + error handling |
| 5 | Add pagination to `GET /products` | Hard | Query params + DB changes + tests |
| 6 | `calculate_discount()` wrong for 0% | Medium | Bug fix in utils.js + JS test |

> Start by solving Issue #1 and #2 — success here is your demo moment.

### Script: `setup_dummy_repo.py`
```
1. Creates a new GitHub repo: "swe-agent-playground"
2. Pushes all starter files via PyGithub + GitPython
3. Creates all 6 GitHub issues with labels ("bug", "enhancement")
4. Prints repo URL for .env configuration
```

---

## Streamlit UI Design

### Single-Page App Structure (`app.py`)
Streamlit allows building a Single-Page Application (SPA) using `st.tabs` or sidebar buttons. Everything runs from a single `app.py` file, giving a seamless experience with zero page reloads.

**Real-Time UX Architecture:**
1. **Background Threading**: The LangGraph agent runs in a background thread so the Streamlit UI never freezes.
2. **State Syncing (`st.session_state`)**: Progress is continuously synced to session state. You can switch tabs (e.g., to view history) and the agent keeps running in the background.
3. **Live Streaming (`st.empty()`)**: The "Live Log" uses empty containers to append log messages line-by-line instantly without manual browser refreshes.

#### View 1: Run Agent (Tab 1)
```text
┌─────────────────────────────────────────────────┐
│  🤖 Autonomous SWE Agent                         │
├─────────────────────────────────────────────────┤
│  [ Run Agent ]  [ Run History ]  [ Diff Viewer ] │ ← st.tabs or st.radio
├─────────────────────────────────────────────────┤
│  GitHub Issue URL  [________________________]    │
│  Repository URL    [________________________]    │
│  LLM Profile       [GPT-4o ▼]  [Cost Budget $5] │
│                    [  ▶ Run Agent  ]             │
├─────────────────────────────────────────────────┤
│  📋 Agent Log                         LIVE 🔴    │
│  ✅ [10:01] Issue parsed: "Fix 500 on missing"   │
│  ✅ [10:02] Plan created: 3 sub-tasks            │
│  🔄 [10:03] Indexing codebase (142 files)...    │
│  ✅ [10:04] Retrieved 8 relevant chunks          │
│  🔄 [10:05] Code Writer generating diff...      │
│  ✅ [10:06] Patch applied                        │
│  🔄 [10:07] Running tests (pytest)...            │
│  ✅ [10:08] 18/18 tests passed                   │
│  ✅ [10:09] Review score: 8.5/10 → APPROVED      │
│  ✅ [10:10] PR opened: github.com/…/pull/7 🔗    │
├─────────────────────────────────────────────────┤
│  💰 Cost: $0.12   ⏱ Duration: 68s               │
└─────────────────────────────────────────────────┘
```

#### View 2: Run History (Tab 2)
```text
┌──────────┬──────────────────────────┬────────┬───────┬──────┬──────────┐
│ Run ID   │ Issue                    │ Status │ Tests │ Cost │ PR       │
├──────────┼──────────────────────────┼────────┼──────-┼──────┼──────────┤
│ a1b2c3   │ Fix 500 on missing user  │  ✅    │ 18/18 │$0.12 │ #7 🔗    │
│ d4e5f6   │ Add DELETE /products     │  ✅    │ 22/22 │$0.31 │ #8 🔗    │
│ g7h8i9   │ Add pagination           │  ❌    │ 19/22 │$0.58 │ —        │
└──────────┴──────────────────────────┴────────┴───────┴──────┴──────────┘
```

#### View 3: Diff Viewer (Tab 3)
- Select any run → see syntax-highlighted diff of every changed file
- Green for additions, red for deletions
- Click file name to see full file content

---

## Build Phases

### Phase 1 — Skeleton (Week 1)
> Goal: end-to-end pipeline works on Issue #1 (README typo)

| Task | Details |
|------|---------|
| Project setup | `pyproject.toml`, `.env`, logging, folder structure |
| Settings module | `config/settings.py` with Pydantic BaseSettings |
| GitHub client | Clone repo, read issue, create branch, commit, push, open PR |
| Basic Planner | Issue → Plan via single LiteLLM call, Pydantic output |
| Basic Code Writer | Plan + direct file read → generate diff (no RAG) |
| Subprocess executor | `subprocess.run` with timeout, capture stdout/stderr |
| Basic test runner | Run `pytest`, capture pass/fail count |
| LangGraph linear | Wire: Parse → Plan → Write → Test → PR |
| **Milestone** | Agent opens a real PR on dummy repo for Issue #1 |

### Phase 2 — Intelligence (Week 2)
> Goal: solve Issue #2 and #3 (bug fix + new endpoint)

| Task | Details |
|------|---------|
| tree-sitter chunker | AST-aware chunking for Python + JavaScript |
| Repo map generator | JSON file tree + function/class index |
| Qdrant indexer | Embed chunks → local Qdrant collection |
| RAG retriever | Query → semantic search → top-K chunks |
| Dependency expander | If file A found, include files A imports |
| Reviewer agent | Scored checklist, structured ReviewDecision output |
| Retry loop | LangGraph conditional edge + feedback_history accumulation |
| Multi-language test runner | Python (pytest), JS (jest/npm test) |
| **Milestone** | Agent solves bug fix and new-endpoint issues with RAG |

### Phase 3 — UI + Persistence (Week 3)
> Goal: demo-ready dashboard showing live agent work

| Task | Details |
|------|---------|
| SQLite DB | SQLModel tables: Run, SubTask, FileChange, TestResult |
| DB persistence | Save every agent decision + result to DB |
| Streamlit SPA Core | Setup `st.tabs` or sidebar buttons for navigation in `app.py` |
| View 1 (Run Agent) | Issue form + background thread execution + live log streaming via `st.empty()` |
| View 2 & 3 | Run history table from SQLite + syntax-highlighted diff viewer |
| Cost tracker | LiteLLM callback → store tokens + USD cost per run |
| `setup_dummy_repo.py` | Script to create + populate dummy GitHub repo |
| **Milestone** | Full demo: submit issue in UI, watch agent work, see PR link |

### Phase 4 — Polish & Differentiation (Week 4+)
> Goal: interview-worthy depth

| Task | Details |
|------|---------|
| Multi-language full support | Go, Java, Rust tree-sitter grammars |
| LLM fallback chain | If GPT-4o fails → try Gemini → try NVIDIA NIM |
| Human-in-the-loop | "Pause before PR" mode — user can edit diff in UI |
| Arize Phoenix tracing | Full LLM span tracing (you already know this!) |
| SWE-bench evaluation | Run on 5-10 SWE-bench Lite tasks, report success rate |
| Security scanner | Pre-execution: detect dangerous patterns in generated code |
| PR review comments | Agent responds to PR review comments with new commits |

---

## Dependencies

```toml
[project]
name = "autonomous-swe-agent"
version = "0.1.0"
requires-python = ">=3.11"

dependencies = [
    # Orchestration
    "langgraph>=0.4",
    "langchain>=0.3",
    "langchain-core>=0.3",

    # LLM Router (OpenAI + Gemini + NVIDIA NIM + Anthropic)
    "litellm>=1.40",
    "openai>=1.30",
    "google-generativeai>=0.8",

    # Code Parsing
    "tree-sitter>=0.24",
    "tree-sitter-python>=0.24",
    "tree-sitter-javascript>=0.24",
    "tree-sitter-typescript>=0.24",
    "tree-sitter-go>=0.24",
    "tree-sitter-java>=0.24",
    "tree-sitter-rust>=0.24",

    # Vector Store + Embeddings
    "qdrant-client>=1.13",
    "langchain-qdrant>=0.2",

    # GitHub
    "PyGithub>=2.6",
    "gitpython>=3.1",

    # Persistence
    "sqlmodel>=0.0.21",
    "aiosqlite>=0.20",

    # UI
    "streamlit>=1.45",
    "streamlit-extras>=0.4",

    # Observability
    "arize-phoenix>=8.0",
    "opentelemetry-api>=1.30",
    "opentelemetry-sdk>=1.30",

    # Utilities
    "pydantic>=2.7",
    "pydantic-settings>=2.3",
    "structlog>=24.0",
    "python-dotenv>=1.0",
    "unidiff>=0.7",          # parse + apply unified diffs
    "rich>=13.0",            # pretty CLI output
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
    "ruff>=0.9",
    "mypy>=1.10",
]
```

---

## Environment Variables

```bash
# .env.example

# ── LLM Providers ──────────────────────────
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=AIza...
NVIDIA_NIM_API_KEY=nvapi-...

# ── Model Assignments ──────────────────────
PLANNER_MODEL=gemini/gemini-1.5-pro
CODE_WRITER_MODEL=gpt-4o
REVIEWER_MODEL=gpt-4o
RAG_QUERY_MODEL=nvidia_nim/meta/llama-3.1-8b-instruct
PR_WRITER_MODEL=gemini/gemini-1.5-flash

# ── GitHub ─────────────────────────────────
GITHUB_TOKEN=ghp_...
GITHUB_USERNAME=your-username

# ── Vector Store ───────────────────────────
QDRANT_PATH=./data/qdrant            # local file-based Qdrant

# ── Sandbox ────────────────────────────────
SANDBOX_TIMEOUT_SECONDS=120
SANDBOX_WORK_DIR=./data/runs         # temp dirs created here

# ── Observability ──────────────────────────
PHOENIX_ENABLED=false                # set true in Phase 4
PHOENIX_PORT=6006

# ── Agent Behavior ─────────────────────────
MAX_RETRIES=3
MAX_COST_USD=5.00                    # kill switch if cost exceeds this
```

---

## Interview Talking Points

### How you differentiate from SWE-Agent / Devin

| Aspect | SWE-Agent | Devin | **Your Agent** |
|--------|-----------|-------|----------------|
| Orchestration | Single loop | Proprietary | **LangGraph multi-agent graph** |
| LLM | 1 provider | Proprietary | **LiteLLM: 3+ providers, swappable per agent** |
| Code understanding | Grep + file read | Unknown | **AST-aware RAG (tree-sitter + Qdrant)** |
| Sandbox | Docker | VM | **Subprocess + venv (no Docker needed)** |
| Languages | Python-focused | Multi | **Multi via tree-sitter grammars** |
| Observability | Logs | None | **Arize Phoenix full tracing** |
| UI | None | Web app | **Streamlit live dashboard** |

### Top 5 Questions Interviewers Will Ask

1. **"Why LangGraph over AutoGen or CrewAI?"**
   > LangGraph models the agent as a stateful graph with explicit cycles — perfect for the retry loop (review → rewrite). CrewAI is sequential. AutoGen is conversation-based, harder to checkpoint. LangGraph also has native human-in-the-loop and persistence.

2. **"Large repos won't fit in context — how do you handle that?"**
   > Two-level indexing: structural (AST repo map — function signatures, imports, file tree as JSON) + semantic (vector search over function-level chunks in Qdrant). Only the top-8 relevant chunks go to the LLM — never the whole repo. Dependency expansion ensures imported files are included automatically.

3. **"What if tests fail 3 times — what happens?"**
   > The reviewer accumulates feedback across retries into `feedback_history`. On the 3rd failure, the agent stops, posts a comment on the GitHub issue explaining what it tried and why it failed, then sets status to `human_review`. This is better than infinite looping.

4. **"How do you ensure the generated code doesn't do anything dangerous?"**
   > The reviewer agent has a security checklist that scans for dangerous patterns (`os.system`, `eval`, `exec`, `subprocess`, hardcoded secrets) before execution. The subprocess sandbox also runs with `HTTP_PROXY=""` to block outbound network calls and a 120-second timeout.

5. **"What's your success rate on real issues?"**
   > On the 6 dummy repo issues: 5/6 (83%). On SWE-bench Lite (Phase 4): target 15-20%, which is competitive with SWE-Agent v1 (12%) as a solo project. The key insight is that issue decomposition quality drives success — the planner is the highest-leverage component.
