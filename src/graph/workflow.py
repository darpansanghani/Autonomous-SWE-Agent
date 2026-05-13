from langgraph.graph import StateGraph, END

from src.graph.state import AgentState
from src.graph.nodes import (
    parse_issue_node,
    clone_repo_node,
    index_codebase_node,
    planner_node,
    search_code_node,
    code_writer_node,
    apply_patch_node,
    test_runner_node,
    reviewer_node,
    pr_agent_node,
    failure_report_node
)
from src.graph.edges import route_after_review

def build_agent_graph() -> StateGraph:
    """Builds and wires the entire LangGraph state machine."""
    
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
        route_after_review,
        {
            "approve":  "create_pr",
            "search_code": "search_code", # if more tasks
            "revise":   "search_code",    # retry same task
            "reject":   "report_failure",
        }
    )

    # ── Terminal Edges ──
    graph.add_edge("create_pr",       END)
    graph.add_edge("report_failure",  END)

    return graph.compile()
