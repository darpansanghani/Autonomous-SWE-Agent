from src.graph.state import AgentState
from src.github_client.issue_reader import IssueReader
from src.github_client.repo_manager import RepoManager
from src.rag.indexer import index_repository
from src.agents import planner, code_writer, reviewer, pr_agent
from src.rag import retriever
from src.sandbox import patch_applier, test_runner
from src.utils.logger import log_event

async def parse_issue_node(state: AgentState) -> dict:
    reader = IssueReader()
    repo_url, issue_num = reader.extract_from_issue_url(state["issue_url"]) if "issue_url" in state else (state["repo_url"], state["issue_number"])
    
    issue = reader.fetch_issue(repo_url, issue_num)
    log_event("issue_parsed", f"Parsed issue: {issue.title}", {"title": issue.title})
    return {"issue": issue, "repo_url": repo_url, "issue_number": issue_num, "status": "running"}

async def clone_repo_node(state: AgentState) -> dict:
    manager = RepoManager()
    local_path = manager.clone(state["repo_url"])
    log_event("repo_cloned", f"Cloned to {local_path}", {"path": local_path})
    return {"repo_local_path": local_path}

async def index_codebase_node(state: AgentState) -> dict:
    # get a friendly name for the repo
    repo_name = state["repo_url"].split("/")[-1].replace(".git", "")
    
    index_result = await index_repository(state["repo_local_path"], repo_name)
    log_event("codebase_indexed", f"Indexed {index_result.total_chunks} chunks", {"chunks": index_result.total_chunks})
    
    return {
        "repo_map": index_result.repo_map,
        "target_languages": list(index_result.profile.languages.keys()),
        "qdrant_collection": index_result.collection_name,
        "language_profile": index_result.profile
    }

async def planner_node(state: AgentState) -> dict:
    plan = await planner.create_plan(state["issue"], state["repo_map"])
    log_event("plan_created", f"Created plan with {len(plan.sub_tasks)} tasks", {"tasks": len(plan.sub_tasks), "complexity": plan.complexity})
    return {
        "plan": plan,
        "current_task_index": 0,
        "retry_count": 0,
        "feedback_history": []
    }

async def search_code_node(state: AgentState) -> dict:
    plan = state["plan"]
    task = plan.sub_tasks[state["current_task_index"]]
    
    chunks = await retriever.retrieve(
        query=task.description,
        collection=state["qdrant_collection"],
        repo_map=state["repo_map"]
    )
    log_event("code_retrieved", f"Retrieved {len(chunks)} chunks for context", {"chunks": len(chunks)})
    return {"code_context": chunks}

async def code_writer_node(state: AgentState) -> dict:
    plan = state["plan"]
    task = plan.sub_tasks[state["current_task_index"]]
    
    changes = await code_writer.generate_changes(
        sub_task=task,
        code_context=state["code_context"],
        repo_path=state["repo_local_path"],
        feedback_history=state["feedback_history"]
    )
    log_event("code_generated", f"Generated {len(changes.operations)} file changes", {"files_changed": len(changes.operations)})
    return {"code_changes": changes.operations}

async def apply_patch_node(state: AgentState) -> dict:
    success = patch_applier.apply_all(state["repo_local_path"], state["code_changes"])
    log_event("patch_applied", "Patches applied successfully" if success else "Patch application failed", {"success": success})
    return {"applied_patch": success}

async def test_runner_node(state: AgentState) -> dict:
    results = await test_runner.run_tests(
        repo_path=state["repo_local_path"],
        language_profile=state["language_profile"],
        code_changes=state["code_changes"]
    )
    log_event("tests_run", f"Tests finished. {results.passed} passed, {results.failed} failed", {"passed": results.passed, "failed": results.failed})
    return {"test_results": results}

async def reviewer_node(state: AgentState) -> dict:
    review = await reviewer.review_changes(
        issue=state["issue"],
        code_changes=state["code_changes"],
        test_results=state["test_results"],
        feedback_history=state["feedback_history"],
        retry_count=state["retry_count"]
    )
    log_event("review_done", f"Review complete. Verdict: {review.verdict}", {"score": review.weighted_score, "verdict": review.verdict})

    updated = {"review": review}
    if review.verdict == "revise":
        updated["retry_count"] = state["retry_count"] + 1
        updated["feedback_history"] = state["feedback_history"] + [review.retry_suggestion]
        # revert so writer can try again cleanly
        patch_applier.revert_all(state["repo_local_path"], state["code_changes"])
        
    return updated

async def pr_agent_node(state: AgentState) -> dict:
    pr_url = await pr_agent.create_pull_request(
        repo_url=state["repo_url"],
        repo_path=state["repo_local_path"],
        issue=state["issue"],
        code_changes=state["code_changes"],
        test_results=state["test_results"],
        plan=state["plan"],
        total_cost=state.get("total_cost_usd", 0.0)
    )
    log_event("pr_created", f"PR Created! {pr_url}", {"url": pr_url})
    return {"pr_url": pr_url, "status": "success"}

async def failure_report_node(state: AgentState) -> dict:
    from src.github_client.pr_creator import PRCreator
    creator = PRCreator()
    creator.post_comment(
        state["repo_url"], 
        state["issue"].number, 
        f"🤖 The autonomous agent failed to complete this issue after {state['retry_count']} retries."
    )
    log_event("failure_reported", "Failed and commented on issue.")
    return {"status": "failed"}
