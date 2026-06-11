from src.graph.state import AgentState

def route_after_review(state: AgentState) -> str:
    """Determine the next step after the reviewer has cast a verdict."""
    review = state["review"]
    retry_count = state["retry_count"]
    max_retries = state["plan"].estimated_retries_budget if state.get("plan") else 3

    if review.verdict == "approve":
        # We need to check if there are more subtasks in the plan
        current_idx = state["current_task_index"]
        plan = state["plan"]
        
        if current_idx + 1 < len(plan.sub_tasks):
            # move to next task
            state["current_task_index"] = current_idx + 1
            return "search_code"
        else:
            # all done!
            return "approve"
            
    elif review.verdict == "revise" and retry_count < max_retries:
        return "revise"
        
    else:
        # reject or ran out of retries
        return "reject"
