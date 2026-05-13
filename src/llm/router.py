import json
import litellm
from typing import List, Dict, Optional, Any

from config.settings import settings
from src.utils.logger import log_event
from src.llm.cost_tracker import cost_tracker

# Suppress annoying debug logs from litellm
litellm.set_verbose = False

# Model assignment mapping
MODEL_CONFIG = {
    "planner": {
        "primary": settings.planner_model,
        "fallback": "gpt-4o",
        "temperature": 0.2,
        "max_tokens": 4096,
    },
    "code_writer": {
        "primary": settings.code_writer_model,
        "fallback": "gemini/gemini-1.5-pro",
        "temperature": 0.1,  # low = more deterministic code
        "max_tokens": 8192,
    },
    "reviewer": {
        "primary": settings.reviewer_model,
        "fallback": "gemini/gemini-1.5-pro",
        "temperature": 0.1,
        "max_tokens": 4096,
    },
    "rag_reranker": {
        "primary": settings.rag_query_model,
        "fallback": "gemini/gemini-1.5-flash",
        "temperature": 0.0,
        "max_tokens": 2048,
    },
    "pr_writer": {
        "primary": settings.pr_writer_model,
        "fallback": "gpt-4o-mini",
        "temperature": 0.3,
        "max_tokens": 2048,
    },
}

class LLMError(Exception):
    pass

async def agent_completion(
    agent_name: str,
    messages: List[Dict[str, str]],
    response_format: Optional[Dict[str, Any]] = None,
    tools: Optional[List[Dict[str, Any]]] = None
):
    """
    Call LLM with automatic fallback to secondary model and cost tracking.
    """
    if cost_tracker.over_budget():
        raise LLMError(f"Budget exceeded (${cost_tracker.total:.2f} >= ${cost_tracker.budget:.2f}). Aborting.")

    config = MODEL_CONFIG[agent_name]
    models_to_try = [config["primary"], config["fallback"]]

    for model in models_to_try:
        try:
            # LiteLLM normalizes the API surface across OpenAI/Google/etc
            response = await litellm.acompletion(
                model=model,
                messages=messages,
                temperature=config["temperature"],
                max_tokens=config["max_tokens"],
                response_format=response_format,
                tools=tools,
            )
            
            # Track cost transparently
            cost_usd = litellm.completion_cost(response) or 0.0
            cost_tracker.record(
                agent=agent_name,
                model=model,
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                cost=cost_usd
            )
            
            return response

        except Exception as e:
            # log failure, try next model
            log_event("llm_fallback", f"Model {model} failed for {agent_name}: {str(e)}", data={"model": model, "error": str(e)})
            continue

    raise LLMError(f"All models failed for agent {agent_name}.")
