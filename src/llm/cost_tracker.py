from typing import List, Dict
from datetime import datetime
from collections import defaultdict

class CostTracker:
    """Tracks LLM spend per run. Kills the agent if budget is exceeded."""

    def __init__(self, budget_usd: float):
        self.budget = budget_usd
        self._entries: List[Dict] = []  # append-only

    def record(self, agent: str, model: str, input_tokens: int, output_tokens: int, cost: float):
        self._entries.append({
            "agent": agent,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost,
            "timestamp": datetime.now()
        })

    @property
    def total(self) -> float:
        """Total spend so far in USD."""
        return sum(e["cost"] for e in self._entries)

    def over_budget(self) -> bool:
        """Check this before every LLM call."""
        return self.total >= self.budget

    def summary(self) -> Dict[str, float]:
        """Returns spend broken down by agent role."""
        by_agent = defaultdict(float)
        for e in self._entries:
            by_agent[e["agent"]] += e["cost"]
        return dict(by_agent)

# One global instance per run
cost_tracker = CostTracker(budget_usd=5.0)
