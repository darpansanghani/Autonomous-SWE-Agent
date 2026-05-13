import queue
import threading
import structlog
from datetime import datetime
from typing import Optional, Dict, Any

# Configure structlog for human-readable console output
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ]
)

log = structlog.get_logger()

# Global log queue — nodes write, UI reads
_log_queue: queue.Queue = queue.Queue()
_log_lock = threading.Lock()

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
    "llm_fallback": "⚠️",
}

def log_event(event_type: str, message: str, data: Optional[Dict[str, Any]] = None):
    """Called by graph nodes to emit progress events to the UI and terminal."""
    emoji = EVENT_EMOJIS.get(event_type, "ℹ️")
    entry = {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "event": event_type,
        "data": data or {},
        "emoji": emoji,
        "message": message
    }
    
    # Push to UI queue
    with _log_lock:
        _log_queue.put(entry)
        
    # Also log to terminal so we can see it in CLI mode
    log.info(message, event=event_type, **(data or {}))

def drain_log_queue() -> list[dict]:
    """Called by Streamlit UI to get all pending log messages."""
    entries = []
    with _log_lock:
        while not _log_queue.empty():
            entries.append(_log_queue.get())
    return entries
