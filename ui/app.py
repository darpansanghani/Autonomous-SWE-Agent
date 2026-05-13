import streamlit as st
import threading
import time
from datetime import datetime
from uuid import uuid4

# Setup paths so imports work correctly
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.logger import drain_log_queue
from src.graph.workflow import build_agent_graph
from db.database import init_db
from config.settings import settings

st.set_page_config(page_title="🤖 SWE Agent", layout="wide")

# ensure DB is initialized on first load
@st.cache_resource
def init_system():
    init_db()
    return True

init_system()

# ── Background Worker ───────────────────────────────────────────

def run_agent_background(issue_url: str):
    """Executes LangGraph pipeline in background thread."""
    try:
        graph = build_agent_graph()
        
        # We start with the issue URL and let the graph handle parsing
        initial_state = {
            "issue_url": issue_url,
            "status": "running",
            "retry_count": 0,
            "feedback_history": [],
            "total_cost_usd": 0.0,
            "run_id": uuid4().hex[:8],
        }

        # blocking call within this thread
        final_state = graph.invoke(initial_state)

        st.session_state["run_result"] = {
            "status": final_state.get("status", "unknown"),
            "pr_url": final_state.get("pr_url"),
            "cost": final_state.get("total_cost_usd", 0.0),
            "retries": final_state.get("retry_count", 0),
        }
    except Exception as e:
        st.session_state["run_result"] = {"status": "failed", "error": str(e)}

# ── UI Construction ──────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["🚀 Run Agent", "📜 Run History", "📝 Diff Viewer"])

with tab1:
    st.markdown("### 🤖 Autonomous SWE Agent")
    st.caption("Provide a GitHub issue URL to let the agent solve it automatically.")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        issue_url = st.text_input("GitHub Issue URL", placeholder="https://github.com/user/repo/issues/1")
    with col2:
        model_choice = st.selectbox("LLM Profile", ["GPT-4o (Default)", "Gemini 1.5 Pro"])
        st.write("") # spacer
        
    if st.button("▶ Run Agent", type="primary", use_container_width=True):
        if issue_url:
            st.session_state["run_active"] = True
            st.session_state["run_logs"] = []
            st.session_state["run_result"] = None
            
            thread = threading.Thread(
                target=run_agent_background,
                args=(issue_url,),
                daemon=True
            )
            thread.start()
        else:
            st.warning("Please provide an issue URL.")

    st.divider()
    st.markdown("### 📋 Agent Log")
    
    # ── Live Streaming Container ──
    log_container = st.empty()
    
    # Initialize state variables if they don't exist
    if "run_active" not in st.session_state:
        st.session_state["run_active"] = False
    if "run_logs" not in st.session_state:
        st.session_state["run_logs"] = []

    # Polling loop if agent is running
    if st.session_state["run_active"]:
        while st.session_state["run_active"]:
            new_logs = drain_log_queue()
            if new_logs:
                st.session_state["run_logs"].extend(new_logs)

            with log_container.container():
                for log in st.session_state["run_logs"]:
                    st.markdown(f"{log['emoji']} **[{log['timestamp']}]** {log['message']}")

            # Check for completion
            if st.session_state.get("run_result"):
                result = st.session_state["run_result"]
                if result["status"] == "success":
                    st.success(f"🎉 PR created: [{result['pr_url']}]({result['pr_url']})")
                else:
                    st.error(f"❌ Agent failed: {result.get('error', 'Max retries reached.')}")
                
                # Show cost
                st.info(f"💰 Total Cost: ${result.get('cost', 0.0):.2f}")
                
                st.session_state["run_active"] = False
                break
                
            time.sleep(0.5)
            # st.rerun() handles forcing UI updates cleanly in newer streamlit
            st.rerun()

    # if not running but we have logs, render them statically
    elif st.session_state["run_logs"]:
        with log_container.container():
            for log in st.session_state["run_logs"]:
                st.markdown(f"{log['emoji']} **[{log['timestamp']}]** {log['message']}")
            
            result = st.session_state.get("run_result", {})
            if result.get("status") == "success":
                st.success(f"🎉 PR created: [{result['pr_url']}]({result['pr_url']})")
            elif result.get("status") == "failed":
                st.error(f"❌ Agent failed: {result.get('error', 'Max retries reached.')}")

with tab2:
    st.markdown("### 📜 Run History")
    st.info("SQL database wiring for history goes here. (Phase 3 implementation)")

with tab3:
    st.markdown("### 📝 Diff Viewer")
    st.info("Diff rendering for historical runs goes here. (Phase 3 implementation)")
