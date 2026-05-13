import os
import sys
import shutil
from pathlib import Path

# Need settings to push to github
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.settings import settings

from github import Github

# Local dummy project files
BACKEND_MAIN = """from fastapi import FastAPI, HTTPException
app = FastAPI()

users_db = {1: {"name": "Alice"}, 2: {"name": "Bob"}}

@app.get("/users/{user_id}")
def get_user(user_id: int):
    # Bug: throws KeyError if user doesn't exist
    return users_db[user_id]
"""

BACKEND_REQS = """fastapi
uvicorn
"""

README = """# Playground Repo

Welcome to the SWE Agent playground. There is a small typo right hear.
"""

def create_playground():
    print("Setting up dummy repository...")
    if not settings.github_token:
        print("ERROR: Please set GITHUB_TOKEN in .env")
        return

    gh = Github(settings.github_token)
    user = gh.get_user()
    
    repo_name = "swe-agent-playground"
    try:
        repo = user.create_repo(repo_name, private=False)
        print(f"Created repo: {repo.html_url}")
    except Exception as e:
        print(f"Repo might already exist: {e}")
        repo = user.get_repo(repo_name)
    
    # 1. Push files
    try:
        repo.create_file("README.md", "Initial commit", README, branch="main")
        repo.create_file("backend/main.py", "Add FastAPI backend", BACKEND_MAIN, branch="main")
        repo.create_file("requirements.txt", "Add reqs", BACKEND_REQS, branch="main")
        print("Pushed starter files.")
    except Exception:
        print("Files already exist, skipping...")

    # 2. Create Issues
    try:
        repo.create_issue(
            title="Fix typo in README",
            body="There is a typo 'hear' instead of 'here'. Please fix it.",
            labels=["good first issue"]
        )
        repo.create_issue(
            title="get_user returns 500 on missing user",
            body="If I request `/users/99`, the server crashes with a KeyError. It should return a 404 HTTP error instead.",
            labels=["bug"]
        )
        print("Issues created successfully.")
    except Exception as e:
        print(f"Failed to create issues: {e}")

    print("\n--- Setup Complete ---")
    print(f"Target Repo URL: {repo.html_url}")
    print("Go to the Streamlit UI and paste this URL into the Issue input box (e.g., .../issues/1)!")

if __name__ == "__main__":
    create_playground()
