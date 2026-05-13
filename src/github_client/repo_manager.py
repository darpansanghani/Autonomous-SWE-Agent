import os
import shutil
from pathlib import Path
from uuid import uuid4
from git import Repo as GitRepo

from config.settings import settings
from src.utils.logger import log_event

class RepoManager:
    """Handles git operations: clone, branch, commit, push."""

    def __init__(self):
        self.token = settings.github_token

    def clone(self, repo_url: str) -> str:
        """Clone to a temp directory. Returns the local path."""
        # inject token into URL for private repo access without prompts
        auth_url = repo_url.replace(
            "https://github.com/",
            f"https://{self.token}@github.com/"
        ) if self.token and "github.com" in repo_url else repo_url

        # generate a unique path for this run
        run_id = uuid4().hex[:8]
        local_path = Path(settings.sandbox_work_dir) / f"run_{run_id}"
        
        if local_path.exists():
            shutil.rmtree(local_path)
            
        # shallow clone — full history wastes time on large repos
        log_event("repo_cloned", f"Cloning {repo_url} to {local_path}...", {"path": str(local_path)})
        GitRepo.clone_from(auth_url, local_path, depth=1)
        
        return str(local_path)

    def create_branch(self, repo_path: str, branch_name: str):
        """Checkout a new branch."""
        repo = GitRepo(repo_path)
        repo.git.checkout("-b", branch_name)

    def commit(self, repo_path: str, message: str):
        """Stage all changes and commit."""
        repo = GitRepo(repo_path)
        repo.git.add("--all")
        # Ensure we always have an author set for automated commits
        author = "SWE Agent <agent@swe-agent.dev>"
        repo.git.commit("-m", message, "--author", author)

    def push(self, repo_path: str, branch_name: str):
        """Push branch to origin."""
        repo = GitRepo(repo_path)
        # force push in case of retries on the same branch
        repo.git.push("-u", "origin", branch_name, "--force")
