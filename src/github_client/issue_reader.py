from github import Github
from dataclasses import dataclass
from datetime import datetime
from typing import List

from config.settings import settings

@dataclass
class GitHubIssue:
    number: int
    title: str
    body: str
    labels: List[str]
    comments: List[str]
    author: str
    created_at: datetime

class IssueReader:
    """Reads issues and comments from GitHub via PyGithub."""

    def __init__(self):
        # uses token from environment via settings
        self.gh = Github(settings.github_token)

    def fetch_issue(self, repo_url: str, issue_number: int) -> GitHubIssue:
        """Fetch full context of an issue from a repo URL."""
        owner, name = self.parse_repo_url(repo_url)
        
        repo = self.gh.get_repo(f"{owner}/{name}")
        issue = repo.get_issue(issue_number)

        # grab any comments from maintainers
        comments = [c.body for c in issue.get_comments()]

        return GitHubIssue(
            number=issue.number,
            title=issue.title,
            body=issue.body or "",
            labels=[l.name for l in issue.labels],
            comments=comments,
            author=issue.user.login,
            created_at=issue.created_at
        )

    @staticmethod
    def parse_repo_url(repo_url: str) -> tuple[str, str]:
        """
        'https://github.com/user/repo' → ('user', 'repo')
        """
        clean_url = repo_url.rstrip("/").replace(".git", "")
        parts = clean_url.split("/")
        return parts[-2], parts[-1]

    @staticmethod
    def extract_from_issue_url(issue_url: str) -> tuple[str, int]:
        """
        'https://github.com/user/repo/issues/5' → ('https://github.com/user/repo', 5)
        """
        parts = issue_url.rstrip("/").split("/")
        issue_number = int(parts[-1])
        repo_url = "/".join(parts[:-2])
        return repo_url, issue_number
