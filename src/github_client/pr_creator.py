from github import Github, GithubException
from typing import List

from config.settings import settings
from src.github_client.issue_reader import IssueReader

class PRCreator:
    """Handles creating and updating Pull Requests via PyGithub."""

    def __init__(self):
        self.gh = Github(settings.github_token)

    def create_pull_request(
        self,
        repo_url: str,
        issue_number: int,
        title: str,
        body: str,
        head_branch: str,
        base_branch: str = "main"
    ) -> str:
        """Create a PR, or update if it already exists."""
        owner, name = IssueReader.parse_repo_url(repo_url)
        repo = self.gh.get_repo(f"{owner}/{name}")

        try:
            pr = repo.create_pull(
                title=title,
                body=body,
                head=head_branch,
                base=base_branch
            )
            # Add labels to indicate this is automated
            try:
                pr.add_to_labels("automated", "ai-generated")
            except Exception:
                pass # ignore if labels don't exist

            return pr.html_url

        except GithubException as e:
            # Handle the case where the agent retried and pushed to an existing branch
            if "A pull request already exists" in str(e):
                pulls = repo.get_pulls(state='open', head=f"{owner}:{head_branch}")
                if pulls.totalCount > 0:
                    existing_pr = pulls[0]
                    existing_pr.edit(body=body) # update description with latest attempt details
                    return existing_pr.html_url
            
            raise e

    def post_comment(self, repo_url: str, issue_number: int, body: str):
        """Post a comment on the issue if we fail completely."""
        owner, name = IssueReader.parse_repo_url(repo_url)
        repo = self.gh.get_repo(f"{owner}/{name}")
        issue = repo.get_issue(issue_number)
        issue.create_comment(body)
