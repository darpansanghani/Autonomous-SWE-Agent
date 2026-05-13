import asyncio
import os
from dataclasses import dataclass
from typing import List, Any
import re

from src.utils.language_detect import RepoLanguageProfile

@dataclass
class TestFailure:
    test_name: str
    file_path: str
    line_number: int
    error_message: str
    error_type: str

@dataclass
class TestResults:
    language: str
    command_run: str
    exit_code: int
    total: int
    passed: int
    failed: int
    duration_seconds: float
    failures: List[TestFailure]
    stdout: str
    stderr: str

async def run_tests(
    repo_path: str,
    language_profile: RepoLanguageProfile,
    code_changes: List[Any],
    timeout: int = 120
) -> TestResults:
    """Complete test execution pipeline in a subprocess."""
    
    cmd = ["pytest", "--tb=short", "-q"] if language_profile.test_framework == "pytest" else ["npm", "test"]
    if not language_profile.has_tests:
        # no tests found
        return TestResults(
            language=language_profile.primary_language,
            command_run="none",
            exit_code=0,
            total=0, passed=0, failed=0,
            duration_seconds=0.0,
            failures=[],
            stdout="No tests found.", stderr=""
        )

    # Execute
    env = os.environ.copy()
    env["HTTP_PROXY"] = ""
    
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=repo_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env
    )
    
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        stdout = stdout_bytes.decode(errors='replace')
        stderr = stderr_bytes.decode(errors='replace')
    except asyncio.TimeoutError:
        proc.kill()
        return TestResults(
            language=language_profile.primary_language,
            command_run=" ".join(cmd),
            exit_code=-1,
            total=0, passed=0, failed=1,
            duration_seconds=timeout,
            failures=[TestFailure("timeout", "", 0, "Tests timed out.", "TimeoutError")],
            stdout="", stderr=f"TIMEOUT after {timeout}s"
        )
        
    # Basic parsing
    passed = 0
    failed = 1 if proc.returncode != 0 else 0
    total = passed + failed
    
    # Try to extract pass/fail from pytest output
    if "pytest" in cmd:
        summary_match = re.search(r"(\d+) passed(?:, (\d+) failed)?", stdout)
        if summary_match:
            passed = int(summary_match.group(1))
            failed = int(summary_match.group(2) or 0)
            total = passed + failed

    return TestResults(
        language=language_profile.primary_language,
        command_run=" ".join(cmd),
        exit_code=proc.returncode,
        total=total,
        passed=passed,
        failed=failed,
        duration_seconds=0.0,
        failures=[],
        stdout=stdout[-2000:],
        stderr=stderr[-1000:]
    )
