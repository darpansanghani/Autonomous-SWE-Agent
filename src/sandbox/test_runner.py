import asyncio
import os
import re
import time
from dataclasses import dataclass, field
from typing import List, Any

from src.utils.language_detect import RepoLanguageProfile
from src.utils.logger import log_event

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
    failures: List[TestFailure] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""

def parse_pytest_failures(stdout: str) -> List[TestFailure]:
    """Parse pytest failure traceback outputs into structured models."""
    failures = []
    # Find traceback blocks: e.g. "____ test_name ____"
    fail_blocks = re.split(r"={3,}\s+FAILURES\s+={3,}", stdout)
    if len(fail_blocks) < 2:
        return failures
        
    failures_section = fail_blocks[1].split("=== short test summary info ===")[0]
    
    # Split by individual failure blocks e.g. "____ test_get_user_missing ____"
    individual_blocks = re.split(r"_+ (test_[a-zA-Z0-9_]+) _+", failures_section)
    
    # First item is empty or header
    for idx in range(1, len(individual_blocks), 2):
        if idx + 1 >= len(individual_blocks):
            break
        test_name = individual_blocks[idx]
        block_body = individual_blocks[idx + 1]
        
        # Look for file path and line number e.g., "tests/test_users.py:42: HTTPException"
        loc_match = re.search(r"([\w_/-]+\.py):(\d+): (.*)", block_body)
        file_path = loc_match.group(1) if loc_match else "unknown_file"
        line_number = int(loc_match.group(2)) if loc_match else 0
        error_message = loc_match.group(3).strip() if loc_match else "Test failure occurred."
        
        # Try to parse error type e.g., "HTTPException"
        err_type_match = re.search(r"E\s+(\w+Error|\w+Exception):?", block_body)
        error_type = err_type_match.group(1) if err_type_match else "AssertionError"
        
        failures.append(TestFailure(
            test_name=test_name,
            file_path=file_path,
            line_number=line_number,
            error_message=error_message,
            error_type=error_type
        ))
        
    return failures

def parse_jest_failures(stdout: str, stderr: str) -> List[TestFailure]:
    """Parse Jest test runner failure logs into structured models."""
    failures = []
    combined = stdout + "\n" + stderr
    
    # Jest individual failure block: starts with "  ● "
    blocks = combined.split("  ● ")
    if len(blocks) < 2:
        return failures
        
    for block in blocks[1:]:
        lines = block.splitlines()
        if not lines:
            continue
            
        test_name = lines[0].strip()
        
        # Look for error details and file path e.g. "at Object.<anonymous> (tests/users.test.js:14:10)"
        file_path = "unknown_file"
        line_number = 0
        
        loc_match = re.search(r"\(([\w_/-]+\.(?:js|ts)):(\d+):\d+\)", block)
        if loc_match:
            file_path = loc_match.group(1)
            line_number = int(loc_match.group(2))
            
        error_message = "Test failed in Jest."
        for line in lines[1:]:
            if "Error:" in line or "expect(" in line:
                error_message = line.strip()
                break
                
        failures.append(TestFailure(
            test_name=test_name,
            file_path=file_path,
            line_number=line_number,
            error_message=error_message,
            error_type="JestError"
        ))
        
    return failures

async def run_tests(
    repo_path: str,
    language_profile: RepoLanguageProfile,
    code_changes: List[Any],
    timeout: int = 120
) -> TestResults:
    """Execute testing suite in subprocess with structured telemetry and detailed failure parsing."""
    
    # Resolve correct runner command
    if language_profile.test_framework == "pytest":
        cmd = ["pytest", "--tb=short", "-q"]
    elif language_profile.test_framework == "jest":
        cmd = ["npm", "test"]
    else:
        cmd = ["pytest"] # default baseline fallback
        
    if not language_profile.has_tests:
        log_event("testing_skipped", "No test framework or test directory was detected.")
        return TestResults(
            language=language_profile.primary_language,
            command_run="none",
            exit_code=0,
            total=0, passed=0, failed=0,
            duration_seconds=0.0,
            failures=[],
            stdout="No tests found.", stderr=""
        )

    log_event("testing_started", f"Running test suite: {' '.join(cmd)}")
    
    env = os.environ.copy()
    # Absolute safety: block internet outbound network operations in sandbox execution
    env["HTTP_PROXY"] = ""
    env["HTTPS_PROXY"] = ""
    env["NO_PROXY"] = "*"
    
    start_time = time.perf_counter()
    
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=repo_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env
    )
    
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        duration = time.perf_counter() - start_time
        stdout = stdout_bytes.decode(errors='replace')
        stderr = stderr_bytes.decode(errors='replace')
    except asyncio.TimeoutError:
        proc.kill()
        log_event("testing_timeout", f"Test suite timed out after {timeout} seconds.")
        return TestResults(
            language=language_profile.primary_language,
            command_run=" ".join(cmd),
            exit_code=-1,
            total=0, passed=0, failed=1,
            duration_seconds=timeout,
            failures=[TestFailure("timeout", "", 0, "Tests timed out.", "TimeoutError")],
            stdout="", stderr=f"TIMEOUT after {timeout}s"
        )
        
    passed = 0
    failed = 0
    failures = []
    
    # Parse pytest results
    if "pytest" in cmd:
        summary_match = re.search(r"(\d+) passed(?:, (\d+) failed)?", stdout)
        if summary_match:
            passed = int(summary_match.group(1))
            failed = int(summary_match.group(2) or 0)
        else:
            # check for "no tests ran in" or syntax compilation errors
            if proc.returncode != 0:
                failed = 1
                
        # Extract failures
        if failed > 0:
            failures = parse_pytest_failures(stdout)
            
    # Parse Jest results
    elif "npm" in cmd or "jest" in cmd:
        # e.g., "Tests:       1 failed, 12 passed, 13 total"
        summary_match = re.search(r"Tests:\s+(?:(\d+) failed,\s+)?(?:(\d+) passed,\s+)?(\d+) total", stderr + "\n" + stdout)
        if summary_match:
            failed = int(summary_match.group(1) or 0)
            passed = int(summary_match.group(2) or 0)
        else:
            if proc.returncode != 0:
                failed = 1
                
        # Extract failures
        if failed > 0:
            failures = parse_jest_failures(stdout, stderr)

    total = passed + failed
    log_event("testing_finished", f"Completed tests. Result: {passed} passed, {failed} failed in {duration:.2f}s")
    
    return TestResults(
        language=language_profile.primary_language,
        command_run=" ".join(cmd),
        exit_code=proc.returncode,
        total=total,
        passed=passed,
        failed=failed,
        duration_seconds=round(duration, 2),
        failures=failures,
        stdout=stdout[-2000:],
        stderr=stderr[-1000:]
    )
