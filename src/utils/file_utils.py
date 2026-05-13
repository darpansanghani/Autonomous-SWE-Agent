import os
from pathlib import Path
from typing import List

def safe_read(repo_path: str, file_path: str) -> str:
    """Read a file safely, preventing directory traversal attacks."""
    base = Path(repo_path).resolve()
    target = (base / file_path).resolve()

    if not str(target).startswith(str(base)):
        raise ValueError(f"Path traversal detected: {file_path}")

    if not target.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(target, "r", encoding="utf-8", errors="replace") as f:
        return f.read()

def safe_write(repo_path: str, file_path: str, content: str):
    """Write a file safely."""
    base = Path(repo_path).resolve()
    target = (base / file_path).resolve()

    if not str(target).startswith(str(base)):
        raise ValueError(f"Path traversal detected: {file_path}")

    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        f.write(content)

def list_files(repo_path: str, ignore_dirs: set = None) -> List[str]:
    """List all files in repo, relative to repo_path."""
    ignore = ignore_dirs or {".git", "node_modules", "__pycache__", ".venv", "venv"}
    
    files = []
    base = Path(repo_path)
    for root, dirs, filenames in os.walk(base):
        # modify dirs in-place to skip ignored directories
        dirs[:] = [d for d in dirs if d not in ignore]
        
        for name in filenames:
            full_path = Path(root) / name
            try:
                # return relative path as string
                files.append(str(full_path.relative_to(base).as_posix()))
            except ValueError:
                pass
                
    return files
