import os
import json
from pathlib import Path

from src.utils.language_detect import RepoLanguageProfile

def generate_repo_map(repo_path: str, profile: RepoLanguageProfile) -> dict:
    """
    Creates a lightweight JSON representation of the repo.
    In a full implementation, this uses tree-sitter to extract function/class signatures.
    Here we provide a robust file-structure mapping with basic class/function regex detection for Python.
    """
    base = Path(repo_path)
    ignore_dirs = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build"}
    
    structure = {}
    
    import re
    py_class_re = re.compile(r"^class\s+([a-zA-Z_]\w*)", re.MULTILINE)
    py_func_re = re.compile(r"^def\s+([a-zA-Z_]\w*)", re.MULTILINE)
    
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        
        for file in files:
            ext = Path(file).suffix
            if ext not in [".py", ".js", ".ts", ".go", ".java", ".rs", ".md", ".json"]:
                continue
                
            full_path = Path(root) / file
            rel_path = full_path.relative_to(base).as_posix()
            
            file_info = {"type": "file", "classes": [], "functions": []}
            
            # Simple regex-based map for Python as a fast fallback
            if ext == ".py":
                try:
                    content = full_path.read_text(encoding="utf-8", errors="ignore")
                    file_info["classes"] = [{"name": m} for m in py_class_re.findall(content)]
                    file_info["functions"] = py_func_re.findall(content)
                except Exception:
                    pass
                    
            structure[rel_path] = file_info

    return {
        "total_files": profile.total_files,
        "languages": profile.languages,
        "structure": structure
    }
