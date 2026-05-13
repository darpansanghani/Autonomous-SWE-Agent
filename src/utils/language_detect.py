import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional, List

@dataclass
class RepoLanguageProfile:
    primary_language: str
    languages: Dict[str, int]
    total_files: int
    total_lines: int
    has_tests: bool
    test_framework: Optional[str]
    package_manager: Optional[str]

LANGUAGE_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".java": "java",
    ".rs": "rust",
}

def detect_languages(repo_path: str) -> RepoLanguageProfile:
    """Scan repo to find primary language, test framework, etc."""
    base = Path(repo_path)
    ignore_dirs = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build"}
    
    lang_counts = {}
    total_files = 0
    total_lines = 0
    has_tests = False
    
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        
        if "test" in Path(root).name.lower() or "tests" in Path(root).name.lower():
            has_tests = True
            
        for file in files:
            ext = Path(file).suffix
            if "test" in file.lower():
                has_tests = True
                
            if ext in LANGUAGE_MAP:
                lang = LANGUAGE_MAP[ext]
                lang_counts[lang] = lang_counts.get(lang, 0) + 1
                total_files += 1
                
                # Rough line count estimation without reading full files
                try:
                    with open(Path(root) / file, 'rb') as f:
                        total_lines += sum(1 for _ in f)
                except Exception:
                    pass

    primary_lang = max(lang_counts, key=lang_counts.get) if lang_counts else "unknown"
    
    # Simple test framework detection
    test_fw = None
    pkg_mgr = None
    
    if primary_lang == "python":
        if (base / "pytest.ini").exists() or "pytest" in safe_read_file(base / "requirements.txt"):
            test_fw = "pytest"
        pkg_mgr = "pip"
    elif primary_lang in ["javascript", "typescript"]:
        pkg_json = safe_read_file(base / "package.json")
        if "jest" in pkg_json: test_fw = "jest"
        elif "mocha" in pkg_json: test_fw = "mocha"
        pkg_mgr = "npm"
    elif primary_lang == "go":
        if (base / "go.mod").exists(): pkg_mgr = "go modules"
        test_fw = "go test"
        
    return RepoLanguageProfile(
        primary_language=primary_lang,
        languages=lang_counts,
        total_files=total_files,
        total_lines=total_lines,
        has_tests=has_tests,
        test_framework=test_fw,
        package_manager=pkg_mgr
    )

def safe_read_file(path: Path) -> str:
    try:
        if path.exists():
            return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        pass
    return ""
