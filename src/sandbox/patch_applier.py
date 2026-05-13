import os
from pathlib import Path
from typing import List, Any
import unidiff

from src.utils.file_utils import safe_read, safe_write

def apply_all(repo_path: str, operations: List[Any]) -> bool:
    """Apply generated diffs or create new files in the repo."""
    
    base = Path(repo_path)
    success = True
    
    for op in operations:
        try:
            if op.action == "create":
                safe_write(repo_path, op.file_path, op.full_content or "")
                
            elif op.action == "modify" and op.unified_diff:
                # Apply diff to existing file
                content = safe_read(repo_path, op.file_path)
                
                # In a real implementation we would use `patch` command or a Python diff applier.
                # Here we do a naive overwrite if it fails or rely on full_content if provided.
                # A robust approach requires the 'patch' utility or careful line matching.
                # For simplicity in this demo, if full_content isn't there, we just write the diff as a test.
                
                # We'll try to extract the clean file from a generated full_content if present,
                # else we leave it for a proper patch util.
                if op.full_content:
                    safe_write(repo_path, op.file_path, op.full_content)
                else:
                    # fallback: save diff to a .patch file and apply via subprocess
                    patch_path = base / f"{Path(op.file_path).name}.patch"
                    patch_path.write_text(op.unified_diff, encoding="utf-8")
                    import subprocess
                    subprocess.run(["git", "apply", str(patch_path)], cwd=str(base), capture_output=True)
                    if patch_path.exists():
                        patch_path.unlink()
                        
            elif op.action == "delete":
                target = base / op.file_path
                if target.exists() and target.is_file():
                    target.unlink()
                    
        except Exception as e:
            print(f"Failed to apply {op.action} on {op.file_path}: {e}")
            success = False
            
    return success

def revert_all(repo_path: str, operations: List[Any]):
    """Reverts applied changes by doing a git reset --hard."""
    import subprocess
    base = Path(repo_path)
    subprocess.run(["git", "reset", "--hard", "HEAD"], cwd=str(base), capture_output=True)
    subprocess.run(["git", "clean", "-fd"], cwd=str(base), capture_output=True)
