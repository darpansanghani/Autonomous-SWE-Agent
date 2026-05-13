from dataclasses import dataclass
from src.utils.language_detect import detect_languages, RepoLanguageProfile
from src.utils.file_utils import list_files, safe_read
from src.rag.repo_map import generate_repo_map
from src.rag.chunker import parse_and_chunk
from src.rag.retriever import embed_and_store

@dataclass
class IndexResult:
    collection_name: str
    repo_map: dict
    profile: RepoLanguageProfile
    total_chunks: int

async def index_repository(repo_path: str, repo_name: str) -> IndexResult:
    """Full indexing pipeline: detect → parse → chunk → embed → store."""
    
    # 1. Detect language
    profile = detect_languages(repo_path)
    
    # 2. Map structure
    repo_map = generate_repo_map(repo_path, profile)
    
    # 3. Parse + chunk
    all_chunks = []
    files = list_files(repo_path)
    for f_path in files:
        if f_path.endswith((".py", ".js", ".ts", ".go", ".java", ".rs")):
            try:
                content = safe_read(repo_path, f_path)
                # primitive language detection based on extension
                lang = "python" if f_path.endswith(".py") else "javascript"
                chunks = parse_and_chunk(f_path, content, lang)
                all_chunks.extend(chunks)
            except Exception:
                pass
                
    # 4. Embed + store
    collection = f"repo_{repo_name}"
    await embed_and_store(collection, all_chunks)
    
    return IndexResult(
        collection_name=collection,
        repo_map=repo_map,
        profile=profile,
        total_chunks=len(all_chunks)
    )
