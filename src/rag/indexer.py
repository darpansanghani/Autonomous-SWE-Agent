from dataclasses import dataclass
from src.utils.language_detect import detect_languages, RepoLanguageProfile
from src.utils.file_utils import list_files, safe_read
from src.rag.repo_map import generate_repo_map
from src.rag.chunker import parse_and_chunk
from src.rag.embedder import embed_chunks, store_in_qdrant
from src.utils.logger import log_event

@dataclass
class IndexResult:
    collection_name: str
    repo_map: dict
    profile: RepoLanguageProfile
    total_chunks: int

async def index_repository(repo_path: str, repo_name: str) -> IndexResult:
    """Full indexing pipeline: detect → parse → chunk → embed → store."""
    log_event("indexing_started", f"Starting full repository indexing for {repo_name}")
    
    # 1. Detect language profile
    profile = detect_languages(repo_path)
    log_event("language_detected", f"Primary language detected: {profile.primary_language}", {"profile": str(profile)})
    
    # 2. Map structural code outline
    repo_map = generate_repo_map(repo_path, profile)
    log_event("repo_mapped", f"Generated JSON structural repository map.", {"files": len(repo_map.get("structure", {}))})
    
    # 3. Parse + chunk files recursively
    all_chunks = []
    files = list_files(repo_path)
    
    for f_path in files:
        # Detect file extension
        if f_path.endswith((".py", ".js", ".ts")):
            try:
                content = safe_read(repo_path, f_path)
                lang = "python" if f_path.endswith(".py") else "javascript"
                chunks = parse_and_chunk(f_path, content, lang)
                all_chunks.extend(chunks)
            except Exception as e:
                log_event("chunk_parsing_warning", f"Skipped parsing chunk for {f_path} due to error: {str(e)}")
                
    log_event("chunks_created", f"Created {len(all_chunks)} parsed AST code chunks.")
    
    # 4. Embed chunks with context-awareness and fallbacks
    embedded_chunks = await embed_chunks(all_chunks, repo_map)
    log_event("chunks_embedded", f"Completed batch embedding for {len(embedded_chunks)} chunks.")
    
    # 5. Save chunks into Qdrant collection
    collection = f"repo_{repo_name.lower().replace('-', '_')}"
    await store_in_qdrant(collection, embedded_chunks)
    log_event("chunks_stored", f"Successfully indexed collection: {collection}")
    
    return IndexResult(
        collection_name=collection,
        repo_map=repo_map,
        profile=profile,
        total_chunks=len(embedded_chunks)
    )
