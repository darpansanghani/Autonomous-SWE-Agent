import re
import json
import litellm
from typing import List, Dict
from qdrant_client.models import Filter, FieldCondition, MatchValue

from config.settings import settings
from src.rag.chunker import CodeChunk
from src.rag.embedder import get_qdrant
from src.utils.logger import log_event

def extract_identifiers(query: str) -> List[str]:
    """Helper to extract potential code identifiers (class, function names) from natural query."""
    return re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", query)

def keyword_search(query: str, repo_map: dict) -> List[str]:
    """Search the structural repo map for exact matches on files, classes, or function names."""
    identifiers = extract_identifiers(query)
    if not identifiers:
        return []
        
    matched_files = []
    
    for file_path, file_info in repo_map.get("structure", {}).items():
        # Match file name
        if any(ident in file_path for ident in identifiers):
            matched_files.append(file_path)
            continue
            
        # Match class names
        matched_class = False
        for cls in file_info.get("classes", []):
            if any(ident == cls["name"] for ident in identifiers):
                matched_files.append(file_path)
                matched_class = True
                break
        if matched_class:
            continue
            
        # Match function names
        for func in file_info.get("functions", []):
            if any(ident == func for ident in identifiers):
                matched_files.append(file_path)
                break
                
    return list(set(matched_files))

async def semantic_search(query: str, collection: str, top_k: int = 10) -> List[CodeChunk]:
    """Search local Qdrant collection using semantic embeddings with a score threshold."""
    client = get_qdrant()
    
    model = "openai/text-embedding-3-small"
    if not settings.openai_api_key and settings.google_api_key:
        model = "gemini/text-embedding-004"
        litellm.api_key = settings.google_api_key
    elif settings.openai_api_key:
        litellm.api_key = settings.openai_api_key
        
    try:
        # Embed search query
        response = await litellm.aembedding(
            model=model,
            input=[query]
        )
        query_vector = response.data[0]["embedding"]
        
        results = client.search(
            collection_name=collection,
            query_vector=query_vector,
            limit=top_k,
            score_threshold=0.3
        )
        
        return [
            CodeChunk(
                chunk_id=str(r.id),
                file_path=r.payload.get("file_path", ""),
                language=r.payload.get("language", ""),
                chunk_type=r.payload.get("chunk_type", "retrieved"),
                name=r.payload.get("name", ""),
                signature=r.payload.get("signature", ""),
                body=r.payload.get("body", ""),
                start_line=r.payload.get("start_line", 0),
                end_line=r.payload.get("end_line", 0),
                parent_class=r.payload.get("parent_class", None),
                relevance_score=r.score
            )
            for r in results
        ]
    except Exception as e:
        log_event("retrieval_error", f"Semantic search failed: {str(e)}", {"error": str(e)})
        return []

def expand_dependencies(found_files: List[str], dependency_graph: Dict[str, List[str]], max_depth: int = 1) -> List[str]:
    """Incorporate high-probability imported files to enrich retrieval depth."""
    expanded = set(found_files)
    
    for _ in range(max_depth):
        new_files = set()
        for f in list(expanded):
            deps = dependency_graph.get(f, [])
            new_files.update(deps)
        expanded.update(new_files)
        
    return list(expanded)

RERANK_PROMPT = """Given this software engineering task: "{query}"

Score each of the code chunks below from 1 to 10 based on how essential it is to understand or modify this code to complete the task.

Return a JSON object containing a list under "scores", matching this format:
{{
  "scores": [
    {{"chunk_id": "chunk_id_here", "score": 9, "reason": "Defines the main class requiring correction."}}
  ]
}}

Chunks to evaluate:
{chunks_text}"""

async def rerank_chunks(query: str, chunks: List[CodeChunk]) -> List[CodeChunk]:
    """Use a fast Llama/LLM model to grade and sort the chunks, keeping only the Top-8 best contexts."""
    if not chunks:
        return []
        
    # Format chunks cleanly
    chunks_text = "\n---\n".join(
        f"[ID: {c.chunk_id}] {c.file_path} > {c.name or 'module'}\n```\n{c.body[:400]}\n```"
        for c in chunks
    )
    
    try:
        response = await litellm.acompletion(
            model=settings.rag_query_model or "openai/gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a professional codebase relevance scoring engine."},
                {"role": "user", "content": RERANK_PROMPT.format(query=query, chunks_text=chunks_text)}
            ],
            response_format={"type": "json_object"}
        )
        
        raw = response.choices[0].message.content
        data = json.loads(raw)
        
        score_map = {s["chunk_id"]: s["score"] for s in data.get("scores", [])}
        
        for chunk in chunks:
            chunk.relevance_score = score_map.get(chunk.chunk_id, 0.0)
            
    except Exception as e:
        log_event("reranking_error", f"Reranking failed: {str(e)}. Falling back to original order.", {"error": str(e)})
        # If reranking fails, default to existing order based on semantic scores
        for idx, chunk in enumerate(chunks):
            chunk.relevance_score = chunk.relevance_score or (10.0 - idx)
            
    # Sort and return Top-8
    sorted_chunks = sorted(chunks, key=lambda c: c.relevance_score, reverse=True)
    return sorted_chunks[:8]

async def retrieve(query: str, collection: str, repo_map: dict, top_k: int = 8) -> List[CodeChunk]:
    """Orchestrates the entire multi-tier retrieval pipeline: Keyword -> Semantic -> Deduplicate -> Dependency -> Rerank."""
    log_event("retrieval_started", f"Starting multi-tier codebase retrieval for query: {query}")
    
    # 1. Keyword search
    keyword_files = keyword_search(query, repo_map)
    
    # 2. Semantic search
    semantic_chunks = await semantic_search(query, collection, top_k=15)
    
    # Merge keyword & semantic chunks
    merged_chunks = {c.chunk_id: c for c in semantic_chunks}
    
    # If keyword search matched files, pull their first few lines or functions as context
    # by fetching points from Qdrant associated with those files
    if keyword_files:
        client = get_qdrant()
        try:
            for kw_file in keyword_files:
                matched_points = client.scroll(
                    collection_name=collection,
                    scroll_filter=Filter(
                        must=[FieldCondition(key="file_path", match=MatchValue(value=kw_file))]
                    ),
                    limit=5
                )[0]
                
                for r in matched_points:
                    chunk_id = str(r.id)
                    if chunk_id not in merged_chunks:
                        merged_chunks[chunk_id] = CodeChunk(
                            chunk_id=chunk_id,
                            file_path=r.payload.get("file_path", ""),
                            language=r.payload.get("language", ""),
                            chunk_type=r.payload.get("chunk_type", "retrieved"),
                            name=r.payload.get("name", ""),
                            signature=r.payload.get("signature", ""),
                            body=r.payload.get("body", ""),
                            start_line=r.payload.get("start_line", 0),
                            end_line=r.payload.get("end_line", 0),
                            parent_class=r.payload.get("parent_class", None),
                            relevance_score=5.0 # baseline keyword score
                        )
        except Exception as e:
            log_event("keyword_fetch_error", f"Failed fetching keyword matched chunks: {str(e)}")
            
    # 3. Dependency expansion (1 level deep)
    initial_files = list(set(c.file_path for c in merged_chunks.values()))
    expanded_files = expand_dependencies(
        found_files=initial_files,
        dependency_graph=repo_map.get("dependency_graph", {}),
        max_depth=1
    )
    
    # Add dependency files chunks if missing
    new_files_to_fetch = [f for f in expanded_files if f not in initial_files]
    if new_files_to_fetch:
        client = get_qdrant()
        try:
            for dep_file in new_files_to_fetch[:3]: # limit to first 3 files to avoid overload
                matched_points = client.scroll(
                    collection_name=collection,
                    scroll_filter=Filter(
                        must=[FieldCondition(key="file_path", match=MatchValue(value=dep_file))]
                    ),
                    limit=2
                )[0]
                for r in matched_points:
                    chunk_id = str(r.id)
                    if chunk_id not in merged_chunks:
                        merged_chunks[chunk_id] = CodeChunk(
                            chunk_id=chunk_id,
                            file_path=r.payload.get("file_path", ""),
                            language=r.payload.get("language", ""),
                            chunk_type=r.payload.get("chunk_type", "dependency"),
                            name=r.payload.get("name", ""),
                            signature=r.payload.get("signature", ""),
                            body=r.payload.get("body", ""),
                            start_line=r.payload.get("start_line", 0),
                            end_line=r.payload.get("end_line", 0),
                            parent_class=r.payload.get("parent_class", None),
                            relevance_score=3.0 # baseline dep score
                        )
        except Exception as e:
            log_event("dep_fetch_error", f"Failed fetching expanded dependency chunks: {str(e)}")
            
    # 4. LLM Re-ranking to top-8
    final_chunks = await rerank_chunks(query, list(merged_chunks.values()))
    
    log_event("retrieval_finished", f"Retrieved final {len(final_chunks)} high-relevance code chunks.")
    return final_chunks
