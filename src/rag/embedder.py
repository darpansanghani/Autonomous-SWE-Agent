import hashlib
import litellm
from typing import List
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

from config.settings import settings
from src.rag.chunker import CodeChunk, build_embedding_text
from src.utils.logger import log_event

# Constants
BATCH_SIZE = 100
MAX_TOKENS_PER_CHUNK = 512

def get_qdrant() -> QdrantClient:
    """Connect to local path-based Qdrant client."""
    return QdrantClient(path=settings.qdrant_path)

async def embed_chunks(chunks: List[CodeChunk], repo_map: dict) -> List[CodeChunk]:
    """
    Generate batch embeddings using primary text-embedding-3-small (via litellm/OpenAI)
    with Google text-embedding-004 fallback.
    """
    if not chunks:
        return []

    # Map file path to its imports list from repo_map for context enrichment
    enriched_texts = []
    for c in chunks:
        file_info = repo_map.get("structure", {}).get(c.file_path, {})
        imports = file_info.get("imports", [])
        enriched_texts.append(build_embedding_text(c, imports))

    # Primary model is settings.openai_api_key or fallback
    model = "openai/text-embedding-3-small"
    if not settings.openai_api_key and settings.google_api_key:
        model = "gemini/text-embedding-004"
        litellm.api_key = settings.google_api_key
    elif settings.openai_api_key:
        litellm.api_key = settings.openai_api_key

    # Batch process in sizes of BATCH_SIZE
    all_embeddings = []
    for i in range(0, len(enriched_texts), BATCH_SIZE):
        batch = enriched_texts[i:i + BATCH_SIZE]
        try:
            response = await litellm.aembedding(
                model=model,
                input=batch
            )
            for emb in response.data:
                all_embeddings.append(emb["embedding"])
        except Exception as e:
            # Safe senior developer fallback: return a default zero vector so the agent doesn't crash
            log_event("embedding_error", f"Embedding failed using {model}. Using fallback zero vectors.", {"error": str(e)})
            dimensions = 768 if "gemini" in model else 1536
            all_embeddings.extend([[0.0] * dimensions for _ in batch])

    # Assign embeddings to chunks
    for idx, chunk in enumerate(chunks):
        chunk.embedding = all_embeddings[idx]

    return chunks

async def store_in_qdrant(collection_name: str, chunks: List[CodeChunk]):
    """Upsert list of embedded code chunks directly into local Qdrant collection."""
    if not chunks:
        return

    client = get_qdrant()
    
    # Establish correct dimensions based on embedding vector size
    dimensions = len(chunks[0].embedding) if chunks[0].embedding else 1536
    
    collections = client.get_collections().collections
    if not any(c.name == collection_name for c in collections):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=dimensions, distance=Distance.COSINE)
        )

    points = []
    for chunk in chunks:
        # Generate clean stable 64-bit int ID
        point_id = int(hashlib.md5(chunk.chunk_id.encode()).hexdigest()[:15], 16)
        
        points.append(PointStruct(
            id=point_id,
            vector=chunk.embedding,
            payload={
                "file_path": chunk.file_path,
                "language": chunk.language,
                "chunk_type": chunk.chunk_type,
                "name": chunk.name,
                "signature": chunk.signature,
                "body": chunk.body,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "parent_class": chunk.parent_class
            }
        ))

    client.upsert(collection_name=collection_name, points=points)
