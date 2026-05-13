from typing import List
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from langchain_openai import OpenAIEmbeddings
import hashlib

from config.settings import settings
from src.rag.chunker import CodeChunk

# We use LangChain's OpenAI Embeddings (assumes OPENAI_API_KEY is set)
# Alternatively, could use LiteLLM embedding
def get_embeddings():
    return OpenAIEmbeddings(model="text-embedding-3-small")

def get_qdrant() -> QdrantClient:
    return QdrantClient(path=settings.qdrant_path)

async def embed_and_store(collection_name: str, chunks: List[CodeChunk]):
    """Embed chunks and save to Qdrant local store."""
    client = get_qdrant()
    
    # Check if collection exists
    collections = client.get_collections().collections
    if not any(c.name == collection_name for c in collections):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
        )
        
    embedder = get_embeddings()
    texts = [f"File: {c.file_path}\n{c.body}" for c in chunks]
    
    # Batch embed (simplification for safety)
    try:
        vectors = embedder.embed_documents(texts)
    except Exception as e:
        # Fallback to empty vectors if embedding fails (e.g. no API key)
        vectors = [[0.0] * 1536 for _ in texts]
        
    points = []
    for i, chunk in enumerate(chunks):
        # generate a stable numeric ID
        point_id = int(hashlib.md5(chunk.chunk_id.encode()).hexdigest()[:15], 16)
        
        points.append(PointStruct(
            id=point_id,
            vector=vectors[i],
            payload={
                "file_path": chunk.file_path,
                "language": chunk.language,
                "name": chunk.name,
                "body": chunk.body
            }
        ))
        
    client.upsert(collection_name=collection_name, points=points)

async def retrieve(query: str, collection: str, repo_map: dict, top_k: int = 8) -> List[CodeChunk]:
    """Semantic search over the codebase."""
    client = get_qdrant()
    
    try:
        embedder = get_embeddings()
        query_vector = embedder.embed_query(query)
        
        results = client.search(
            collection_name=collection,
            query_vector=query_vector,
            limit=top_k
        )
        
        return [
            CodeChunk(
                chunk_id=str(r.id),
                file_path=r.payload.get("file_path", ""),
                language=r.payload.get("language", ""),
                chunk_type="retrieved",
                name=r.payload.get("name", ""),
                signature="",
                body=r.payload.get("body", ""),
                start_line=0,
                end_line=0,
                relevance_score=r.score
            ) for r in results
        ]
    except Exception:
        # If vector search fails, return empty context
        return []
