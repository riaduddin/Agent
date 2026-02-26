# backend/app/services/chat/embedding_service.py
from app.services import vertex_ai_service
from app.models.system_log_model import SystemLogModel
from app import config
import time
import json
import re
from app.utils.redis_client import get_redis_client
from app.utils.debug_logger import debug_log, debug_perf


def normalize_query(query: str) -> str:
    """
    Normalize query for better cache hit rate.
    - Lowercase
    - Strip whitespace
    - Collapse multiple spaces
    - Remove punctuation variations
    """
    normalized = query.lower().strip()
    normalized = re.sub(r'\s+', ' ', normalized)  # Collapse multiple spaces
    return normalized


def generate_query_embedding(query: str, session_id: str, message_id: str) -> list:
    """
    Generates an embedding for the user's query and logs the process.
    Uses normalized query for cache key to improve hit rate.
    """
    redis_client = get_redis_client()
    
    # Normalize query for better cache hits
    normalized_query = normalize_query(query)
    cache_key = f"embedding:{normalized_query}"

    debug_log(f"Generating embedding for query: {query}")
    debug_log(f"Normalized query for cache: {normalized_query}")
    
    try:
        cached_embedding = redis_client.get(cache_key)
        if cached_embedding:
            debug_log(f"CACHE HIT: Found embedding for query '{normalized_query}'")
            return json.loads(cached_embedding)
    except Exception as e:
        debug_log(f"Redis cache get failed for embedding: {e}")

    debug_log(f"CACHE MISS: Generating embedding for query '{normalized_query}'")
    start_time = time.time()
    
    SystemLogModel.add_log_entry(SystemLogModel(
        session_id=session_id, message_id=message_id, user_query_text=query,
        step_name="QUERY_EMBEDDING_START", status="INFO"
    ).to_dict())
    
    query_embedding = None
    try:
        query_embedding = vertex_ai_service.get_text_embedding(query)
        debug_perf("Step 1 (Query Embedding)", time.time() - start_time)
        
        if not query_embedding:
            raise ValueError("Failed to generate query embedding (no result)")

        # Use configurable TTL for embedding cache (default 7 days)
        embedding_ttl = getattr(config, 'EMBEDDING_CACHE_TTL_SECONDS', 604800)
        try:
            redis_client.set(cache_key, json.dumps(query_embedding), ex=embedding_ttl)
            debug_log(f"CACHE SET: Stored embedding for query '{normalized_query}' (TTL: {embedding_ttl}s)")
        except Exception as e:
            debug_log(f"Redis cache set failed for embedding: {e}")

        SystemLogModel.add_log_entry(SystemLogModel(
            session_id=session_id, message_id=message_id, user_query_text=query,
            step_name="QUERY_EMBEDDING_SUCCESS", status="SUCCESS",
            step_details={"embedding_dimensions": len(query_embedding)} 
        ).to_dict())
        
        return query_embedding

    except Exception as e_embed:
        SystemLogModel.add_log_entry(SystemLogModel(
            session_id=session_id, message_id=message_id, user_query_text=query,
            step_name="QUERY_EMBEDDING_FAILED_EXCEPTION_TERMINATING", status="ERROR", 
            error_message=f"Exception during embedding: {str(e_embed)}",
            step_details={"exception_type": type(e_embed).__name__}
        ).to_dict())
        # Re-raise the exception to be handled by the main route
        raise
