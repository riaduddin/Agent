# backend/app/services/chat/reference_service.py
from app.services import vertex_ai_service
from app.models.system_log_model import SystemLogModel
import time
import json
from app.utils.redis_client import get_redis_client

def determine_final_reference(
    query: str, 
    context_chunk_map: dict, 
    parent_metadata_map: dict, 
    session_id: str, 
    message_id: str,
    user_email: str = None
) -> list:
    """
    Determines the single most relevant document reference for a query,
    with Redis caching.
    """
    redis_client = get_redis_client()
    cache_key = f"ref_cache:{session_id}:{query}:{user_email}"

    try:
        cached_reference = redis_client.get(cache_key)
        if cached_reference:
            print(f"CACHE HIT: Found reference for query '{query}' in session '{session_id}'")
            return json.loads(cached_reference)
    except Exception as e:
        print(f"Redis cache get failed: {e}")

    print(f"CACHE MISS: Determining reference for query '{query}' in session '{session_id}'")
    start_time = time.time()
    
    chunk_id_with_data = list(context_chunk_map.items())
    
    chunk_id_with_reference = vertex_ai_service.get_most_relevant_chunk_id(chunk_id_with_data, query)
    print(f"PERF_LOG: Step 8 (Get Most Relevant Chunk ID) took: {time.time() - start_time:.2f} seconds")

    if not chunk_id_with_reference:
        return []

    try:
        chunk_id_with_reference = chunk_id_with_reference.replace("[", "").replace("]", "")
        print(f"STREAM_RESPONSE: Most relevant chunk ID for query '{query}': {chunk_id_with_reference}")
        
        chunk_data = context_chunk_map.get(chunk_id_with_reference, {})
        original_doc_id = chunk_data.get("original_doc_firestore_id")
        
        if not original_doc_id:
            return []
            
        parent_metadata = parent_metadata_map.get(original_doc_id, {})
        
        final_doc_reference = {
            "filename": parent_metadata.get("original_filename", "Filename Unavailable"),
            "doc_id": original_doc_id,
            "start_page": chunk_data.get("start_page"), 
            "end_page": chunk_data.get("end_page")
        }
        
        final_reference_list = [final_doc_reference]

        try:
            redis_client.set(cache_key, json.dumps(final_reference_list), ex=86400) # Cache for 1 day
            print(f"CACHE SET: Stored reference for query '{query}' in session '{session_id}'")
        except Exception as e:
            print(f"Redis cache set failed: {e}")

        return final_reference_list

    except Exception as e:
        print(f"Error processing Chunk ID, Error: {e}")
        return []
