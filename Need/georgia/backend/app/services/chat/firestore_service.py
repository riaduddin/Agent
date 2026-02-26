# backend/app/services/chat/firestore_service.py
from app.models.metadata_model import get_chunks_by_ids, get_session_messages, get_chunks_by_ids_by_email
from app.models.system_log_model import SystemLogModel
from app import db, config
from app.utils.redis_client import get_redis_client
from app.utils.debug_logger import debug_log, debug_perf
import time
import logging
import json
import hashlib

logger = logging.getLogger(__name__)

def fetch_chunk_data(
    chunk_ids: list,
    query: str,
    session_id: str,
    message_id: str,
    user_email: str = None,
    user_role: str = None,
    auth_header: str = None
) -> dict:
    """
    Fetches the content of relevant chunks from Firestore, with Redis caching.
    """
    start_time = time.time()
    redis_client = get_redis_client()

    # Create a stable cache key from the sorted list of chunk IDs
    sorted_ids = sorted(chunk_ids)
    ids_string = ",".join(sorted_ids)
    cache_key = f"firestore_chunks:${user_email}:{hashlib.sha256(ids_string.encode()).hexdigest()}"

    # 1. Check cache first
    try:
        cached_chunks = redis_client.get(cache_key)
        if cached_chunks:
            SystemLogModel.add_log_entry(SystemLogModel(
                session_id=session_id, message_id=message_id, user_query_text=query,
                step_name="FIRESTORE_CHUNK_CACHE_HIT", status="INFO",
                step_details={"cache_key": cache_key, "user_email": user_email, "user_role": user_role}
            ).to_dict())
            debug_perf(f"Step 3 (Firestore Chunk Fetch from CACHE) took: {time.time() - start_time:.2f} seconds")
            return json.loads(cached_chunks)
    except Exception as e_redis_get:
        SystemLogModel.add_log_entry(SystemLogModel(
            session_id=session_id, message_id=message_id, user_query_text=query,
            step_name="FIRESTORE_CHUNK_CACHE_READ_FAILED", status="WARNING",
            error_message=f"Failed to read from Redis cache for chunks: {str(e_redis_get)}",
            step_details={"cache_key": cache_key, "user_email": user_email, "user_role": user_role}
        ).to_dict())

    SystemLogModel.add_log_entry(SystemLogModel(
        session_id=session_id, message_id=message_id, user_query_text=query,
        step_name="FIRESTORE_CHUNK_CACHE_MISS_START", status="INFO",
        step_details={
            "chunk_ids_to_fetch_count": len(chunk_ids),
            "chunk_ids_preview": chunk_ids[:3],
            "user_email": user_email,
            "user_role": user_role
        }
    ).to_dict())


    # If User Role is Null

    try:
        if user_role in ["superadmin"]:
            context_chunk_map = get_chunks_by_ids(chunk_ids)
        else:
            context_chunk_map = get_chunks_by_ids_by_email(chunk_ids, user_email=user_email, auth_header=auth_header)

        debug_perf(f"Step 3 (Firestore Chunk Fetch from DB) took: {time.time() - start_time:.2f} seconds")

        # 3. Store successful result in cache
        if context_chunk_map:
            try:
                redis_client.setex(
                    cache_key,
                    config.FIRESTORE_CHUNK_CACHE_TTL_SECONDS,
                    json.dumps(context_chunk_map)
                )
                SystemLogModel.add_log_entry(SystemLogModel(
                    session_id=session_id, message_id=message_id, user_query_text=query,
                    step_name="FIRESTORE_CHUNK_CACHE_WRITE_SUCCESS", status="INFO",
                    step_details={
                        "cache_key": cache_key,
                        "ttl": config.FIRESTORE_CHUNK_CACHE_TTL_SECONDS,
                        "user_email": user_email,
                        "user_role": user_role
                    }
                ).to_dict())
            except Exception as e_redis_set:
                SystemLogModel.add_log_entry(SystemLogModel(
                    session_id=session_id, message_id=message_id, user_query_text=query,
                    step_name="FIRESTORE_CHUNK_CACHE_WRITE_FAILED", status="WARNING",
                    error_message=f"Failed to write chunk data to Redis cache: {str(e_redis_set)}",
                    step_details={"cache_key": cache_key, "user_email": user_email, "user_role": user_role}
                ).to_dict())
        
        return context_chunk_map
        
    except Exception as e_firestore_fetch:
        SystemLogModel.add_log_entry(SystemLogModel(
            session_id=session_id, message_id=message_id, user_query_text=query,
            step_name="FIRESTORE_CHUNK_FETCH_FAILED_EXCEPTION", status="ERROR",
            error_message=f"Exception during Firestore chunk fetch: {str(e_firestore_fetch)}",
            step_details={
                "chunk_ids_queried": chunk_ids,
                "exception_type": type(e_firestore_fetch).__name__,
                "user_email": user_email,
                "user_role": user_role
            }
        ).to_dict())
        # Return empty map but don't raise, allow route to handle no context
        return {}

def fetch_parent_document_metadata(context_chunk_map: dict, query: str, session_id: str, message_id: str, user_email: str = None) -> dict:
    """
    Fetches metadata for the parent documents of the given chunks.
    """

    # Debug information
    debug_log(f"Fetching parent document metadata for {user_email} {len(context_chunk_map)} chunks at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

    start_time = time.time()
    parent_doc_ids = list(set(
        chunk_data.get("original_doc_firestore_id")
        for chunk_data in context_chunk_map.values() if chunk_data.get("original_doc_firestore_id")
    ))
    
    parent_metadata_map = {}
    if not parent_doc_ids:
        return {}

    try:
        parent_doc_id_batches = [parent_doc_ids[i:i + 30] for i in range(0, len(parent_doc_ids), 30)]
        for batch_ids in parent_doc_id_batches:
            if not batch_ids: continue
            parent_docs_query = db.collection("document_metadata").where("__name__", 'in', batch_ids)
            parent_docs_snaps = parent_docs_query.stream()
            for snap in parent_docs_snaps:
                if snap.exists:
                    parent_metadata_map[snap.id] = snap.to_dict()
                else:
                    logger.warning(f"Parent document metadata not found for ID: {snap.id} during batch fetch.")
        debug_perf(f"Step 4 (Parent Metadata Fetch) took: {time.time() - start_time:.2f} seconds")
        return parent_metadata_map
    except Exception as e:
        logger.error(f"Error batch fetching parent document metadata: {e}", exc_info=True)
        # Return what we have, or an empty dict
        return parent_metadata_map

def fetch_chat_history(session_id: str, current_user_email: str, query: str, message_id: str) -> list:
    """
    Fetches and formats the chat history for a given session.
    """
    start_time = time.time()
    SystemLogModel.add_log_entry(SystemLogModel(
        session_id=session_id, message_id=message_id, user_query_text=query,
        step_name="CHAT_HISTORY_FETCH_START", status="INFO",
        step_details={"session_id_for_history": session_id}
    ).to_dict())

    chat_history_formatted = []
    if session_id:
        history_messages, history_error = get_session_messages(session_id, current_user_email)
        debug_perf(f"Step 5 (Chat History Fetch) took: {time.time() - start_time:.2f} seconds")
        if history_error:
            logger.warning(f"Could not fetch chat history for session {session_id}: {history_error}")
        elif history_messages:
            for msg in history_messages: 
                role = "user" if msg.get("sender") == "user" else "model"
                text_content = msg.get("text")
                if text_content:
                    chat_history_formatted.append({'role': role, 'parts': [{'text': text_content}]})
                else:
                    logger.warning(f"Skipping history message with empty text: {msg}")
            logger.info(f"Formatted {len(chat_history_formatted)} messages from history for session {session_id}")

    SystemLogModel.add_log_entry(SystemLogModel(
        session_id=session_id, message_id=message_id, user_query_text=query,
        step_name="CHAT_HISTORY_FETCH_SUCCESS", status="SUCCESS",
        step_details={"formatted_history_message_count": len(chat_history_formatted)}
    ).to_dict())
    
    return chat_history_formatted


def save_system_log(session_id: str, message_id: str, step_name: str, status: str, details: dict = None, error_message: str = None, user_query_text: str = ""):
    """
    Saves a system log entry. Wrapper for SystemLogModel.
    """
    try:
        log_entry = SystemLogModel(
            session_id=session_id,
            message_id=message_id,
            user_query_text=user_query_text,
            step_name=step_name,
            status=status,
            step_details=details,
            error_message=error_message
        )
        SystemLogModel.add_log_entry(log_entry.to_dict())
    except Exception as e:
        logger.error(f"Failed to save system log: {e}")

