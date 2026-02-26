# backend/app/routes/secure_document_routes.py
import logging
import json
import requests
import re
import threading
import queue
import time
from flask import Blueprint, jsonify, request, Response
from flask_jwt_extended import jwt_required, get_jwt_identity, get_jwt
from app import db
from app.models import metadata_model
from app.services.chat import vector_search_service, firestore_service, embedding_service
from app.services import vertex_ai_service, query_entity_extractor, entity_query_generator, gcs_service
from app.services.chat import multi_query_search_service
from app.routes.document_routes import stream_response, stream_no_grounded_answer_response
from app import config
import uuid
from app.utils.debug_logger import debug_log, debug_error
from app.utils.redis_client import get_redis_client
from app.llm.gemini_api_key_client import GeminiConfigurationError

logger = logging.getLogger(__name__)
secure_doc_bp = Blueprint('secure_doc_bp', __name__)

def get_user_accessible_doc_ids(user_email: str) -> list:
    """
    Fetches the list of document IDs a user is allowed to access based on their categories.
    """
    # Query users by email field instead of using email as document ID
    user_query = db.collection("users").where("email", "==", user_email).limit(1).get()
    if not user_query:
        debug_log(f"No user found with email: {user_email}")
        return []
    
    user_doc = user_query[0]
    user_data = user_doc.to_dict()
    accessible_categories = user_data.get("accessible_categories", [])
    debug_log(f"User {user_email} has accessible_categories: {accessible_categories}")
    
    if not accessible_categories:
        return []

    # This could be slow for very large datasets. Consider optimization if needed.
    allowed_docs_query = db.collection("document_metadata").where("categories", "array_contains_any", accessible_categories).stream()
    allowed_doc_ids = [doc.id for doc in allowed_docs_query]
    debug_log(f"Found {len(allowed_doc_ids)} documents matching user categories")
    debug_log(f"Document IDs: {allowed_doc_ids[:5]}{'...' if len(allowed_doc_ids) > 5 else ''}")
    return allowed_doc_ids


def get_user_accessible_doc_ids_from_api() -> list:
    """
    Fetches the list of document IDs a user is allowed to access from an external API,
    with Redis caching.
    
    Caching Layers:
    1. categories (30 min): Fetched from external API.
    2. allowed_doc_ids (5 min): Resolved from Firestore based on categories.
    
    Returns an empty list if the API call fails or if no auth header is present.
    """
    try:
        # Get the Authorization header from the current request
        auth_header = request.headers.get('Authorization', '')
        if not auth_header:
            logger.warning("No Authorization header found in request for accessible docs API")
            return []
        
        # Get current user email for cache keys
        current_user_email = get_jwt_identity()
        if not current_user_email:
             current_user_email = "unknown_user"

        id_cache_key = f"user_allowed_doc_ids:{current_user_email}"
        category_cache_key = f"user_categories_api:{current_user_email}"
        
        redis_client = get_redis_client()
        
        # LAYER 2 CACHE: Try to get the fully resolved document IDs first
        if redis_client:
            try:
                cached_ids = redis_client.get(id_cache_key)
                if cached_ids:
                    allowed_doc_ids = json.loads(cached_ids)
                    debug_log(f"🚀 REDIS HIT (Layer 2): Found {len(allowed_doc_ids)} cached allowed_doc_ids for {current_user_email}")
                    return allowed_doc_ids
            except Exception as e:
                debug_error(f"Redis Layer 2 cache lookup failed: {e}")

        # LAYER 1 CACHE: Try to get categories if IDs weren't cached
        accessible_categories = None
        if redis_client:
            try:
                cached_categories = redis_client.get(category_cache_key)
                if cached_categories:
                    accessible_categories = json.loads(cached_categories)
                    debug_log(f"✅ REDIS HIT (Layer 1): Found cached categories for {current_user_email}: {accessible_categories}")
            except Exception as e:
                debug_error(f"Redis Layer 1 cache lookup failed: {e}")

        # API FALLBACK: If categories aren't cached, call external API
        if accessible_categories is None:
            # Get the API URL from config
            api_url = getattr(config, 'ACCESSIBLE_DOCS_API_URL', '')
            if not api_url:
                logger.warning("ACCESSIBLE_DOCS_API_URL not configured")
                return []
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": auth_header
            }
            
            logger.info(f"Fetching accessible categories from API: {api_url}")
            debug_log(f"📡 Calling external API for categories: {api_url}")
            start_time = time.time()
            response = requests.get(api_url, headers=headers, timeout=10)
            response.raise_for_status()
            debug_log(f"API call took {time.time() - start_time:.3f}s")
            
            data = response.json()
            # API response format: {"categories": [{"code": "...", ...}, ...]}
            categories_data = data.get('categories', [])
            accessible_categories = [cat.get('code') for cat in categories_data if cat.get('code')]
            
            # Cache newly fetched categories (30 minutes)
            if redis_client and accessible_categories:
                try:
                    redis_client.setex(category_cache_key, 1800, json.dumps(accessible_categories))
                    debug_log(f"💾 Cached categories in Redis for 30 minutes")
                except Exception as e:
                    debug_error(f"Failed to cache categories: {e}")

        if not accessible_categories:
            return []
        
        # RESOLVE IDS: Query Firestore for documents matching categories
        debug_log(f"🔍 Resolving document IDs from Firestore for categories: {accessible_categories}")
        allowed_docs_query = db.collection("document_metadata")\
            .where("categories", "array_contains_any", accessible_categories)\
            .select([])\
            .stream()
        allowed_doc_ids = [doc.id for doc in allowed_docs_query]
        
        # CACHE RESOLVED IDS (5 minutes)
        if redis_client:
            try:
                redis_client.setex(id_cache_key, 300, json.dumps(allowed_doc_ids))
                debug_log(f"💾 Cached {len(allowed_doc_ids)}cument_routes.py allowed_doc_ids in Redis for 5 minutes")
            except Exception as e:
                debug_error(f"Failed to cache allowed_doc_ids: {e}")

        logger.info(f"Successfully fetched {len(allowed_doc_ids)} accessible document entries (Categories: {accessible_categories})")
        return allowed_doc_ids

        
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch accessible categories from API: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error fetching accessible document IDs from API: {e}", exc_info=True)
        return []


def add_download_urls_to_history(history: list, expiration_minutes: float = 5.0) -> list:
    """
    Adds download URLs to each history item.
    
    Args:
        history: List of document metadata items
        expiration_minutes: URL expiration time in minutes (default 5 minutes)
    
    Returns:
        Updated history list with 'download_url' field added to each item
    """
    for item in history:
        try:
            blob_name = item.get('gcs_blob_name')
            if blob_name:
                # Generate signed URL with custom expiration
                download_url = gcs_service.generate_download_signed_url(
                    blob_name=blob_name,
                    expiration_minutes=expiration_minutes
                )
                item['download_url'] = download_url if download_url else None
                
                if not download_url:
                    debug_log(f"Failed to generate download URL for blob: {blob_name}")
                else:
                    debug_log(f"Generated download URL for {item.get('original_filename', 'N/A')}")
            else:
                debug_log(f"No gcs_blob_name found for document {item.get('id', 'unknown')}")
                item['download_url'] = None
        except Exception as e:
            debug_error(f"Error generating download URL for item {item.get('id', 'unknown')}: {e}")
            item['download_url'] = None
    
    return history


@secure_doc_bp.route('/docs/history', methods=['GET'])
@jwt_required()
def get_secure_history():
    """
    Retrieves paginated document history, filtered by user's accessible categories.
    Admins and superadmins can see all files.
    """
    current_user_email = get_jwt_identity()
    claims = get_jwt()
    user_role = claims.get("role")
    debug_log(f"USER ROLE ======= : {user_role} at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    debug_log(f"USER MAIL ======= : {current_user_email} at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    debug_log(f"CLAIMS ======= : {claims} at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

    limit = request.args.get('limit', 10, type=int)
    start_after_doc_id = request.args.get('start_after', None, type=str)
    search_term = request.args.get('search', None, type=str)
    frontend_status_filter = request.args.get('status', None, type=str)


    filter_to_granular_map = {
        "Pending": ["Pending", "pending", "queued_for_splitting"],
        "Processing": [
            "processing", "splitting", "splitting_in_progress", 
            "pending_chunk_processing", "ocr_pending", "ocr_in_progress", 
            "pending_vectorization", "vectorizing"
        ],
        "Failed": [
            "error", "error_splitting", "error_creating_chunks", "incomplete", 
            "error_worker_failure", "ocr_failed", "embedding_failed", 
            "vectorization_failed", "upload_failed", "processing_error", "unknown"
        ],
        "Completed": ["completed"]
    }

    backend_status_query_value = frontend_status_filter 
    if frontend_status_filter and frontend_status_filter != "All Statuses":
        standardized_filter_key = frontend_status_filter.capitalize() if frontend_status_filter else None
        
        granular_statuses_for_filter = filter_to_granular_map.get(standardized_filter_key)
        
        if granular_statuses_for_filter:
            backend_status_query_value = granular_statuses_for_filter
        else:
            logger.warning(f"Unknown status filter category received: {frontend_status_filter}. No status filter will be applied.")
            backend_status_query_value = frontend_status_filter



    allowed_doc_ids = None
    if user_role not in ["superadmin"]:
        # allowed_doc_ids = get_user_accessible_doc_ids(current_user_email)
        allowed_doc_ids = get_user_accessible_doc_ids_from_api()
        if not allowed_doc_ids:
            return jsonify({"history": [], "next_cursor": None, "total_items": 0}), 200

    # Debug PRINT
    debug_log("=======================================================")
    debug_log("=======================================================")
    debug_log(f"ALLOWED DOC IDS ======= : {allowed_doc_ids} at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    debug_log("=======================================================")
    debug_log("=======================================================")




    history, last_doc_id, total_items, error = metadata_model.get_upload_history(
        limit=limit,
        start_after_doc_id=start_after_doc_id,
        search_term=search_term,
        status_filter=backend_status_query_value,
        allowed_doc_ids=allowed_doc_ids
    )

    if error:
        return jsonify({"msg": error}), 500

    # Add download URLs to each history item
    history = add_download_urls_to_history(history, expiration_minutes=5.0)

    return jsonify({
        "history": history,
        "next_cursor": last_doc_id,
        "total_items": total_items
    }), 200


@secure_doc_bp.route('/docs/chat', methods=['POST'])
@jwt_required()
def secure_chat_with_documents():
    """
    Handles user chat queries, ensuring vector search is restricted
    to documents the user has permission to access. Admins and superadmins can access all documents.
    """
    start_time = time.time()
    current_user_email = get_jwt_identity()
    claims = get_jwt()
    user_role = claims.get("role")
    data = request.get_json()
    query = data.get('query')
    session_id = data.get('session_id', None)
    message_id = uuid.uuid4().hex


    # Print debug information role and user email
    debug_log(f"USER ROLE ======= : {user_role} at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    debug_log(f"USER MAIL ======= : {current_user_email} at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

    if not query:
        return jsonify({"msg": "Query is required"}), 400
    
    # Extract Authorization header while we still have request context
    auth_header = request.headers.get('Authorization', '')

    def generator():
        try:

            # 1. Generate Query Embedding
            yield f"data: {json.dumps({'status': 'Understanding your question...'})}\n\n"
            query_embedding = embedding_service.generate_query_embedding(query, session_id, message_id)

            # 1.5 Extract Entities and Metadata for Filtering
            yield f"data: {json.dumps({'status': 'Analyzing your request...'})}\n\n"
            
            # NEW: Extract searchable entities dynamically
            entities = query_entity_extractor.extract_searchable_entities(query)
            if entities:
                debug_log(f"Extracted entities from query:")
                debug_log(f"  - Identifiers: {entities.get('identifiers', [])}")
                debug_log(f"  - Names: {entities.get('names', [])}")
                debug_log(f"  - Keywords: {entities.get('keywords', [])}")
                
                # Save entity extraction to system logs for debugging
                firestore_service.save_system_log(
                    session_id=session_id,
                    message_id=message_id,
                    step_name="ENTITY_EXTRACTION",
                    status="SUCCESS",
                    details={
                        "identifiers": entities.get('identifiers', []),
                        "names": entities.get('names', []),
                        "dates": entities.get('dates', {}),
                        "amounts": entities.get('amounts', []),
                        "keywords": entities.get('keywords', []),
                        "entity_count": len(entities.get('entity_details', {}))
                    }
                )
            
            # Generate entity-focused queries if multi-query search is enabled
            entity_queries = []
            if entities and config.MULTI_QUERY_SEARCH_ENABLED:
                entity_queries = entity_query_generator.generate_entity_queries(
                    entities,
                    max_queries=config.ENTITY_QUERY_MAX_COUNT
                )
                debug_log(f"Generated {len(entity_queries)} entity queries: {entity_queries}")
                
                # Save entity query generation to system logs for debugging
                firestore_service.save_system_log(
                    session_id=session_id,
                    message_id=message_id,
                    step_name="ENTITY_QUERY_GENERATION",
                    status="SUCCESS",
                    details={
                        "entity_query_count": len(entity_queries),
                        "queries": entity_queries
                    }
                )
            
            # Also extract metadata filters for backward compatibility
            metadata_filters = vertex_ai_service.extract_query_metadata(query)
            if metadata_filters:
                debug_log(f"Applying metadata filters from query: {metadata_filters}")
            
            # Log metadata filter extraction
            firestore_service.save_system_log(
                session_id=session_id,
                message_id=message_id,
                step_name="METADATA_FILTER_EXTRACTION",
                status="SUCCESS" if metadata_filters else "INFO",
                details={
                    "filters_found": bool(metadata_filters),
                    "filters": metadata_filters or {}
                }
            )

            # 2. Find Relevant Chunks (with permission filter and metadata filter)
            yield f"data: {json.dumps({'status': 'Looking through your documents...'})}\n\n"
            
            # Log search strategy decision
            search_strategy = "MULTI_QUERY" if (config.MULTI_QUERY_SEARCH_ENABLED and entity_queries) else "TRADITIONAL"
            firestore_service.save_system_log(
                session_id=session_id,
                message_id=message_id,
                step_name="SEARCH_STRATEGY_DECISION",
                status="INFO",
                details={
                    "strategy": search_strategy,
                    "multi_query_enabled": config.MULTI_QUERY_SEARCH_ENABLED,
                    "entity_queries_count": len(entity_queries) if entity_queries else 0,
                    "has_metadata_filters": bool(metadata_filters)
                }
            )
            
            # Create a queue for status messages from vector search fallback
            status_queue = queue.Queue()
            search_result = [None]  # Use list to store result from thread
            search_error = [None]   # Use list to store any errors
            
            def status_callback(message):
                """Callback to send status updates during vector search fallback attempts"""
                status_queue.put(message)
            
            # Run vector search in a separate thread so we can stream status messages
            def search_thread():
                try:
                    # Determine which search mode to use
                    allowed_doc_ids_for_search = None  # Will be determined by search service
                    
                    if config.MULTI_QUERY_SEARCH_ENABLED and entity_queries:
                        # NEW: Multi-query entity-based search
                        debug_log(f"Using multi-query search with {len(entity_queries)} entity queries")
                        results, stats = multi_query_search_service.multi_query_search(
                            query=query,
                            query_embedding=query_embedding,
                            entities=entities,
                            allowed_doc_ids=allowed_doc_ids_for_search,
                            metadata_filters=metadata_filters,
                            entity_queries=entity_queries
                        )
                        
                        # Save detailed multi-query stats to system log
                        # NEW: Log Phase 0 Deterministic Results separately for clarity
                        # Always log Phase 0 Deterministic Results for visibility
                        det_matches = stats.get("deterministic_matches", 0)
                        firestore_service.save_system_log(
                            session_id=session_id,
                            message_id=message_id,
                            step_name="DETERMINISTIC_SEARCH_RESULTS",
                            status="SUCCESS" if det_matches > 0 else "INFO",
                            details={
                                "total_matches": det_matches,
                                "breakdown": stats.get("deterministic_search_details", [])
                            }
                        )

                        firestore_service.save_system_log(
                            session_id=session_id,
                            message_id=message_id,
                            step_name="MULTI_QUERY_SEARCH_RESULTS",
                            status="SUCCESS",
                            details=stats
                        )

                        if results:
                            search_result[0] = results
                        else:
                            # TRIGGER FALLBACK (Path A): If multi-query found absolutely 0 vectors
                            debug_log("Multi-query returned 0 results. Triggering traditional fallback with multipliers.")
                            
                            firestore_service.save_system_log(
                                session_id=session_id,
                                message_id=message_id,
                                step_name="MULTI_QUERY_EMPTY_TRIGGER_FALLBACK",
                                status="WARNING",
                                details={"reason": "Multi-query search returned 0 results"}
                            )

                            if status_callback:
                                status_callback("Refining search strategy...")
                                
                            search_result[0] = vector_search_service.find_relevant_chunks(
                                query_embedding, query, session_id, message_id,
                                user_email=current_user_email,
                                allowed_doc_ids=allowed_doc_ids_for_search, 
                                metadata_filters=metadata_filters,
                                status_callback=status_callback
                            )
                            # Mark that we already used fallback
                            search_result[1] = True 
                    else:
                        # FALLBACK: Existing traditional vector search
                        debug_log("Using traditional vector search (multi-query disabled or no entity queries)")
                        search_result[0] = vector_search_service.find_relevant_chunks(
                            query_embedding, query, session_id, message_id,
                            user_email=current_user_email,
                            allowed_doc_ids=allowed_doc_ids_for_search, 
                            metadata_filters=metadata_filters,
                            status_callback=status_callback
                        )
                except Exception as e:
                    search_error[0] = e
                finally:
                    # Signal that search is complete
                    status_queue.put(None)
            
            # Start the search in a background thread
            # search_result[1] indicates if fallback was already triggered in thread
            search_result.append(False) 
            thread = threading.Thread(target=search_thread, daemon=True)
            thread.start()
            
            # Stream status messages as they arrive from the search thread
            while True:
                try:
                    # Wait for messages with a timeout to avoid blocking forever
                    message = status_queue.get(timeout=0.5)
                    
                    # None signals the search is complete
                    if message is None:
                        break
                    
                    # Yield status message to client
                    yield f"data: {json.dumps({'status': message})}\n\n"
                except queue.Empty:
                    # No message yet, check if thread is still alive
                    if not thread.is_alive():
                        # Thread finished but might have one final message
                        try:
                            message = status_queue.get_nowait()
                            if message is not None:
                                yield f"data: {json.dumps({'status': message})}\n\n"
                        except queue.Empty:
                            break
            
            # Check if there was an error in the search thread
            if search_error[0]:
                raise search_error[0]
            
            neighbors = search_result[0]
            fallback_already_occurred = len(search_result) > 1 and search_result[1]
            
            # Log search results summary
            firestore_service.save_system_log(
                session_id=session_id,
                message_id=message_id,
                step_name="VECTOR_SEARCH_COMPLETE",
                status="SUCCESS" if neighbors else "WARNING",
                details={
                    "neighbors_found": len(neighbors) if neighbors else 0,
                    "fallback_triggered_in_search": fallback_already_occurred,
                    "top_5_distances": [n.get('distance', 'N/A') for n in (neighbors or [])[:5]]
                }
            )
            
            # 3. Fetch Data from Firestore
            yield f"data: {json.dumps({'status': 'Gathering relevant information...'})}\n\n"
            
            def fetch_and_structure_context(neighbor_list):
                 """Helper to fetch data and create structured chunks"""
                 if not neighbor_list: return [], {}, {}
                 c_ids = [n['id'] for n in neighbor_list]
                 # Use the auth_header captured from the outer scope
                 c_map = firestore_service.fetch_chunk_data(c_ids, query, session_id, message_id, user_email=current_user_email, user_role=user_role, auth_header=auth_header)
                 p_map = firestore_service.fetch_parent_document_metadata(c_map, query, session_id, message_id, user_email=current_user_email)
                 
                 s_chunks = []
                 t_ids = c_ids[:config.LLM_CONTEXT_CHUNK_LIMIT]
                 for cid in t_ids:
                    if cid in c_map:
                        c_data = c_map[cid]
                        text = c_data.get("ocr_text_preview") or c_data.get("extracted_text_preview")
                        if text:
                            dist = next((n['distance'] for n in neighbor_list if n['id'] == cid), "N/A")
                            s_chunks.append({
                                'id': cid,
                                'original_doc_id': c_data.get('original_doc_firestore_id'),
                                'text': text,
                                'distance': dist,
                                'extracted_entities': c_data.get('extracted_entities', {}), # For backward compatibility
                                'entities': c_data.get('entities', []) # NEW: Flattened entities array
                            })
                            debug_log(f"   🏗️  Structured chunk {cid[:8]}... with {len(s_chunks[-1]['entities'])} entities")
                 return s_chunks, c_map, p_map

            structured_context_chunks, context_chunk_map, parent_metadata_map = fetch_and_structure_context(neighbors)
            
            # Log Firestore fetch results
            firestore_service.save_system_log(
                session_id=session_id,
                message_id=message_id,
                step_name="FIRESTORE_CHUNK_FETCH",
                status="SUCCESS" if structured_context_chunks else "WARNING",
                details={
                    "chunk_ids_requested": len(neighbors) if neighbors else 0,
                    "chunks_retrieved": len(context_chunk_map),
                    "structured_chunks_created": len(structured_context_chunks),
                    "parent_docs_found": len(parent_metadata_map),
                    "llm_context_limit": config.LLM_CONTEXT_CHUNK_LIMIT
                }
            )
            
            # POST-RETRIEVAL FALLBACK TRIGGER (Path B)
            # Scenario: Multi-Query found neighbors (vectors), but they were all filtered out 
            # (e.g., deleted files, permissions, or ghost vectors), resulting in 0 context chunks.
            # OR the system suspects the result is poor and user requested explicit fallback logic.
            
            should_run_fallback_path_b = (
                config.MULTI_QUERY_SEARCH_ENABLED 
                and entity_queries 
                and neighbors 
                and not fallback_already_occurred 
                and not structured_context_chunks 
            )
            
            if should_run_fallback_path_b:
                 debug_log("Multi-Query neighbors yielded ZERO structured chunks. Triggering Post-Retrieval Fallback.")
                 
                 firestore_service.save_system_log(
                    session_id=session_id,
                    message_id=message_id,
                    step_name="MULTI_QUERY_CONTEXT_EMPTY_FALLBACK",
                    status="WARNING",
                    details={"reason": "Neighbors found but 0 valid chunks retrieved"}
                 )
                 
                 yield f"data: {json.dumps({'status': 'Verifying information availability...'})}\n\n"
                 yield f"data: {json.dumps({'status': 'Refining search strategy...'})}\n\n"
                 
                 # Run Traditional Fallback Search (Blocking in Main Thread)
                 fallback_neighbors = vector_search_service.find_relevant_chunks(
                    query_embedding, query, session_id, message_id,
                    user_email=current_user_email,
                    allowed_doc_ids=None,
                    metadata_filters=metadata_filters,
                    status_callback=None # Cannot stream granular updates from here
                 )
                 
                 if fallback_neighbors:
                     debug_log(f"Fallback Search found {len(fallback_neighbors)} neighbors.")
                     # Re-fetch context
                     yield f"data: {json.dumps({'status': 'Gathering relevant information (Retry)...'})}\n\n"
                     structured_context_chunks, context_chunk_map, parent_metadata_map = fetch_and_structure_context(fallback_neighbors)
                     # Update main neighbors list for consistency checks if any
                     neighbors = fallback_neighbors
                     # Mark fallback as occurred to prevent infinite loops
                     fallback_already_occurred = True

            # ═══════════════════════════════════════════════════════════════════════════
            # PATH C: SMART CONTEXT RELEVANCE CHECK (Entity-Weighted Validation)
            # ═══════════════════════════════════════════════════════════════════════════
            # Problem: Vector similarity doesn't guarantee semantic relevance.
            # "check" matches millions of checks, but we need CHECK #12345 specifically.
            # Solution: Validate that CRITICAL identifiers from the query exist in chunks.
            
            def calculate_entity_relevance(entities_dict: dict, chunks: list) -> tuple:
                """
                Smart relevance calculation prioritizing identifiers over common terms.
                
                Priority:
                - IDENTIFIERS (50%): MANDATORY - check numbers, account IDs, etc.
                - NAMES (30%): HIGH - payee names, employee names
                - DATES/AMOUNTS (20%): MEDIUM - specific dates, dollar amounts
                - KEYWORDS (0%): IGNORED - "check", "report" (too common)
                
                Returns: (score, debug_info, should_fallback)
                """
                if not chunks or not entities_dict:
                    return 1.0, {"reason": "no_entities_to_validate"}, False
                
                # Combine all chunk text for searching
                combined_text = " ".join([
                    (chunk.get('text', '') or '').lower() 
                    for chunk in chunks
                ]).strip()
                
                if not combined_text:
                    return 0.0, {"reason": "no_chunk_text"}, True
                
                # ─── CRITICAL: Identifier Validation ───
                identifiers = entities_dict.get('identifiers', [])
                identifier_matches = 0
                identifier_details = []
                
                # Validate that CRITICAL identifiers from the query exist in chunks.
                for identifier in identifiers:
                    id_lower = str(identifier).lower().strip()
                    # CHECK 1: OCR Text Match
                    if id_lower and id_lower in combined_text:
                        identifier_matches += 1
                        identifier_details.append({"id": identifier, "found": True, "source": "text"})
                        continue # Found in text, move to next identifier

                    # CHECK 2: Metadata Match (Robust Fallback for Flat or Grouped structure)
                    found_in_metadata = False
                    for chunk in chunks:
                        chunk_entities = chunk.get('extracted_entities', {}) or {}
                        
                        # Gather ALL normalized values from this chunk's metadata
                        # This works for both flat (v2.4) and grouped (v2.1) metadata structures
                        meta_values = []
                        for k, v in chunk_entities.items():
                            if v is not None:
                                if isinstance(v, list):
                                    v_strings = [str(item).lower().strip() for item in v if item]
                                    meta_values.extend(v_strings)
                                else:
                                    meta_values.append(str(v).lower().strip())
                        
                        # Also check specifically for old grouped key 'identifiers' just in case
                        if 'identifiers' in chunk_entities and isinstance(chunk_entities['identifiers'], list):
                             meta_values.extend([str(v).lower().strip() for v in chunk_entities['identifiers']])
                        
                        # CHECK 3: Flattened Entities Array (NEW)
                        flat_entities = [str(e).lower().strip() for e in chunk.get('entities', [])]
                        
                        if id_lower in meta_values or id_lower in flat_entities:
                            found_in_metadata = True
                            debug_log(f"DEBUG: MATCH FOUND! Identifier '{identifier}' matched in metadata/entities of chunk {chunk.get('id', 'unknown')}")
                            break
                    
                    if found_in_metadata:
                        identifier_matches += 1
                        identifier_details.append({"id": identifier, "found": True, "source": "metadata"})
                    else:
                        identifier_details.append({"id": identifier, "found": False})
                
                # MANDATORY CHECK: If we have identifiers and NONE matched, immediate fallback
                if identifiers and identifier_matches == 0:
                    return 0.0, {
                        "reason": "CRITICAL_IDENTIFIER_MISSING",
                        "identifiers": identifier_details,
                        "message": f"Query contains {len(identifiers)} identifier(s) but NONE found in retrieved chunks"
                    }, True
                
                identifier_score = identifier_matches / len(identifiers) if identifiers else 1.0
                
                # ─── HIGH: Name Validation (fuzzy - match any part) ───
                names = entities_dict.get('names', [])
                name_matches = 0
                name_details = []
                
                for name in names:
                    name_lower = str(name).lower().strip()
                    # Split name into parts and check if ANY part exists
                    name_parts = name_lower.split()
                    matched_parts = [p for p in name_parts if len(p) > 2 and p in combined_text]
                    
                    if matched_parts:
                        name_matches += 1
                        name_details.append({"name": name, "found": True, "matched_parts": matched_parts})
                    else:
                        name_details.append({"name": name, "found": False})
                
                name_score = name_matches / len(names) if names else 1.0
                
                # ─── MEDIUM: Date/Amount Validation ───
                dates = entities_dict.get('dates', {})
                amounts = entities_dict.get('amounts', [])
                aux_matches = 0
                aux_total = 0
                aux_details = []
                
                # Check amounts (extract just numeric parts)
                for amount in amounts:
                    aux_total += 1
                    # Extract numeric value: "$1,500.00" -> "1500.00" or "1500"
                    amount_clean = re.sub(r'[^\d.]', '', str(amount))
                    if amount_clean and (amount_clean in combined_text or amount_clean.replace('.', '') in combined_text):
                        aux_matches += 1
                        aux_details.append({"type": "amount", "value": amount, "found": True})
                    else:
                        aux_details.append({"type": "amount", "value": amount, "found": False})
                
                # Check dates (check year, month values)
                for date_key, date_val in dates.items():
                    if date_val:
                        aux_total += 1
                        date_str = str(date_val).lower()
                        if date_str in combined_text:
                            aux_matches += 1
                            aux_details.append({"type": "date", "key": date_key, "value": date_val, "found": True})
                        else:
                            aux_details.append({"type": "date", "key": date_key, "value": date_val, "found": False})
                
                aux_score = aux_matches / aux_total if aux_total > 0 else 1.0
                
                # ─── WEIGHTED FINAL SCORE ───
                # Identifiers: 50%, Names: 30%, Dates/Amounts: 20%
                final_score = (identifier_score * 0.50) + (name_score * 0.30) + (aux_score * 0.20)
                
                # Determine if fallback is needed
                threshold = getattr(config, 'CONTEXT_RELEVANCE_THRESHOLD', 0.4)
                should_fallback = final_score < threshold
                
                debug_info = {
                    "final_score": round(final_score, 3),
                    "threshold": threshold,
                    "identifier_score": round(identifier_score, 3),
                    "name_score": round(name_score, 3),
                    "aux_score": round(aux_score, 3),
                    "identifiers": identifier_details,
                    "names": name_details,
                    "dates_amounts": aux_details
                }
                
                return final_score, debug_info, should_fallback
            
            # Execute Path C check ONLY if:
            # 1. We have chunks (not empty)
            # 2. We used multi-query search (has entities)
            # 3. Fallback hasn't already been triggered
            should_run_path_c = (
                structured_context_chunks 
                and entities 
                and not fallback_already_occurred
                and config.MULTI_QUERY_SEARCH_ENABLED
            )
            
            if should_run_path_c:
                relevance_score, relevance_debug, needs_fallback = calculate_entity_relevance(entities, structured_context_chunks)
                
                debug_log(f"Context Relevance Check: score={relevance_score:.3f}, needs_fallback={needs_fallback}")
                debug_log(f"Relevance Details: {relevance_debug}")
                
                # Log the relevance check result
                firestore_service.save_system_log(
                    session_id=session_id,
                    message_id=message_id,
                    step_name="CONTEXT_RELEVANCE_CHECK",
                    status="WARNING" if needs_fallback else "SUCCESS",
                    details=relevance_debug
                )
                
                # EXPLICIT LOGGING FOR METADATA MATCHES (User Debugging Requirement)
                meta_matches = [d['id'] for d in relevance_debug.get('identifiers', []) if d.get('source') == 'metadata']
                if meta_matches:
                     firestore_service.save_system_log(
                        session_id=session_id,
                        message_id=message_id,
                        step_name="CONTEXT_RELEVANCE_METADATA_MATCH", 
                        status="SUCCESS",
                        details={
                            "matched_identifiers": meta_matches,
                            "count": len(meta_matches),
                            "note": "Critical entities found via metadata fallback (OCR text missing)"
                        }
                     )
                
                if needs_fallback:
                    debug_log(f"Path C TRIGGERED: Relevance score {relevance_score:.3f} below threshold. Critical identifiers may be missing.")
                    
                    firestore_service.save_system_log(
                        session_id=session_id,
                        message_id=message_id,
                        step_name="CONTEXT_RELEVANCE_LOW_FALLBACK",
                        status="WARNING",
                        details={
                            "reason": relevance_debug.get("reason", "Low relevance score"),
                            "score": relevance_score,
                            "threshold": relevance_debug.get("threshold", 0.4)
                        }
                    )
                    
                    yield f"data: {json.dumps({'status': 'Validating search accuracy...'})}\n\n"
                    yield f"data: {json.dumps({'status': 'Expanding search for precise matches...'})}\n\n"
                    
                    # Run Traditional Fallback Search with Multipliers
                    fallback_neighbors = vector_search_service.find_relevant_chunks(
                        query_embedding, query, session_id, message_id,
                        user_email=current_user_email,
                        allowed_doc_ids=None,
                        metadata_filters=None,  # Remove filters for maximum recall
                        status_callback=None
                    )
                    
                    if fallback_neighbors:
                        debug_log(f"Path C Fallback found {len(fallback_neighbors)} neighbors.")
                        yield f"data: {json.dumps({'status': 'Found additional results, verifying...'})}\n\n"
                        
                        new_chunks, new_context_map, new_parent_map = fetch_and_structure_context(fallback_neighbors)
                        
                        # Re-check relevance of new results
                        new_score, new_debug, still_needs_fallback = calculate_entity_relevance(entities, new_chunks)
                        
                        debug_log(f"Path C Re-check: new_score={new_score:.3f}, original_score={relevance_score:.3f}")
                        
                        # Use new results if they're better
                        if new_score > relevance_score:
                            debug_log("Path C: New results have BETTER relevance. Switching.")
                            structured_context_chunks = new_chunks
                            context_chunk_map = new_context_map
                            parent_metadata_map = new_parent_map
                            neighbors = fallback_neighbors
                            
                            firestore_service.save_system_log(
                                session_id=session_id,
                                message_id=message_id,
                                step_name="CONTEXT_RELEVANCE_IMPROVED",
                                status="SUCCESS",
                                details={"old_score": relevance_score, "new_score": new_score}
                            )
                        else:
                            debug_log("Path C: Fallback results NOT better. Keeping original.")
                            firestore_service.save_system_log(
                                session_id=session_id,
                                message_id=message_id,
                                step_name="CONTEXT_RELEVANCE_FALLBACK_NO_IMPROVEMENT",
                                status="INFO",
                                details={"old_score": relevance_score, "new_score": new_score}
                            )
            
            # Validate final context
            if not structured_context_chunks:
                # Log final validation failure
                firestore_service.save_system_log(
                    session_id=session_id,
                    message_id=message_id,
                    step_name="FINAL_CONTEXT_VALIDATION",
                    status="ERROR",
                    details={
                        "result": "NO_VALID_CHUNKS",
                        "neighbors_count": len(neighbors) if neighbors else 0,
                        "fallback_attempted": fallback_already_occurred
                    }
                )
                no_info_answer = "Based on the available documents, I could not find specific information to answer your query, or you may not have permission to access the relevant documents."
                for chunk in stream_no_grounded_answer_response(no_info_answer, [], "NO_NEIGHBORS", session_id, message_id, query, current_user_email):
                    yield chunk
                return
            
            # Log successful final validation
            firestore_service.save_system_log(
                session_id=session_id,
                message_id=message_id,
                step_name="FINAL_CONTEXT_VALIDATION",
                status="SUCCESS",
                details={
                    "structured_chunks_count": len(structured_context_chunks),
                    "unique_parent_docs": len(parent_metadata_map),
                    "chunks_to_llm": min(len(structured_context_chunks), config.LLM_CONTEXT_CHUNK_LIMIT)
                }
            )
            
            # Debug information print 
            debug_log(f"context_chunk_map size: {len(context_chunk_map)}")
            debug_log(f"parent_metadata_map size: {len(parent_metadata_map)}")
            
            # 4. Fetch Chat History
            chat_history_formatted = firestore_service.fetch_chat_history(session_id, current_user_email, query, message_id)
            
            # Log chat history fetch
            firestore_service.save_system_log(
                session_id=session_id,
                message_id=message_id,
                step_name="CHAT_HISTORY_FETCH",
                status="SUCCESS",
                details={
                    "history_messages_count": len(chat_history_formatted) if chat_history_formatted else 0,
                    "session_id": session_id
                }
            )

            # 5. Stream LLM Response
            yield f"data: {json.dumps({'status': 'Preparing your answer...'})}\n\n"
            
            # Log LLM call start
            firestore_service.save_system_log(
                session_id=session_id,
                message_id=message_id,
                step_name="LLM_RESPONSE_START",
                status="INFO",
                details={
                    "model": getattr(config, 'GEMINI_CHAT_MODEL_NAME', 'gemini-2.5-flash'),
                    "context_chunks_sent": len(structured_context_chunks),
                    "chat_history_messages": len(chat_history_formatted) if chat_history_formatted else 0
                }
            )
            
            for chunk in stream_response(query, structured_context_chunks, chat_history_formatted, context_chunk_map, parent_metadata_map, session_id, message_id, current_user_email, start_time):
                yield chunk

        except Exception as e:
            logger.error(f"Error in secure chat orchestrator for message_id {message_id}: {e}", exc_info=True)
            
            error_message = f"An unexpected error occurred: {str(e)}"
            if isinstance(e, GeminiConfigurationError):
                 error_message = "The AI service is temporarily unavailable due to a configuration issue. Please contact support."
                 
            # Log the exception
            firestore_service.save_system_log(
                session_id=session_id,
                message_id=message_id,
                step_name="CHAT_ORCHESTRATOR_ERROR",
                status="ERROR",
                details={
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            )
            yield f"data: {json.dumps({'error': error_message})}\n\n"

    return Response(generator(), mimetype='text/event-stream')


@secure_doc_bp.route('/docs/<string:doc_id>/metadata', methods=['GET'])
@jwt_required()
def get_secure_document_metadata(doc_id):
    """
    Retrieves metadata for a specific document, but only if the user has access.
    Admins and superadmins can access all documents.
    """
    current_user_email = get_jwt_identity()
    claims = get_jwt()
    user_role = claims.get("role")
    
    doc_ref = db.collection("document_metadata").document(doc_id)
    doc_snap = doc_ref.get()

    if not doc_snap.exists:
        return jsonify({"msg": "Document not found."}), 404

    # Admins and superadmins can access all documents
    if user_role in ["superadmin"]:
        doc_data = doc_snap.to_dict()
        return jsonify({"msg": "Access granted.", "document_id": doc_id, "categories": doc_data.get("categories", [])})

    doc_categories = doc_snap.to_dict().get("categories", [])
    
    # Query users by email field instead of using email as document ID
    user_query = db.collection("users").where("email", "==", current_user_email).limit(1).get()
    if not user_query:
        return jsonify({"msg": "User not found."}), 403
    
    user_doc = user_query[0]
    user_accessible_categories = user_doc.to_dict().get("accessible_categories", [])

    # Check for overlap
    if not any(cat in user_accessible_categories for cat in doc_categories):
        return jsonify({"msg": "Access denied to this document."}), 403

    # If access is granted, we can call the original function or duplicate logic.
    # For simplicity, we'll just return a success message here.
    # A real implementation would return the full metadata.
    return jsonify({"msg": "Access granted.", "document_id": doc_id, "categories": doc_categories})
