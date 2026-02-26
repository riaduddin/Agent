# backend/app/services/chat/vector_search_service.py
from app.services import vertex_ai_service
from app.models.system_log_model import SystemLogModel
import time
import json
import hashlib
import re
from app.utils.redis_client import get_redis_client
from app import config
from typing import List, Optional, Tuple
import logging
from app.utils.debug_logger import debug_log, debug_perf, debug_separator

# Configure logger
logger = logging.getLogger(__name__)

# Quality threshold for vector search results
# Distance values: 0 = identical, higher = more different
# For DOT_PRODUCT_DISTANCE, higher values (closer to 1) are better
# Adjust based on your embedding model and use case
MINIMUM_QUALITY_NEIGHBORS = getattr(config, 'VECTOR_SEARCH_MIN_QUALITY_NEIGHBORS', 10)
MAX_DISTANCE_THRESHOLD = getattr(config, 'VECTOR_SEARCH_MAX_DISTANCE_THRESHOLD', 0.90)  # None = no threshold


def normalize_query(query: str) -> str:
    """
    Normalize query for better cache hit rate.
    """
    normalized = query.lower().strip()
    normalized = re.sub(r'\s+', ' ', normalized)
    return normalized


def _filter_by_quality(neighbors: list, max_distance: float = None) -> list:
    """
    Filter neighbors by distance quality threshold.
    
    Args:
        neighbors: List of neighbor results with 'distance' field
        max_distance: Maximum acceptable distance (lower = more similar for cosine/euclidean)
    
    Returns:
        Filtered list of high-quality neighbors
    """
    if not max_distance or not neighbors:
        return neighbors
    
    filtered = [n for n in neighbors if n.get('distance', float('inf')) <= max_distance]
    logger.info(f"Quality filter: {len(neighbors)} -> {len(filtered)} neighbors (max_distance={max_distance})")
    return filtered


def _has_sufficient_quality_results(neighbors: list, min_count: int = MINIMUM_QUALITY_NEIGHBORS) -> bool:
    """
    Check if we have enough quality results to proceed without fallback.
    
    Args:
        neighbors: List of neighbor results
        min_count: Minimum number of neighbors required
    
    Returns:
        True if we have sufficient quality results
    """
    if not neighbors:
        return False
    
    # Check count
    if len(neighbors) < min_count:
        logger.info(f"Insufficient results: {len(neighbors)} < {min_count} minimum required")
        return False
    
    return True


def _perform_vector_search_with_variations(
    query_embedding: list, 
    query: str, 
    session_id: str, 
    message_id: str, 
    allowed_doc_ids: Optional[List[str]], 
    metadata_filters: Optional[dict],
    neighbor_multiplier: int = 1,
    cached_variations: Optional[List[str]] = None  # NEW: Reuse variations across attempts
) -> Tuple[list, List[str]]:
    """
    Internal helper to perform vector search with query variations.
    
    Args:
        query_embedding: The query embedding vector
        query: Original query text
        session_id: Session ID for logging
        message_id: Message ID for logging
        allowed_doc_ids: Document IDs the user has access to
        metadata_filters: Metadata filters to apply
        neighbor_multiplier: Multiplier for the number of neighbors (1x, 3x, 5x)
        cached_variations: Pre-generated variations to reuse (avoids regenerating on each attempt)
    
    Returns:
        Tuple of (neighbor results list, generated variations list for reuse)
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    all_neighbors = []
    seen_ids = set()
    
    # Apply multiplier to neighbor counts
    top_k_original = config.VECTOR_SEARCH_TOP_K_ORIGINAL * neighbor_multiplier
    top_k_variation = config.VECTOR_SEARCH_TOP_K_VARIATION * neighbor_multiplier
    merged_cap = config.VECTOR_SEARCH_MERGED_CAP * neighbor_multiplier
    
    logger.info(f"Vector search with {neighbor_multiplier}x multiplier: TOP_K_ORIGINAL={top_k_original}, TOP_K_VARIATION={top_k_variation}, MERGED_CAP={merged_cap}")
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        # Helper to run search safely
        def run_search(search_query, is_original=False):
            try:
                k = top_k_original if is_original else top_k_variation
                q_embedding = vertex_ai_service.get_text_embedding(search_query) if not is_original else query_embedding
                if not q_embedding: 
                    return []
                return vertex_ai_service.find_vector_neighbors(q_embedding, num_neighbors=k, allowed_doc_ids=allowed_doc_ids, metadata_filters=metadata_filters) or []
            except Exception as e:
                logger.error(f"Search failed for query '{search_query}': {e}")
                return []

        # Start original search
        future_original = executor.submit(run_search, query, True)
        
        # Generate variations only if not already cached
        variations = cached_variations
        if variations is None:
            SystemLogModel.add_log_entry(SystemLogModel(
                session_id=session_id, message_id=message_id, user_query_text=query,
                step_name=f"QUERY_EXPANSION_START_{neighbor_multiplier}X", status="INFO"
            ).to_dict())
            
            future_variations = executor.submit(vertex_ai_service.generate_query_variations, query)
            variations = future_variations.result()
            debug_log(f"Generated variations (multiplier={neighbor_multiplier}x): {variations}")
            
            SystemLogModel.add_log_entry(SystemLogModel(
                session_id=session_id, message_id=message_id, user_query_text=query,
                step_name=f"QUERY_EXPANSION_COMPLETE_{neighbor_multiplier}X", status="INFO",
                step_details={"variations": variations, "multiplier": neighbor_multiplier}
            ).to_dict())
        else:
            logger.info(f"Reusing {len(variations)} cached variations for {neighbor_multiplier}x search")
        
        # Use original results as soon as available
        original_neighbors = future_original.result()
        if original_neighbors:
            for n in original_neighbors:
                if n['id'] not in seen_ids:
                    seen_ids.add(n['id'])
                    all_neighbors.append(n)
        
        # Search variations using Batch Embeddings + Parallel Search
        unique_variations = [v for v in variations if v.lower() != query.lower()]
        
        if unique_variations:
            try:
                # Batch generate embeddings
                t_emb_start = time.time()
                var_embeddings = vertex_ai_service.get_text_embeddings_batch(unique_variations)
                debug_perf(f"Batch Embedding for {len(unique_variations)} variations (multiplier={neighbor_multiplier}x)", time.time() - t_emb_start)

                # Parallel Vector Search
                search_futures = {
                    executor.submit(vertex_ai_service.find_vector_neighbors, emb, top_k_variation, allowed_doc_ids, metadata_filters): v_query 
                    for emb, v_query in zip(var_embeddings, unique_variations)
                }

                for future in as_completed(search_futures):
                    v_query = search_futures[future]
                    try:
                        v_neighbors = future.result() or []
                        v_count = 0
                        for n in v_neighbors:
                            if n['id'] not in seen_ids:
                                seen_ids.add(n['id'])
                                all_neighbors.append(n)
                                v_count += 1
                        debug_log(f"Variation '{v_query}' (multiplier={neighbor_multiplier}x) added {v_count} unique neighbors.")
                    except Exception as e:
                        logger.error(f"Variation search future failed for '{v_query}': {e}")
            except Exception as e:
                logger.error(f"Batch embedding or search failed for variations: {e}")

    # Limit to configured cap
    neighbors = all_neighbors[:merged_cap]
    debug_perf(f"Vector Search (multiplier={neighbor_multiplier}x)", neighbors=len(neighbors), total=len(all_neighbors))
    
    # Return both neighbors and variations (for reuse in subsequent attempts)
    return neighbors, variations or []


def find_relevant_chunks(query_embedding: list, query: str, session_id: str, message_id: str, user_email: str, allowed_doc_ids: Optional[List[str]] = None, metadata_filters: Optional[dict] = None, status_callback=None) -> list:
    """
    Finds relevant document chunks using Vector Search, with Redis caching.
    Logs the process.
    
    Args:
        query_embedding: The query embedding vector
        query: Original query text
        session_id: Session ID for logging
        message_id: Message ID for logging
        user_email: User email for caching
        allowed_doc_ids: Document IDs the user has access to
        metadata_filters: Metadata filters to apply
        status_callback: Optional callback function that receives status messages (e.g., for streaming progress)
    """
    start_time = time.time()
    redis_client = get_redis_client()
    
    # Normalize query for better cache hit rate
    normalized_query = normalize_query(query)
    
    # Create a stable, user-specific cache key including filters
    # Uses normalized query for better cache hits on similar queries
    filters_str = json.dumps(metadata_filters, sort_keys=True) if metadata_filters else ""
    cache_string = f"{normalized_query}:{user_email}:{filters_str}"
    cache_key = f"vector_search:{hashlib.sha256(cache_string.encode()).hexdigest()}"

    # 1. Check cache first
    try:
        cached_neighbors = redis_client.get(cache_key)
        if cached_neighbors:
            SystemLogModel.add_log_entry(SystemLogModel(
                session_id=session_id, message_id=message_id, user_query_text=query,
                step_name="VECTOR_SEARCH_CACHE_HIT", status="INFO",
                step_details={"cache_key": cache_key}
            ).to_dict())
            debug_perf("Step 2 (Vector Search from CACHE)", time.time() - start_time)
            return json.loads(cached_neighbors)
    except Exception as e_redis_get:
        SystemLogModel.add_log_entry(SystemLogModel(
            session_id=session_id, message_id=message_id, user_query_text=query,
            step_name="VECTOR_SEARCH_CACHE_READ_FAILED", status="WARNING",
            error_message=f"Failed to read from Redis cache: {str(e_redis_get)}",
            step_details={"cache_key": cache_key}
        ).to_dict())

    SystemLogModel.add_log_entry(SystemLogModel(
        session_id=session_id, message_id=message_id, user_query_text=query,
        step_name="VECTOR_SEARCH_CACHE_MISS_START", status="INFO"
    ).to_dict())

    try:
        # Progressive Fallback Strategy:
        # 1. Try with base neighbor count (1x multiplier)
        # 2. If no results OR insufficient quality results, try with 3x multiplier
        # 3. If still insufficient, try with 5x multiplier
        # 4. If still no results, try without metadata filters (existing fallback)
        
        neighbors = []
        cached_variations = None  # Cache variations across attempts to avoid regenerating
        multipliers = config.VECTOR_SEARCH_FALLBACK_MULTIPLIERS  # [1, 3, 5]
        
        for attempt_index, multiplier in enumerate(multipliers):
            if neighbors:  # If we already found results, don't continue
                break
                
            # Send a user-friendly status before retrying with broader search
            # Messages should be POSITIVE and REASSURING - focus on what we're doing, not what we didn't find
            if attempt_index > 0:  # Only show messages for 2nd, 3rd attempts
                if attempt_index == 1:
                    status_msg = "Searching deeper for the most relevant information..."
                elif attempt_index == 2:
                    status_msg = "Performing a comprehensive search across all documents..."
                else:
                    status_msg = "Running an extended search to ensure accuracy..."

                if status_callback:
                    status_callback(status_msg)
                logger.info(f"Fallback attempt {attempt_index + 1}: {status_msg}")
                
            logger.info(f"Attempting vector search with {multiplier}x neighbor multiplier...")
            SystemLogModel.add_log_entry(SystemLogModel(
                session_id=session_id, message_id=message_id, user_query_text=query,
                step_name=f"VECTOR_SEARCH_ATTEMPT_{multiplier}X", status="INFO",
                step_details={"multiplier": multiplier, "has_metadata_filters": bool(metadata_filters)}
            ).to_dict())
            
            # IMPROVED: Pass cached_variations to avoid regenerating on each attempt
            # Function returns (neighbors, variations) tuple
            new_neighbors, cached_variations = _perform_vector_search_with_variations(
                query_embedding, query, session_id, message_id,
                allowed_doc_ids, metadata_filters, 
                neighbor_multiplier=multiplier,
                cached_variations=cached_variations
            )
            
            # Merge new results with existing (deduplication)
            if new_neighbors:
                existing_ids = {n['id'] for n in neighbors}
                for n in new_neighbors:
                    if n['id'] not in existing_ids:
                        neighbors.append(n)
                        existing_ids.add(n['id'])
                
                logger.info(f"Vector search with {multiplier}x multiplier: found {len(new_neighbors)} neighbors, total now: {len(neighbors)}")
                SystemLogModel.add_log_entry(SystemLogModel(
                    session_id=session_id, message_id=message_id, user_query_text=query,
                    step_name=f"VECTOR_SEARCH_SUCCESS_{multiplier}X", status="SUCCESS",
                    step_details={"multiplier": multiplier, "new_neighbor_count": len(new_neighbors), "total_count": len(neighbors)}
                ).to_dict())
                
                # IMPROVED: Check for sufficient quality results to decide if fallback needed
                if _has_sufficient_quality_results(neighbors, MINIMUM_QUALITY_NEIGHBORS):
                    # Inform the user if results were found after expanding the search
                    if status_callback and multiplier > 1:
                        status_callback("Found relevant information for you!")
                    break  # Found sufficient results, exit the loop
                else:
                    logger.info(f"Found {len(neighbors)} neighbors but below minimum quality threshold ({MINIMUM_QUALITY_NEIGHBORS}), continuing fallback...")
            else:
                logger.warning(f"Vector search with {multiplier}x multiplier returned 0 results")
                SystemLogModel.add_log_entry(SystemLogModel(
                    session_id=session_id, message_id=message_id, user_query_text=query,
                    step_name=f"VECTOR_SEARCH_EMPTY_{multiplier}X", status="WARNING",
                    step_details={"multiplier": multiplier}
                ).to_dict())

        if not neighbors:
             # If completely empty after trying all multipliers, try one more fallback:
             # FALLBACK STRATEGY FOR METADATA FILTERS:
             # If we used metadata_filters and got 0 results even with increased neighbors,
             # it's highly likely the filters were too strict (e.g. AI extracted "BOQ Report" but DB has "INVOICE").
             # We should retry WITHOUT filters to avoid a "No Info" response.
             if metadata_filters:
                 logger.warning(f"Vector search with filters {metadata_filters} returned 0 results even with progressive fallback. Retrying FINAL FALLBACK without filters.")

                 # Log the fallback for performance tracking
                 SystemLogModel.add_log_entry(SystemLogModel(
                    session_id=session_id, message_id=message_id, user_query_text=query,
                    step_name="VECTOR_SEARCH_METADATA_FILTER_FALLBACK_TRIGGERED", status="WARNING",
                    step_details={"failed_filters": metadata_filters}
                 ).to_dict())

                 # Notify the user we are refining search parameters
                 if status_callback:
                     status_callback("Refining search parameters to ensure the best results...")

                 # Recursive call without metadata filters, but with base multiplier only
                 # This prevents infinite recursion and excessive API calls
                 return find_relevant_chunks(
                     query_embedding, query, session_id, message_id, 
                     allowed_doc_ids=allowed_doc_ids, # Keep security filter!
                     user_email=user_email, 
                     metadata_filters=None # Disable filters for fallback
                 )

             logger.warning("Vector search returned no results even with progressive fallback and filter removal.")
             # Return empty list - caller will handle "not found" message
             # NOTE: We do NOT cache empty results to allow retry

        # 3. Store successful result in cache ONLY if we have results
        # IMPORTANT: Do NOT cache empty results - this allows users to retry
        # and get fresh results if documents are added or issues are resolved
        if neighbors:
            try:
                redis_client.setex(
                    cache_key,
                    config.VECTOR_SEARCH_CACHE_TTL_SECONDS,
                    json.dumps(neighbors)
                )
                SystemLogModel.add_log_entry(SystemLogModel(
                    session_id=session_id, message_id=message_id, user_query_text=query,
                    step_name="VECTOR_SEARCH_CACHE_WRITE_SUCCESS", status="INFO",
                    step_details={"cache_key": cache_key, "ttl": config.VECTOR_SEARCH_CACHE_TTL_SECONDS}
                ).to_dict())
            except Exception as e_redis_set:
                SystemLogModel.add_log_entry(SystemLogModel(
                    session_id=session_id, message_id=message_id, user_query_text=query,
                    step_name="VECTOR_SEARCH_CACHE_WRITE_FAILED", status="WARNING",
                    error_message=f"Failed to write to Redis cache: {str(e_redis_set)}",
                    step_details={"cache_key": cache_key}
                ).to_dict())
        else:
            # Log that we're NOT caching empty results
            SystemLogModel.add_log_entry(SystemLogModel(
                session_id=session_id, message_id=message_id, user_query_text=query,
                step_name="VECTOR_SEARCH_CACHE_SKIP_EMPTY", status="INFO",
                step_details={"reason": "Empty results are not cached to allow retry"}
            ).to_dict())
            logger.info(f"Skipping cache for empty results - user can retry query: {query[:50]}...")

        SystemLogModel.add_log_entry(SystemLogModel(
            session_id=session_id, message_id=message_id, user_query_text=query,
            step_name="VECTOR_SEARCH_SUCCESS", status="SUCCESS",
            step_details={"retrieved_neighbor_count": len(neighbors), "neighbor_ids_preview": [n['id'][:10] for n in neighbors[:3]], "distances_preview": [n.get('distance') for n in neighbors[:3]]}
        ).to_dict())

        # Print the neighbor IDs and distances for debugging
        debug_separator("Vector Search Results")
        debug_log(f"Retrieved {len(neighbors)} neighbors: {[n['id'][:10] for n in neighbors[:3]]} with distances: {[n.get('distance') for n in neighbors[:3]]}")

        return neighbors

    except Exception as e_vec_search:
        SystemLogModel.add_log_entry(SystemLogModel(
            session_id=session_id, message_id=message_id, user_query_text=query,
            step_name="VECTOR_SEARCH_FAILED_EXCEPTION_TERMINATING", status="ERROR", 
            error_message=f"Exception during vector search: {str(e_vec_search)}",
            step_details={"exception_type": type(e_vec_search).__name__}
        ).to_dict())
        # Re-raise the exception to be handled by the main route
        raise
