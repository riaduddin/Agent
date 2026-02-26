# backend/app/services/chat/multi_query_search_service.py
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional
from app.services import vertex_ai_service
from app.services.chat.deterministic_search_service import DeterministicSearchService
from app import config
from app.utils.debug_logger import debug_log, debug_perf, debug_separator

logger = logging.getLogger(__name__)


def multi_query_search(
    query: str,
    query_embedding: list,
    entities: dict,
    allowed_doc_ids: Optional[List[str]],
    metadata_filters: Optional[dict],
    entity_queries: List[str]
) -> list:
    """
    Executes multi-query parallel search with intelligent result fusion.
    Now includes Phase 0: Deterministic "Hard Match" for exact entity lookups.
    
    Args:
        query: Original user query
        query_embedding: Embedding of original query
        entities: Extracted entities dictionary
        allowed_doc_ids: Document IDs user can access
        metadata_filters: Metadata filters for vector search
        entity_queries: Pre-generated entity-focused queries
    
    Returns:
        List of ranked neighbors (top 60)
    """
    t_start = time.time()

    # --- PHASE 0: DETERMINISTIC FETCH (HARD MATCH) ---
    # Fetch chunks that strictly match extracted entities (identifiers, names, etc.)
    # We do this FIRST to prioritize them in the final ranking.
    deterministic_chunks = []
    deduped_entities = [] # Initialize here to ensure availability for stats
    try:
        t_det_start = time.time()
        # Extract ALL entity values from the structured dict to a flat list for deterministic search
        flat_entities = []
        if entities:
            def extract_values(data):
                if isinstance(data, list):
                    for item in data:
                        extract_values(item)
                elif isinstance(data, dict):
                    for key, val in data.items():
                        # Skip strictly internal or non-searchable fields if any
                        if key == "entity_details": # This is redundant if we already have the values in other keys
                            continue
                        extract_values(val)
                elif data is not None:
                    # Append string representation of anything else (numbers, specific dates)
                    val_str = str(data).strip()
                    if val_str and val_str.lower() not in ['none', 'null', 'n/a']:
                        flat_entities.append(val_str)

            extract_values(entities)
        
        if flat_entities:
            # Deduplicate while preserving some order AND expanding case variations
            seen = set()
            deduped_entities = []
            
            for e in flat_entities:
                if not e: continue
                
                # Generate variations to ensure we hit the DB regardless of stored case
                variations = [e.strip()]            # 1. Original (e.g. "j. rakesh kumar")
                variations.append(e.strip().title()) # 2. Title Case (e.g. "J. Rakesh Kumar")
                variations.append(e.strip().upper()) # 3. UPPERCASE (e.g. "J. RAKESH KUMAR")
                
                for var in variations:
                    # We want to add DIFFERENT cases as distinct search terms
                    if var and var not in seen:
                        seen.add(var)
                        deduped_entities.append(var)

            debug_log(f"Phase 0: Executing deterministic search for {len(deduped_entities)} entities: {deduped_entities[:5]}...")
            deterministic_chunks = DeterministicSearchService.find_chunks_by_entities(
                deduped_entities, allowed_doc_ids
            )
            debug_perf("Phase 0 Deterministic Fetch", time.time() - t_det_start)
    except Exception as e:
        logger.error(f"Phase 0 Deterministic Search failed: {e}")

    
    all_queries = []
    
    # 1. Original query (highest weight)
    all_queries.append({
        "query": query,
        "embedding": query_embedding,
        "top_k": config.VECTOR_SEARCH_TOP_K_ORIGINAL,
        "type": "original",
        "weight": 1.0
    })
    
    # 2. Query variations (existing system - medium weight)
    try:
        variations = vertex_ai_service.generate_query_variations(query)
        for var in variations[:config.VECTOR_SEARCH_VARIATION_COUNT]:
            all_queries.append({
                "query": var,
                "embedding": None,  # Will be generated in batch
                "top_k": config.VECTOR_SEARCH_TOP_K_VARIATION,
                "type": "variation",
                "weight": 0.8
            })
    except Exception as e:
        logger.warning(f"Failed to generate query variations: {e}")
    
    # 3. Entity-focused queries (NEW - highest weight)
    for eq in entity_queries[:config.ENTITY_QUERY_MAX_COUNT]:
        all_queries.append({
            "query": eq,
            "embedding": None,  # Will be generated in batch
            "top_k": config.ENTITY_QUERY_TOP_K,
            "type": "entity",
            "weight": config.ENTITY_QUERY_WEIGHT
        })
    
    debug_separator("Multi-Query Search")
    debug_log(f"Total queries: {len(all_queries)} (1 original + {len([q for q in all_queries if q['type'] == 'variation'])} variations + {len(entity_queries)} entity)")
    
    # 4. Batch generate embeddings for queries that need them
    queries_needing_embedding = [q for q in all_queries if q["embedding"] is None]
    if queries_needing_embedding:
        try:
            t_emb_start = time.time()
            query_texts = [q["query"] for q in queries_needing_embedding]
            embeddings = vertex_ai_service.get_text_embeddings_batch(query_texts)
            
            # Assign embeddings back to query objects
            for i, q in enumerate(queries_needing_embedding):
                q["embedding"] = embeddings[i]
            
            debug_perf(f"Batch Embedding for {len(query_texts)} queries", time.time() - t_emb_start)
        except Exception as e:
            logger.error(f"Batch embedding failed: {e}")
            # Remove queries that failed to get embeddings
            all_queries = [q for q in all_queries if q["embedding"] is not None]
    
    
    # 5. Parallel vector search
    all_neighbors = {}
    
    # --- MERGE PHASE 0 RESULTS ---
    # Hydrate all_neighbors with deterministic limits FIRST.
    # They get a special "hard match" status.
    for chunk in deterministic_chunks:
        c_id = chunk.get('chunk_id') or chunk.get('id') # Handle potential key variance
        if not c_id: continue
        
        all_neighbors[c_id] = {
            "id": c_id,
            "distance": 0.0, # 0 distance = perfect match
            "min_distance": 0.0,
            "score": chunk.get('search_score', 2.0), # Default 2.0 if not set
            "match_count": 1,
            "matched_by_types": {'deterministic'},
            "best_source_query": f"Hard Match: {chunk.get('matched_entity', 'unknown')}",
            "is_deterministic": True # Flag to ensure it survives filtering
        }

    query_stats = {}  # Track contribution of each query
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {}
        for query_obj in all_queries:
            # INTELLIGENT FILTERING STRATEGY (as implemented previously)
            active_filters = None
            if query_obj["type"] == "original":
                active_filters = metadata_filters
            elif query_obj["type"] == "entity" and entities and query_obj["query"] in entities.get('identifiers', []):
                 active_filters = metadata_filters
            else:
                active_filters = None

            future = executor.submit(
                vertex_ai_service.find_vector_neighbors,
                query_obj["embedding"],
                query_obj["top_k"],
                allowed_doc_ids,
                active_filters
            )
            futures[future] = query_obj
        
        # 6. Collect results with weighted scoring
        for future in as_completed(futures):
            query_obj = futures[future]
            q_text = query_obj["query"]
            q_type = query_obj["type"]
            
            try:
                neighbors = future.result() or []
                debug_log(f"Query '{q_text[:50]}...' ({q_type}) returned {len(neighbors)} neighbors")
                
                # Record stats
                query_stats[q_text] = {
                    "type": q_type,
                    "count": len(neighbors),
                    "active_filters": active_filters,
                    "contribution": 0,
                    "raw_results": neighbors # Store raw results for detailed debugging
                }
                
                for neighbor in neighbors:
                    chunk_id = neighbor["id"]
                    distance = neighbor["distance"]
                    similarity = 1 - distance
                    weighted_score = similarity * query_obj["weight"]
                    
                    if chunk_id in all_neighbors:
                        all_neighbors[chunk_id]["score"] += weighted_score
                        all_neighbors[chunk_id]["match_count"] += 1
                        all_neighbors[chunk_id]["matched_by_types"].add(q_type)
                        
                        # Update best source if this query is closer (but don't overwrite 0.0 deterministic)
                        if distance < all_neighbors[chunk_id]["min_distance"]:
                            all_neighbors[chunk_id]["min_distance"] = distance
                            all_neighbors[chunk_id]["best_source_query"] = q_text
                    else:
                        all_neighbors[chunk_id] = {
                            "id": chunk_id,
                            "distance": distance,
                            "min_distance": distance,
                            "score": weighted_score,
                            "match_count": 1,
                            "matched_by_types": {q_type},
                            "best_source_query": q_text,
                            "is_deterministic": False
                        }
            except Exception as e:
                logger.error(f"Search failed for query '{q_text[:50]}...': {e}")
                query_stats[q_text] = {"type": q_type, "error": str(e), "count": 0, "contribution": 0}
    
    # 7. Rank by priority score
    for chunk_id, data in all_neighbors.items():
        # Boost deterministic matches significantly to ensure they stay on top
        base_priority = (
            data["match_count"] * config.FUSION_MATCH_COUNT_WEIGHT +
            data["score"] * config.FUSION_SCORE_WEIGHT -
            data["min_distance"] * config.FUSION_DISTANCE_PENALTY
        )
        if data.get("is_deterministic"):
            base_priority += 10.0 # Huge boost
            
        data["priority"] = base_priority
    
    # --- STRATEGY: GUARANTEED DIVERSITY ---
    # Ensure top 5 results from EVERY query make it to the final list
    # ignoring the global priority score if necessary.
    must_have_chunk_ids = set()
    
    for q_text, stats in query_stats.items():
        if "raw_results" in stats and stats["raw_results"]:
            # Sort by distance (closest first)
            top_hits = sorted(stats["raw_results"], key=lambda x: x['distance'])[:config.FUSION_GUARANTEED_PER_QUERY]
            for hit in top_hits:
                must_have_chunk_ids.add(hit['id'])
    
    # Also ensure ALL deterministic matches are must-haves
    for c_id, data in all_neighbors.items():
        if data.get("is_deterministic"):
            must_have_chunk_ids.add(c_id)
    
    debug_log(f"Identified {len(must_have_chunk_ids)} 'must-have' chunks (including {len(deterministic_chunks)} deterministic)")

    # Sort ALL results by priority
    sorted_all = sorted(
        all_neighbors.values(),
        key=lambda x: (x["priority"], x["match_count"], -x["min_distance"]),
        reverse=True
    )
    
    # Build Final List
    final_neighbors = []
    seen_ids = set()
    
    # 1. Add Must-Haves (sorted by their priority to keep best ones first)
    must_have_candidates = [n for n in sorted_all if n["id"] in must_have_chunk_ids]
    for n in must_have_candidates:
        final_neighbors.append(n)
        seen_ids.add(n["id"])
        
    # 2. Fill remaining slots with other high-priority results
    for n in sorted_all:
        if len(final_neighbors) >= config.VECTOR_SEARCH_MERGED_CAP:
            break
        if n["id"] not in seen_ids:
            final_neighbors.append(n)
            seen_ids.add(n["id"])
            
    # Remove fusion_result slice since we built final_neighbors manually
    fusion_result = final_neighbors 
    
    # Calculate final contribution stats
    result_ids = set(n["id"] for n in final_neighbors)
    
    # 8. Prepare Detailed Search Stats
    execution_stats = {
        "execution_time_ms": round((time.time() - t_start) * 1000, 2),
        "total_queries": len(all_queries),
        "total_unique_neighbors": len(all_neighbors),
        "final_results_count": len(final_neighbors),
        "final_results_count": len(final_neighbors),
        "deterministic_matches": len(deterministic_chunks), # NEW STAT
        "deterministic_search_details": [], # NEW: detailed breakdown by entity
        "search_configuration": {
            "entity_query_weight": config.ENTITY_QUERY_WEIGHT,
            "fusion_model": {
                "match_weight": config.FUSION_MATCH_COUNT_WEIGHT,
                "score_weight": config.FUSION_SCORE_WEIGHT,
                "distance_penalty": config.FUSION_DISTANCE_PENALTY
            }
        },
        "query_performance": [],
        "detailed_query_results": [], # NEW: Exact breakdown of what each query found
        "top_fused_results": [] 
    }
    
    for q_text, stats in query_stats.items():
        execution_stats["query_performance"].append({
            "query": q_text,
            "type": stats["type"],
            "results_found": stats["count"],
            "active_filters": stats.get("active_filters") or "None"
        })
        
        # Add detailed result mapping (Top 3 per query for brevity in logs)
        if "raw_results" in stats and stats["raw_results"]:
            top_raw = sorted(stats["raw_results"], key=lambda x: x['distance'])[:3]
            execution_stats["detailed_query_results"].append({
                "source_query": q_text,
                "query_type": stats["type"],
                "results_found_count": stats["count"],
                "top_3_hits": [
                    {
                        "chunk_id": r["id"], 
                        "similarity_score": round(1 - r["distance"], 4)
                    } 
                    for r in top_raw
                ]
            })

    # NEW: Populate deterministic breakdown with ALL searched entities
    # First, map matches to entities for easy lookup
    det_matches_map = {}
    for chunk in deterministic_chunks:
        matched_entity = chunk.get('matched_entity', '').lower().strip()
        if matched_entity:
            if matched_entity not in det_matches_map:
                det_matches_map[matched_entity] = []
            det_matches_map[matched_entity].append(chunk.get('chunk_id') or chunk.get('id'))
    
    # Now iterate over ALL entities that were searched for (deduped_entities)
    # This ensures we log entities even if they found 0 results
    for entity_str in deduped_entities:
        norm_entity = entity_str.lower().strip()
        matches = det_matches_map.get(norm_entity, [])
        
        execution_stats["deterministic_search_details"].append({
            "entity": entity_str, # Use original case for display
            "count": len(matches),
            "chunk_ids": matches
        })

    
    # Sort performance by results found (descending)
    execution_stats["query_performance"].sort(key=lambda x: x["results_found"], reverse=True)
    execution_stats["detailed_query_results"].sort(key=lambda x: x["results_found_count"], reverse=True)
    
    # Add top 5 results details for deep debugging
    for rank, res in enumerate(final_neighbors[:5]):
        execution_stats["top_fused_results"].append({
            "rank": rank + 1,
            "chunk_id": res["id"],
            "priority_score": round(res["priority"], 2),
            "match_count": res["match_count"],
            "matched_by": list(res["matched_by_types"]),
            "best_source_query": res.get("best_source_query", "N/A"),
            "is_deterministic": res.get("is_deterministic", False)
        })

    debug_separator("Fusion Results")
    debug_log(f"Total unique neighbors: {len(all_neighbors)}")
    debug_log(f"Deterministic matches: {len(deterministic_chunks)}")
    debug_log(f"Final output: {len(final_neighbors)}")

    debug_perf("Multi-Query Search (total)", time.time() - t_start)
    
    # Return BOTH results and stats
    results = [{"id": n["id"], "distance": n["min_distance"]} for n in final_neighbors]
    return results, execution_stats
