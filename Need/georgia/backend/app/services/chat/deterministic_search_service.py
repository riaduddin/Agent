
import logging
from typing import List, Dict, Any, Optional
from google.cloud import firestore
from app import db
from app.utils.debug_logger import debug_log

logger = logging.getLogger(__name__)

class DeterministicSearchService:
    """
    Service for performing deterministic "Hard Match" searches against Firestore
    using extracted entities. This serves as Phase 0 in the search pipeline,
    prioritizing exact matches before falling back to vector search.
    """

    MAX_CHUNKS_PER_ENTITY = 5
    MAX_TOTAL_CHUNKS = 25

    @classmethod
    def find_chunks_by_entities(cls, entities: List[str], allowed_doc_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Finds chunks that contain any of the provided entities.
        
        Args:
            entities: List of entity strings to search for (e.g. document IDs, names).
            allowed_doc_ids: Optional list of parent document IDs the user is allowed to access.
                             If None, assumes no RBAC filtering is needed at this stage (but it really should be).
                             If empty list, returns empty result immediately.

        Returns:
            List of unique chunks with 'deterministic_match' metadata.
        """
        if not entities:
            return []
            
        if allowed_doc_ids is not None and len(allowed_doc_ids) == 0:
            logger.info("DeterministicSearchService: No allowed doc IDs provided, returning empty.")
            return []

        # Deduplicate entities
        unique_entities = list(set([e.strip() for e in entities if e and e.strip()]))
        
        logger.info(f"DeterministicSearchService: Searching for {len(unique_entities)} entities: {unique_entities}")
        debug_log("Deterministic Search Started", entities=unique_entities)

        all_chunks = {}
        
        # Firestore 'in' query supports max 10 values, but 'array-contains' only supports single value.
        # We must iterate. Parallelization could be added if latency becomes an issue,
        # but for < 5 entities per query, serial is likely fine.
        
        total_fetched = 0
        
        for entity in unique_entities:
            if total_fetched >= cls.MAX_TOTAL_CHUNKS:
                break
                
            try:
                # Query: collection('document_chunks').where('entities', 'array_contains', entity).limit(5)
                chunks_ref = db.collection('document_chunks')
                query = chunks_ref.where('entities', 'array_contains', entity)
                
                # Apply RBAC filtering if possible?
                # Firestore client-side filtering for 'parent_doc_id' IN list is hard with array-contains.
                # Strategy: Fetch potential matches, THEN filter by allowed_doc_ids in python.
                # This risks over-fetching, but essential for security.
                
                query = query.limit(cls.MAX_CHUNKS_PER_ENTITY)
                results = query.stream()
                
                entity_match_count = 0
                for doc in results:
                    chunk_data = doc.to_dict()
                    chunk_id = doc.id
                    parent_doc_id = chunk_data.get('original_doc_firestore_id')
                    
                    # RBAC Check
                    if allowed_doc_ids is not None and parent_doc_id not in allowed_doc_ids:
                        continue
                        
                    if chunk_id not in all_chunks:
                        chunk_data['chunk_id'] = chunk_id
                        chunk_data['search_score'] = 2.0 # High priority score for exact match
                        chunk_data['match_type'] = 'deterministic'
                        chunk_data['matched_entity'] = entity
                        all_chunks[chunk_id] = chunk_data
                        
                        entity_match_count += 1
                        total_fetched += 1
                        
            except Exception as e:
                logger.error(f"DeterministicSearchService: Error querying entity '{entity}': {e}")
                
        logger.info(f"DeterministicSearchService: Found {len(all_chunks)} unique chunks.")
        debug_log("Deterministic Search Completed", found_count=len(all_chunks))
        
        return list(all_chunks.values())
