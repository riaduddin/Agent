# backend/app/models/metadata_model.py
import logging # Import logging
from typing import Optional, Tuple, Union, List, Any # Import Union and List
from app import db # Import the initialized Firestore client
from google.cloud import firestore # Import firestore module
from google.cloud.firestore_v1.base_query import FieldFilter
import datetime
import uuid # For message_id if not provided
import json
import logging
from typing import Dict, Set, List
from app.utils.redis_client import get_redis_client # Import Redis client utility
from app.utils.debug_logger import debug_log, debug_warn, debug_error, debug_separator
import time
import requests
from app import config
try:
    from flask import request
    FLASK_REQUEST_AVAILABLE = True
except ImportError:
    FLASK_REQUEST_AVAILABLE = False
    request = None


# Reference to the 'document_metadata' collection in Firestore (Corrected name)
docs_ref = db.collection('document_metadata')
# Reference to the 'document_chunks' collection
chunks_ref = db.collection('document_chunks')
# Reference to the 'document_chunk_details' collection (New: For Heavy Debugging Data)
chunk_details_ref = db.collection('document_chunk_details')
# Reference to the 'chat_sessions' collection
sessions_ref = db.collection('chat_sessions')

users_ref = db.collection('users')


# --- Logger ---
logger = logging.getLogger(__name__)

def create_doc_metadata(user_email, original_filename, gcs_uri, gcs_blob_name, content_type, file_size, status='pending', total_chunks=0, completed_chunks=0, categories=None, destination_path=None):
    """Creates a new document metadata record in Firestore.
    
    Args:
        destination_path: GCS file path from where file is picked during scheduled batch process.
                         Should be None for manual uploads and other process types.
    """
    try:
        if categories is None:
            categories = []
            
        doc_data = {
            'user_email': user_email,
            'original_filename': original_filename,
            'gcs_uri': gcs_uri, # This is gcs_path_original for parent doc
            'gcs_blob_name': gcs_blob_name,
            'content_type': content_type,
            'file_size_bytes': file_size,
            'upload_timestamp': datetime.datetime.now(tz=datetime.timezone.utc),
            'status': status, # Use the passed status
            'total_chunks': total_chunks, # Initialize
            'completed_chunks': completed_chunks, # Initialize
            'processing_events': [], # Initialize for parent document logging
            'error_message': None,
            'source': 'user_upload',
            'categories': categories, # Add categories
            'destination_path': destination_path, # GCS path for scheduled batch process, None for others
        }
        # Let Firestore generate the document ID
        doc_ref = docs_ref.document()
        doc_ref.set(doc_data)
        debug_log(f"Metadata created for {original_filename} with ID: {doc_ref.id}")
        return doc_ref.id, None # Return document ID and no error
    except Exception as e:
        debug_error(f" Failed to create metadata for {original_filename}: {e}")
        return doc_ref.id, None # Return document ID and no error
    except Exception as e:
        debug_error(f" Failed to create metadata for {original_filename}: {e}")
        return None, f"Failed to create metadata record: {e}"

def save_chunk_details(chunk_id: str, details_data: dict):
    """
    Saves detailed debugging information for a chunk to the separate 'document_chunk_details' collection.
    This ensures we don't pollute the main 'document_chunks' collection with heavy text/metadata.
    """
    try:
        # validation to ensure we don't accidentally overwrite strict existing data with bad data if logic changes
        # meant to be a direct set for this usecase
        chunk_details_ref.document(chunk_id).set(details_data)
        # print(f"Saved details for chunk {chunk_id}") 
    except Exception as e:
        logger.error(f"Failed to save chunk details for {chunk_id}: {e}")

def get_chunk_details(chunk_id: str) -> dict:
    """Retrieves the detailed debugging data for a specific chunk."""
    try:
        doc = chunk_details_ref.document(chunk_id).get()
        if doc.exists:
            return doc.to_dict()
        return None
    except Exception as e:
        logger.error(f"Failed to get chunk details for {chunk_id}: {e}")
        return None

def get_chunks_by_ids(chunk_ids: list[str]) -> dict:
    """
    Retrieves specified fields for given chunk IDs from Firestore using 'in' queries.

    Args:
        chunk_ids: A list of chunk document IDs.

    Returns:
        A dictionary where keys are chunk IDs and values are dictionaries
        containing 'chunk_id', 'original_filename', 'start_page', 'end_page',
        and 'ocr_text_preview'. Returns empty dict on error or if no chunks found.
    """
    chunk_data_map = {}
    if not chunk_ids:
        return chunk_data_map
    
    # Firestore 'in' query limit is 30
    chunk_id_batches = [chunk_ids[i:i + 30] for i in range(0, len(chunk_ids), 30)]

    try:
        for batch_ids in chunk_id_batches:
            if not batch_ids: continue # Skip empty batches

            query = chunks_ref.where("__name__", 'in', batch_ids)
            docs = query.stream()

            for doc in docs:
                if doc.exists:
                    data = doc.to_dict()
                    chunk_data_map[doc.id] = {
                        "chunk_id": doc.id,
                        "original_filename": data.get("original_filename"),
                        "start_page": data.get("start_page"),
                        "end_page": data.get("end_page"),
                        "ocr_text_preview": data.get("extracted_text_preview") or data.get("ocr_text_preview"),
                        "original_doc_firestore_id": data.get("original_doc_firestore_id"),
                        "extracted_entities": data.get("extracted_entities", {}), # For backward compatibility
                        "entities": data.get("entities", []) # Flattened entities array
                    }
                else:
                    logger.warning(f"Document ID {doc.id} from 'in' query result does not exist.")

        found_ids = set(chunk_data_map.keys())
        missing_ids = [cid for cid in chunk_ids if cid not in found_ids]
        if missing_ids:
            logger.warning(f"Could not find chunk documents for IDs: {missing_ids}")

        return chunk_data_map

    except Exception as e:
        logger.error(f"Error retrieving chunks by IDs: {e}", exc_info=True)
        return {}



# =======================================================

def get_chunks_by_ids_by_email(chunk_ids: list[str], user_email: str, auth_header: Optional[str] = None) -> dict:
    """
    Retrieves specified fields for given chunk IDs from Firestore with Redis-optimized caching.
    Filters results based on user permissions using accessible_categories.
    
    Args:
        chunk_ids: A list of chunk document IDs.
        user_email: The email of the user requesting the chunks.
        auth_header: Optional Authorization header for API calls.
    
    Returns:
        A dictionary where keys are chunk IDs and values are dictionaries
        containing 'chunk_id', 'original_filename', 'start_page', 'end_page',
        and 'ocr_text_preview'. Only returns chunks the user has permission to access.
        Returns empty dict on error or if no chunks found.
    """
    start_time = time.time()
    debug_separator()
    debug_log(f"🚀 STARTING get_chunks_by_ids_by_email")
    debug_log(f"   User: {user_email}")
    debug_log(f"   Requested chunks: {len(chunk_ids)}")
    debug_log(f"   Chunk IDs: {chunk_ids[:5]}{'...' if len(chunk_ids) > 5 else ''}")
    debug_separator()
    
    if not chunk_ids:
        debug_error("❌ No chunk IDs provided, returning empty dict")
        return {}
    
    try:
        # Try to connect to Redis
        debug_log(f"📡 STEP 1: Connecting to Redis...")
        try:
            redis_client = get_redis_client()
            debug_log(f"✅ Redis connection successful")
        except Exception as e:
            debug_error(f"❌ Redis connection failed: {e}")
            debug_log(f"🔄 Falling back to Firestore-only mode")
            return _get_chunks_fallback(chunk_ids, user_email)
        
        # Step 1: Get user's accessible categories from external API (Redis cached)
        debug_log(f"🔐 STEP 2: Getting user categories from API for {user_email}")
        user_categories = _get_user_categories_from_api_cached(redis_client, user_email, auth_header=auth_header)
        debug_log(f"   User categories found: {list(user_categories)}")
        
        if not user_categories:
            debug_error("❌ User has no accessible categories or user not found")
            return {}
        
        # Step 2: Batch fetch chunks with Redis caching
        debug_log(f"📦 STEP 3: Fetching chunk metadata...")
        chunks_data = _get_chunks_batch_cached(redis_client, chunk_ids)
        debug_log(f"   Chunks retrieved: {len(chunks_data)}/{len(chunk_ids)}")
        
        if not chunks_data:
            debug_error("❌ No chunk data retrieved")
            return {}
        
        # Step 3: Group chunks by document and prepare data
        debug_log(f"🗂️  STEP 4: Processing chunk data...")
        chunk_data_map = {}
        doc_ids_needed = set()
        chunk_to_doc_map = {}
        
        for chunk_id, chunk_data in chunks_data.items():
            original_doc_id = chunk_data.get("original_doc_firestore_id")
            if original_doc_id:
                doc_ids_needed.add(original_doc_id)
                chunk_to_doc_map[chunk_id] = original_doc_id
                chunk_data_map[chunk_id] = {
                    "chunk_id": chunk_id,
                    "original_filename": chunk_data.get("original_filename"),
                    "start_page": chunk_data.get("start_page"),
                    "end_page": chunk_data.get("end_page"),
                    "ocr_text_preview": chunk_data.get("extracted_text_preview") or chunk_data.get("ocr_text_preview"),
                    "original_doc_firestore_id": original_doc_id,
                    "extracted_entities": chunk_data.get("extracted_entities", {}), # For backward compatibility
                    "entities": chunk_data.get("entities", []) # Flattened entities array
                }
                debug_log(f"   📄 Processed chunk {chunk_id[:8]}... -> entities count: {len(chunk_data_map[chunk_id]['entities'])}")
                debug_log(f"   📄 Processed chunk {chunk_id[:8]}... -> doc {original_doc_id[:8]}...")
        
        debug_log(f"   Total unique documents needed: {len(doc_ids_needed)}")
        debug_log(f"   Document IDs: {list(doc_ids_needed)[:3]}{'...' if len(doc_ids_needed) > 3 else ''}")
        
        if not doc_ids_needed:
            debug_error("❌ No valid document references found in chunks")
            return {}
        
        # Step 4: Check permissions using Redis cache
        debug_log(f"🔒 STEP 5: Checking permissions...")
        permissions = _get_permissions_batch_cached(redis_client, user_email, user_categories, list(doc_ids_needed))
        debug_log(f"   Permission results:")
        for doc_id, has_perm in permissions.items():
            status = "✅ ALLOWED" if has_perm else "❌ DENIED"
            debug_log(f"     {doc_id[:8]}... -> {status}")
        
        # Step 5: Filter chunks based on permissions
        debug_log(f"🎯 STEP 6: Filtering chunks by permissions...")
        permitted_chunks = {}
        for chunk_id, chunk_data in chunk_data_map.items():
            doc_id = chunk_to_doc_map[chunk_id]
            if permissions.get(doc_id, False):
                permitted_chunks[chunk_id] = chunk_data
                debug_log(f"   ✅ ALLOWED: {chunk_id[:8]}... ...)")
            else:
                debug_log(f"   ❌ DENIED:  {chunk_id[:8]}... (no permission for doc {doc_id[:8]}...)")
        
        # Final results
        denied_count = len(chunk_data_map) - len(permitted_chunks)
        end_time = time.time()
        duration = end_time - start_time
        
        debug_separator()
        debug_log(f"🎉 COMPLETED get_chunks_by_ids_by_email")
        debug_log(f"   User: {user_email}")
        debug_log(f"   Requested: {len(chunk_ids)} chunks")
        debug_log(f"   Retrieved: {len(chunks_data)} chunks from DB")
        debug_log(f"   Permitted: {len(permitted_chunks)} chunks (after permission filter)")
        debug_log(f"   Denied:    {denied_count} chunks")
        debug_log(f"   Total time: {duration:.3f} seconds")
        debug_log(f"   Avg per chunk: {duration/len(chunk_ids)*1000:.1f}ms")
        debug_separator()
        
        if denied_count > 0:
            logging.info(f"User {user_email}: Access denied to {denied_count} chunks due to category restrictions")
        
        logging.info(f"User {user_email}: Retrieved {len(permitted_chunks)}/{len(chunk_ids)} requested chunks in {duration:.3f}s")
        return permitted_chunks
        
    except Exception as e:
        debug_error(f"💥 ERROR in main function: {e}")
        logging.error(f"Error retrieving chunks with permissions for {user_email}: {e}", exc_info=True)
        debug_log(f"🔄 Falling back to Firestore-only mode")
        return _get_chunks_fallback(chunk_ids, user_email)


def _get_user_categories_cached(redis_client, user_email: str) -> Set[str]:
    """Get user categories with Redis caching"""
    cache_key = f"user_categories:{user_email}"
    debug_log(f"   🔍 Looking for user categories in Redis: {cache_key}")
    
    try:
        # Try Redis first
        cached_categories = redis_client.get(cache_key)
        if cached_categories:
            categories = set(json.loads(cached_categories))
            debug_log(f"   🎯 REDIS HIT: Found cached categories: {list(categories)}")
            return categories
        
        debug_log(f"   ❌ REDIS MISS: Categories not in cache, querying Firestore...")
        
        # Fallback to Firestore - query by email field instead of document ID
        user_query = users_ref.where("email", "==", user_email.lower()).limit(1).get()
        if not user_query:
            debug_log(f"   ❌ User {user_email} not found in Firestore")
            logging.warning(f"User {user_email} not found")
            return set()
        
        user_doc = user_query[0]
        user_data = user_doc.to_dict()
        debug_log(f"   📋 User data retrieved from Firestore")
        debug_log(f"   📋 User fields: {list(user_data.keys())}")
        
        # Get accessible_categories field or empty list if not set
        categories = set(user_data.get("accessible_categories", []))
        debug_log(f"   📂 User accessible_categories: {list(categories)}")
        
        # Cache for 24 hours (86400 seconds)
        redis_client.setex(cache_key, 86400, json.dumps(list(categories)))
        debug_log(f"   💾 Cached categories in Redis for 10 minutes")
        return categories
        
    except Exception as e:
        debug_log(f"   💥 ERROR getting user categories: {e}")
        logging.error(f"Error getting user categories for {user_email}: {e}")
        
        # Fallback to direct Firestore
        debug_log(f"   🔄 Trying direct Firestore fallback...")
        try:
            user_query = users_ref.where("email", "==", user_email.lower()).limit(1).get()
            if user_query:
                user_doc = user_query[0]
                user_data = user_doc.to_dict()
                categories = set(user_data.get("accessible_categories", []))
                debug_log(f"   ✅ Fallback successful: {list(categories)}")
                return categories
        except Exception as e2:
            debug_log(f"   💥 Fallback also failed: {e2}")
        
        debug_log(f"   ❌ Returning empty set")
        return set()


def _get_user_categories_from_api_cached(redis_client, user_email: str, auth_header: Optional[str] = None) -> Set[str]:
    """
    Get user categories from external API with Redis caching.
    
    Args:
        redis_client: Redis client instance
        user_email: User email for cache key
        auth_header: Optional Authorization header. If not provided, tries to get from Flask request context.
    
    Returns:
        Set[str] of category codes accessible to the user
    """
    cache_key = f"user_categories_api:{user_email}"
    debug_log(f"   🔍 Looking for user categories from API in Redis: {cache_key}")
    
    try:
        # Try Redis first
        cached_categories = redis_client.get(cache_key)
        if cached_categories:
            categories = set(json.loads(cached_categories))
            debug_log(f"   🎯 REDIS HIT: Found cached categories from API: {list(categories)}")
            return categories
        
        debug_log(f"   ❌ REDIS MISS: Categories not in cache, calling external API...")
        
        # Get auth header if not provided
        if not auth_header and FLASK_REQUEST_AVAILABLE:
            try:
                auth_header = request.headers.get('Authorization', '') if request else ''
            except RuntimeError:
                # Flask request context not available
                auth_header = ''
        
        if not auth_header:
            debug_log(f"   ❌ No Authorization header available for API call")
            logger.warning(f"No Authorization header available for API call for user {user_email}")
            return set()
        
        # Get the API URL from config
        api_url = getattr(config, 'ACCESSIBLE_DOCS_API_URL', '')
        if not api_url:
            debug_log(f"   ❌ ACCESSIBLE_DOCS_API_URL not configured")
            logger.warning("ACCESSIBLE_DOCS_API_URL not configured")
            return set()
        
        # Prepare headers with the Authorization header
        headers = {
            "Content-Type": "application/json",
            "Authorization": auth_header
        }
        
        debug_log(f"   📡 Calling external API: {api_url}")
        logger.info(f"Fetching accessible categories from API for user {user_email}: {api_url}")
        response = requests.get(api_url, headers=headers, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        debug_log(f"   📋 API response received")
        
        # Extract category codes from the response
        # API response format: {"categories": [{"code": "...", ...}, ...]}
        categories_list = data.get('categories', [])
        if not categories_list:
            debug_log(f"   ❌ No categories returned from API")
            logger.info(f"No accessible categories returned from API for user {user_email}")
            # Cache empty result for 5 minutes to avoid repeated API calls
            redis_client.setex(cache_key, 300, json.dumps([]))
            return set()
        
        # Extract category codes from the categories list
        categories = set()
        for category in categories_list:
            category_code = category.get('code')
            if category_code:
                categories.add(category_code)
        
        debug_log(f"   📂 User accessible_categories from API: {list(categories)}")
        logger.info(f"Successfully fetched {len(categories)} accessible categories from API for user {user_email}")
        
        # Cache for 24 hours (86400 seconds)
        redis_client.setex(cache_key, 86400, json.dumps(list(categories)))
        debug_log(f"   💾 Cached categories from API in Redis for 10 minutes")
        return categories
        
    except requests.exceptions.RequestException as e:
        debug_log(f"   💥 ERROR calling API: {e}")
        logger.error(f"Failed to fetch accessible categories from API for user {user_email}: {e}")
        
        # Fallback to Firestore if API fails
        debug_log(f"   🔄 Falling back to Firestore...")
        try:
            user_query = users_ref.where("email", "==", user_email.lower()).limit(1).get()
            if user_query:
                user_doc = user_query[0]
                user_data = user_doc.to_dict()
                categories = set(user_data.get("accessible_categories", []))
                debug_log(f"   ✅ Fallback to Firestore successful: {list(categories)}")
                return categories
        except Exception as e2:
            debug_log(f"   💥 Fallback to Firestore also failed: {e2}")
        
        debug_log(f"   ❌ Returning empty set")
        return set()
        
    except Exception as e:
        debug_log(f"   💥 ERROR getting user categories from API: {e}")
        logger.error(f"Error getting user categories from API for {user_email}: {e}", exc_info=True)
        
        # Fallback to Firestore
        debug_log(f"   🔄 Trying direct Firestore fallback...")
        try:
            user_query = users_ref.where("email", "==", user_email.lower()).limit(1).get()
            if user_query:
                user_doc = user_query[0]
                user_data = user_doc.to_dict()
                categories = set(user_data.get("accessible_categories", []))
                debug_log(f"   ✅ Fallback successful: {list(categories)}")
                return categories
        except Exception as e2:
            debug_log(f"   💥 Fallback also failed: {e2}")
        
        debug_log(f"   ❌ Returning empty set")
        return set()


def _get_chunks_batch_cached(redis_client, chunk_ids: List[str]) -> Dict[str, dict]:
    """Batch get chunks with Redis caching"""
    debug_log(f"   🔍 Checking Redis cache for {len(chunk_ids)} chunks...")
    chunks_data = {}
    uncached_ids = []
    
    try:
        # Step 1: Try Redis for all chunks using pipeline
        debug_log(f"   📡 Creating Redis pipeline for batch lookup...")
        pipe = redis_client.pipeline()
        for chunk_id in chunk_ids:
            pipe.get(f"chunk_meta:{chunk_id}")
        
        debug_log(f"   ⚡ Executing Redis pipeline...")
        cached_results = pipe.execute()
        
        cache_hits = 0
        cache_misses = 0
        
        for i, cached_result in enumerate(cached_results):
            chunk_id = chunk_ids[i]
            if cached_result:
                chunks_data[chunk_id] = json.loads(cached_result)
                cache_hits += 1
                debug_log(f"     🎯 CACHE HIT:  {chunk_id[:8]}...")
            else:
                uncached_ids.append(chunk_id)
                cache_misses += 1
                debug_log(f"     ❌ CACHE MISS: {chunk_id[:8]}...")
        
        debug_log(f"   📊 Redis Results: {cache_hits} hits, {cache_misses} misses")
        
        # Step 2: Fetch uncached chunks from Firestore
        if uncached_ids:
            debug_log(f"   🗃️  Fetching {len(uncached_ids)} chunks from Firestore...")
            firestore_chunks = {}
            
            # Batch fetch from Firestore (30 at a time due to Firestore limit)
            batches = [uncached_ids[i:i + 30] for i in range(0, len(uncached_ids), 30)]
            debug_log(f"   📦 Processing {len(batches)} Firestore batches...")
            
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            def process_firestore_batch(batch_ids, batch_idx, total_batches):
                try:
                    debug_log(f"     🔄 Processing batch {batch_idx + 1}/{total_batches} ({len(batch_ids)} chunks)...")
                    chunk_refs = [chunks_ref.document(chunk_id) for chunk_id in batch_ids]
                    chunk_docs = db.get_all(chunk_refs)
                    
                    found_chunks = {}
                    found_count = 0
                    for chunk_doc in chunk_docs:
                        if chunk_doc.exists:
                            found_chunks[chunk_doc.id] = chunk_doc.to_dict()
                            found_count += 1
                    return found_chunks
                except Exception as e:
                    debug_log(f"     💥 Error processing batch {batch_idx + 1}: {e}")
                    logger.error(f"Error processing batch {batch_idx + 1}: {e}")
                    return {}

            firestore_chunks = {}
            with ThreadPoolExecutor(max_workers=5) as executor:
                future_to_batch = {
                    executor.submit(process_firestore_batch, batch, i, len(batches)): batch 
                    for i, batch in enumerate(batches)
                }
                
                for future in as_completed(future_to_batch):
                    try:
                        batch_result = future.result()
                        firestore_chunks.update(batch_result)
                    except Exception as e:
                        logger.error(f"Batch future failed: {e}")
            
            debug_log(f"   📊 Firestore total: {len(firestore_chunks)} chunks retrieved")
            
            # Cache new chunks in Redis using pipeline
            if firestore_chunks:
                debug_log(f"   💾 Caching {len(firestore_chunks)} new chunks in Redis...")
                pipe = redis_client.pipeline()
                for chunk_id, chunk_data in firestore_chunks.items():
                    pipe.setex(f"chunk_meta:{chunk_id}", 86400, json.dumps(chunk_data))  # 24 hours
                    debug_log(f"     💾 Caching: {chunk_id[:8]}... for 1 hour")
                
                pipe.execute()
                debug_log(f"   ✅ Successfully cached {len(firestore_chunks)} chunks")
                
                chunks_data.update(firestore_chunks)
        else:
            debug_log(f"   🎯 All chunks found in cache!")
        
        debug_log(f"   📊 Final result: {len(chunks_data)} chunks available")
        return chunks_data
        
    except Exception as e:
        debug_log(f"   💥 ERROR in batch chunk retrieval: {e}")
        logging.error(f"Error in batch chunk retrieval: {e}")
        debug_log(f"   🔄 Falling back to direct Firestore...")
        return _get_chunks_from_firestore_only(chunk_ids)


def _get_permissions_batch_cached(redis_client, user_email: str, user_categories: Set[str], doc_ids: List[str]) -> Dict[str, bool]:
    """Batch check permissions with Redis caching"""
    debug_log(f"   🔒 Checking permissions for {len(doc_ids)} documents...")
    debug_log(f"   👤 User categories: {list(user_categories)}")
    
    permissions = {}
    uncached_docs = []
    
    try:
        # Step 1: Check Redis for cached permissions using pipeline
        debug_log(f"   🔍 Checking Redis for cached permissions...")
        pipe = redis_client.pipeline()
        for doc_id in doc_ids:
            cache_key = f"user_perm:{user_email}:{doc_id}"
            pipe.get(cache_key)
            debug_log(f"     🔍 Checking: {cache_key}")
        
        cached_perms = pipe.execute()
        
        perm_hits = 0
        perm_misses = 0
        
        for i, cached_perm in enumerate(cached_perms):
            doc_id = doc_ids[i]
            if cached_perm is not None:
                has_permission = cached_perm == "1"
                permissions[doc_id] = has_permission
                perm_hits += 1
                status = "ALLOWED" if has_permission else "DENIED"
                debug_log(f"     🎯 PERM CACHE HIT:  {doc_id[:8]}... -> {status}")
            else:
                uncached_docs.append(doc_id)
                perm_misses += 1
                debug_log(f"     ❌ PERM CACHE MISS: {doc_id[:8]}...")
        
        debug_log(f"   📊 Permission cache: {perm_hits} hits, {perm_misses} misses")
        
        # Step 2: Get document categories for uncached permissions
        if uncached_docs:
            debug_log(f"   📂 Need to check categories for {len(uncached_docs)} documents...")
            doc_categories_map = _get_doc_categories_batch_cached(redis_client, uncached_docs)
            
            debug_log(f"   🧮 Calculating permissions...")
            # Calculate and cache new permissions using pipeline
            pipe = redis_client.pipeline()
            for doc_id in uncached_docs:
                doc_categories = doc_categories_map.get(doc_id, set())
                has_permission = bool(user_categories.intersection(doc_categories))
                permissions[doc_id] = has_permission
                
                debug_log(f"     📊 Document {doc_id[:8]}...:")
                debug_log(f"       📂 Doc categories: {list(doc_categories)}")
                debug_log(f"       👤 User categories: {list(user_categories)}")
                debug_log(f"       🔗 Intersection: {list(user_categories.intersection(doc_categories))}")
                debug_log(f"       ✅ Permission: {has_permission}")
                
                # Cache permission for 5 minutes
                cache_key = f"user_perm:{user_email}:{doc_id}"
                cache_value = "1" if has_permission else "0"
                pipe.setex(cache_key, 86400, cache_value)
                debug_log(f"       💾 Caching permission: {cache_key} = {cache_value} (5 min)")
            
            pipe.execute()
            debug_log(f"   ✅ Cached {len(uncached_docs)} new permissions")
        else:
            debug_log(f"   🎯 All permissions found in cache!")
        
        allowed_count = sum(1 for p in permissions.values() if p)
        denied_count = len(permissions) - allowed_count
        debug_log(f"   📊 Permission summary: {allowed_count} allowed, {denied_count} denied")
        
        return permissions
        
    except Exception as e:
        debug_log(f"   💥 ERROR checking permissions: {e}")
        logging.error(f"Error checking permissions: {e}")
        debug_log(f"   🔄 Falling back to direct calculation...")
        return _calculate_permissions_direct(user_categories, doc_ids)


def _get_doc_categories_batch_cached(redis_client, doc_ids: List[str]) -> Dict[str, Set[str]]:
    """Batch get document categories with Redis caching"""
    debug_log(f"     📂 Getting categories for {len(doc_ids)} documents...")
    doc_categories = {}
    uncached_docs = []
    
    try:
        # Step 1: Try Redis first using pipeline
        debug_log(f"     🔍 Checking Redis for document categories...")
        pipe = redis_client.pipeline()
        for doc_id in doc_ids:
            pipe.get(f"doc_categories:{doc_id}")
        
        cached_results = pipe.execute()
        
        cat_hits = 0
        cat_misses = 0
        
        for i, cached_result in enumerate(cached_results):
            doc_id = doc_ids[i]
            if cached_result:
                categories = set(json.loads(cached_result))
                doc_categories[doc_id] = categories
                cat_hits += 1
                debug_log(f"       🎯 CAT CACHE HIT:  {doc_id[:8]}... -> {list(categories)}")
            else:
                uncached_docs.append(doc_id)
                cat_misses += 1
                debug_log(f"       ❌ CAT CACHE MISS: {doc_id[:8]}...")
        
        debug_log(f"     📊 Category cache: {cat_hits} hits, {cat_misses} misses")
        
        # Step 2: Fetch uncached from Firestore
        if uncached_docs:
            debug_log(f"     🗃️  Fetching categories from Firestore for {len(uncached_docs)} documents...")
            
            # Process in batches of 30
            batches = [uncached_docs[i:i + 30] for i in range(0, len(uncached_docs), 30)]
            debug_log(f"     📦 Processing {len(batches)} Firestore batches...")
            
            from concurrent.futures import ThreadPoolExecutor, as_completed

            def process_doc_category_batch(batch_ids, batch_idx, total_batches):
                try:
                    debug_log(f"       🔄 Processing batch {batch_idx + 1}/{total_batches} ({len(batch_ids)} docs)...")
                    doc_refs = [docs_ref.document(doc_id) for doc_id in batch_ids]
                    doc_snapshots = db.get_all(doc_refs)
                    
                    found_docs = {}
                    found_count = 0
                    
                    for doc_snapshot in doc_snapshots:
                        if doc_snapshot.exists:
                            d_data = doc_snapshot.to_dict()
                            found_docs[doc_snapshot.id] = set(d_data.get("categories") or [])
                            found_count += 1
                            debug_log(f"         ✅ Found: {doc_snapshot.id[:8]}... -> {list(found_docs[doc_snapshot.id])}")
                        else:
                            found_docs[doc_snapshot.id] = set()
                            debug_log(f"         ❌ Missing: {doc_snapshot.id[:8]}... -> []")
                    
                    debug_log(f"       📊 Batch {batch_idx + 1} results: {found_count}/{len(batch_ids)} found")
                    return found_docs
                except Exception as e:
                    debug_log(f"       💥 Error processing doc batch {batch_idx + 1}: {e}")
                    logger.error(f"Error processing doc batch {batch_idx + 1}: {e}")
                    return {}

            with ThreadPoolExecutor(max_workers=5) as executor:
                future_to_batch = {
                    executor.submit(process_doc_category_batch, batch, i, len(batches)): batch 
                    for i, batch in enumerate(batches)
                }
                
                pipe = redis_client.pipeline()
                for future in as_completed(future_to_batch):
                    try:
                        batch_result = future.result()
                        doc_categories.update(batch_result)
                        
                        # Cache in Redis pipeline
                        for doc_id, cats in batch_result.items():
                             pipe.setex(f"doc_categories:{doc_id}", 86400, json.dumps(list(cats)))
                    except Exception as e:
                        logger.error(f"Doc category batch future failed: {e}")
                
                pipe.execute() # Execute all cache sets
            
            debug_log(f"     ✅ Successfully processed all document categories")
        else:
            debug_log(f"     🎯 All document categories found in cache!")
        
        debug_log(f"     📊 Final categories result: {len(doc_categories)} documents processed")
        return doc_categories
        
    except Exception as e:
        debug_log(f"     💥 ERROR getting document categories: {e}")
        logging.error(f"Error getting document categories: {e}")
        debug_log(f"     🔄 Falling back to direct Firestore...")
        return _get_doc_categories_from_firestore_only(doc_ids)


# Fallback functions for Redis failures
def _get_chunks_fallback(chunk_ids: list[str], user_email: str) -> dict:
    """Fallback to original Firestore-only implementation"""
    debug_log(f"🔄 FALLBACK MODE: Using Firestore-only for user {user_email}")
    debug_log(f"   📦 Processing {len(chunk_ids)} chunks without Redis...")
    
    chunk_data_map = {}
    if not chunk_ids:
        return chunk_data_map

    try:
        # Get user's accessible categories - query by email field
        debug_log(f"   👤 Getting user data from Firestore...")
        user_query = users_ref.where("email", "==", user_email.lower()).limit(1).get()
        if not user_query:
            debug_log(f"   ❌ User {user_email} not found in Firestore")
            logging.warning(f"User {user_email} not found")
            return {}
        
        user_doc = user_query[0]
        user_data = user_doc.to_dict()
        user_accessible_categories = set(user_data.get("accessible_categories", []))
        debug_log(f"   👤 User categories: {list(user_accessible_categories)}")
        
        if not user_accessible_categories:
            debug_log(f"   ❌ User {user_email} has no accessible categories")
            logging.warning(f"User {user_email} has no accessible categories")     
            return {}

        # Firestore 'in' query limit is 30
        chunk_id_batches = [chunk_ids[i:i + 30] for i in range(0, len(chunk_ids), 30)]
        debug_log(f"   📦 Processing {len(chunk_id_batches)} chunk batches...")
        
        doc_ids_to_fetch = set()
        chunks_to_check = {}
        
        for batch_num, batch_ids in enumerate(chunk_id_batches):
            if not batch_ids:
                continue
            
            debug_log(f"     🔄 Processing chunk batch {batch_num + 1}/{len(chunk_id_batches)} ({len(batch_ids)} chunks)...")
            query = chunks_ref.where("__name__", 'in', batch_ids)
            docs = query.stream()
            
            batch_found = 0
            for doc in docs:
                if doc.exists:
                    data = doc.to_dict()
                    original_doc_id = data.get("original_doc_firestore_id")
                    if original_doc_id:
                        doc_ids_to_fetch.add(original_doc_id)
                        chunks_to_check[doc.id] = {
                            "data": data,
                            "original_doc_id": original_doc_id
                        }
                        batch_found += 1
                        debug_log(f"       ✅ Chunk: {doc.id[:8]}... -> Doc: {original_doc_id[:8]}...")
            
            debug_log(f"     📊 Chunk batch {batch_num + 1} results: {batch_found}/{len(batch_ids)} found")

        debug_log(f"   📊 Total unique documents needed: {len(doc_ids_to_fetch)}")

        if not doc_ids_to_fetch:
            debug_log(f"   ❌ No valid document references found")
            return {}

        # Batch fetch document metadata
        debug_log(f"   📂 Fetching document metadata for {len(doc_ids_to_fetch)} documents...")
        doc_categories_map = {}
        doc_id_list = list(doc_ids_to_fetch)
        doc_id_batches = [doc_id_list[i:i + 30] for i in range(0, len(doc_id_list), 30)]
        
        for batch_num, batch_doc_ids in enumerate(doc_id_batches):
            if not batch_doc_ids:
                continue
            
            debug_log(f"     🔄 Processing doc batch {batch_num + 1}/{len(doc_id_batches)} ({len(batch_doc_ids)} docs)...")
            doc_refs = [docs_ref.document(doc_id) for doc_id in batch_doc_ids]
            doc_snapshots = db.get_all(doc_refs)
            
            batch_found = 0
            for doc_snapshot in doc_snapshots:
                if doc_snapshot.exists:
                    doc_data = doc_snapshot.to_dict()
                    doc_categories = set(doc_data.get("categories", []))
                    doc_categories_map[doc_snapshot.id] = doc_categories
                    batch_found += 1
                    debug_log(f"       ✅ Doc: {doc_snapshot.id[:8]}... -> Categories: {list(doc_categories)}")
            
            debug_log(f"     📊 Doc batch {batch_num + 1} results: {batch_found}/{len(batch_doc_ids)} found")

        # Filter chunks based on permission
        debug_log(f"   🔒 Filtering chunks by permissions...")
        allowed_count = 0
        denied_count = 0
        
        for chunk_id, chunk_info in chunks_to_check.items():
            original_doc_id = chunk_info["original_doc_id"]
            chunk_data = chunk_info["data"]
            
            doc_categories = doc_categories_map.get(original_doc_id, set())
            has_permission = bool(user_accessible_categories.intersection(doc_categories))
            
            debug_log(f"     🔍 Chunk {chunk_id[:8]}... -> Doc {original_doc_id[:8]}...")
            debug_log(f"       📂 Doc categories: {list(doc_categories)}")
            debug_log(f"       🔗 Intersection: {list(user_accessible_categories.intersection(doc_categories))}")
            debug_log(f"       ✅ Permission: {has_permission}")
            
            if has_permission:
                chunk_data_map[chunk_id] = {
                    "chunk_id": chunk_id,
                    "original_filename": chunk_data.get("original_filename"),
                    "start_page": chunk_data.get("start_page"),
                    "end_page": chunk_data.get("end_page"),
                    "ocr_text_preview": chunk_data.get("extracted_text_preview") or chunk_data.get("ocr_text_preview"),
                    "original_doc_firestore_id": original_doc_id,
                    "extracted_entities": chunk_data.get("extracted_entities", {}), # For backward compatibility
                    "entities": chunk_data.get("entities", []) # Flattened entities array
                }
                allowed_count += 1
            else:
                denied_count += 1

        debug_log(f"🔄 FALLBACK COMPLETED")
        debug_log(f"   📊 Results: {allowed_count} allowed, {denied_count} denied")
        debug_log(f"   ✅ Returning {len(chunk_data_map)} chunks")

        return chunk_data_map

    except Exception as e:
        debug_log(f"   💥 ERROR in fallback implementation: {e}")
        logging.error(f"Error in fallback implementation: {e}", exc_info=True)
        return {}


def _get_chunks_from_firestore_only(chunk_ids: List[str]) -> Dict[str, dict]:
    """Direct Firestore fetch without caching"""
    debug_log(f"     🗃️  Direct Firestore fetch for {len(chunk_ids)} chunks...")
    chunks_data = {}
    
    try:
        batches = [chunk_ids[i:i + 30] for i in range(0, len(chunk_ids), 30)]
        debug_log(f"     📦 Processing {len(batches)} batches...")
        
        for batch_num, batch_ids in enumerate(batches):
            debug_log(f"       🔄 Batch {batch_num + 1}/{len(batches)} ({len(batch_ids)} chunks)...")
            chunk_refs = [chunks_ref.document(chunk_id) for chunk_id in batch_ids]
            chunk_docs = db.get_all(chunk_refs)
            
            batch_found = 0
            for chunk_doc in chunk_docs:
                if chunk_doc.exists:
                    chunks_data[chunk_doc.id] = chunk_doc.to_dict()
                    batch_found += 1
            
            debug_log(f"       📊 Batch {batch_num + 1} results: {batch_found}/{len(batch_ids)} found")
        
        debug_log(f"     ✅ Direct fetch completed: {len(chunks_data)} chunks")
    except Exception as e:
        debug_log(f"     💥 ERROR in direct fetch: {e}")
        logging.error(f"Error fetching chunks from Firestore: {e}")
    
    return chunks_data


def _calculate_permissions_direct(user_categories: Set[str], doc_ids: List[str]) -> Dict[str, bool]:
    """Direct permission calculation without caching"""
    debug_log(f"     🧮 Direct permission calculation for {len(doc_ids)} documents...")
    permissions = {}
    
    try:
        doc_categories_map = _get_doc_categories_from_firestore_only(doc_ids)
        for doc_id in doc_ids:
            doc_categories = doc_categories_map.get(doc_id, set())
            has_permission = bool(user_categories.intersection(doc_categories))
            permissions[doc_id] = has_permission
            debug_log(f"       {doc_id[:8]}... -> {has_permission} (categories: {list(doc_categories)})")
        
        debug_log(f"     ✅ Direct calculation completed")
    except Exception as e:
        debug_log(f"     💥 ERROR in direct calculation: {e}")
        logging.error(f"Error calculating permissions: {e}")
    
    return permissions


def _get_doc_categories_from_firestore_only(doc_ids: List[str]) -> Dict[str, Set[str]]:
    """Direct Firestore fetch for document categories"""
    debug_log(f"       📂 Direct fetch of categories for {len(doc_ids)} documents...")
    doc_categories = {}
    
    try:
        batches = [doc_ids[i:i + 30] for i in range(0, len(doc_ids), 30)]
        debug_log(f"       📦 Processing {len(batches)} batches...")
        
        for batch_num, batch_doc_ids in enumerate(batches):
            debug_log(f"         🔄 Batch {batch_num + 1}/{len(batches)} ({len(batch_doc_ids)} docs)...")
            doc_refs = [docs_ref.document(doc_id) for doc_id in batch_doc_ids]
            doc_snapshots = db.get_all(doc_refs)
            
            batch_found = 0
            for doc_snapshot in doc_snapshots:
                if doc_snapshot.exists:
                    categories = set(doc_snapshot.get("categories") or [])
                    doc_categories[doc_snapshot.id] = categories
                    batch_found += 1
                    debug_log(f"           ✅ {doc_snapshot.id[:8]}... -> {list(categories)}")
                else:
                    doc_categories[doc_snapshot.id] = set()
                    debug_log(f"           ❌ {doc_snapshot.id[:8]}... -> [] (not found)")
            
            debug_log(f"         📊 Batch {batch_num + 1} results: {batch_found}/{len(batch_doc_ids)} found")
        
        debug_log(f"       ✅ Direct category fetch completed")
    except Exception as e:
        debug_log(f"       💥 ERROR in direct category fetch: {e}")
        logging.error(f"Error fetching document categories: {e}")
    
    return doc_categories


# Optional: Cache invalidation functions for when data changes
def invalidate_user_cache(user_email: str):
    """Call this when user's accessible_categories change"""
    debug_log(f"🧹 INVALIDATING USER CACHE for {user_email}")
    
    try:
        redis_client = get_redis_client()
        
        # Clear user categories cache
        user_cat_key = f"user_categories:{user_email}"
        result1 = redis_client.delete(user_cat_key)
        debug_log(f"   🗑️  Deleted user categories cache: {user_cat_key} (result: {result1})")
        
        # Clear user permission caches
        pattern = f"user_perm:{user_email}:*"
        debug_log(f"   🔍 Scanning for permission keys: {pattern}")
        
        deleted_perms = 0
        for key in redis_client.scan_iter(match=pattern):
            redis_client.delete(key)
            deleted_perms += 1
            debug_log(f"     🗑️  Deleted: {key}")
        
        debug_log(f"   ✅ Cache invalidation completed: {deleted_perms} permission keys deleted")
        logging.info(f"Invalidated cache for user {user_email}")
        
    except Exception as e:
        debug_log(f"   💥 ERROR invalidating user cache: {e}")
        logging.error(f"Error invalidating user cache: {e}")


def invalidate_document_cache(doc_id: str):
    """Call this when document categories change"""
    debug_log(f"🧹 INVALIDATING DOCUMENT CACHE for {doc_id}")
    
    try:
        redis_client = get_redis_client()
        
        # Clear document categories cache
        doc_cat_key = f"doc_categories:{doc_id}"
        result1 = redis_client.delete(doc_cat_key)
        debug_log(f"   🗑️  Deleted document categories cache: {doc_cat_key} (result: {result1})")
        
        # Clear related permission caches
        pattern = f"user_perm:*:{doc_id}"
        debug_log(f"   🔍 Scanning for permission keys: {pattern}")
        
        deleted_perms = 0
        for key in redis_client.scan_iter(match=pattern):
            redis_client.delete(key)
            deleted_perms += 1
            debug_log(f"     🗑️  Deleted: {key}")
        
        debug_log(f"   ✅ Cache invalidation completed: {deleted_perms} permission keys deleted")
        logging.info(f"Invalidated cache for document {doc_id}")
        
    except Exception as e:
        debug_log(f"   💥 ERROR invalidating document cache: {e}")
        logging.error(f"Error invalidating document cache: {e}")
# =======================================================




def save_chat_message(session_id: str, user_email: str, query: str, answer: str, references: Optional[list] = None, message_id: Optional[str] = None, response_time: Optional[str] = None, reference_time: Optional[str] = None, text_response_time: Optional[str] = None, metadata: Optional[dict] = None):
    """
    Saves a user query and AI answer (with optional references, message_id, response_time, and metadata) to the chat history.
    If session_id is None or empty, a new session is created.
    Returns the session_id (new or existing) and any error.
    """
    try:
        user_message_id = message_id if message_id else uuid.uuid4().hex
        ai_message_id = uuid.uuid4().hex # AI response gets its own unique ID within the message pair

        ai_message_payload = {
            'id': ai_message_id,
            'sender': 'ai',
            'text': answer,
            'timestamp': datetime.datetime.now(tz=datetime.timezone.utc),
            'references': references or []
        }
        if response_time:
            ai_message_payload['responseTime'] = response_time
        if reference_time:
            ai_message_payload['referenceTime'] = reference_time
        if text_response_time:
            ai_message_payload['textResponseTime'] = text_response_time
        if metadata:
            ai_message_payload['metadata'] = metadata

        if not session_id:
            session_ref = sessions_ref.document()
            session_id = session_ref.id
            now = datetime.datetime.now(tz=datetime.timezone.utc)
            session_data = {
                'user_email': user_email,
                'created_at': now,
                'last_updated': now,
                'title': query[:75] + "..." if query and len(query) > 75 else query or "New Chat Session",
                'messages': [
                    {'id': user_message_id, 'sender': 'user', 'text': query, 'timestamp': now},
                    ai_message_payload
                ]
            }
            session_ref.set(session_data)
            logger.info(f"Created new chat session {session_id} with initial message_id {user_message_id}")
            return session_id, None
        else:
            session_ref = sessions_ref.document(session_id)
            now = datetime.datetime.now(tz=datetime.timezone.utc)
            
            user_message_data = {'id': user_message_id, 'sender': 'user', 'text': query, 'timestamp': now}
            
            session_ref.update({
                'messages': firestore.ArrayUnion([user_message_data, ai_message_payload]),
                'last_updated': now,
                'last_message_preview': answer[:100] + "..." if answer else ""
            })
            logger.info(f"Appended messages (user_msg_id: {user_message_id}, ai_msg_id: {ai_message_id}) to chat session {session_id}")
            return session_id, None

    except Exception as e:
        logger.error(f"ERROR: Failed to save chat message for session {session_id}, message_id {message_id}: {e}", exc_info=True)
        return session_id, f"Failed to save chat message: {e}"

def get_upload_history(
    limit: int = 10, 
    start_after_doc_id: Optional[str] = None, 
    search_term: Optional[str] = None, 
    status_filter: Optional[Union[str, List[str]]] = None,  # Updated type hint
    allowed_doc_ids: Optional[List[str]] = None
):
    """
    Retrieves a paginated list of all document metadata history,
    optionally filtered by filename prefix, status, and a list of allowed document IDs.
    ordered by upload time descending.
    Returns history, last_doc_id, total_items, and error.
    """
    debug_log(f"DEBUG: get_upload_history CALLED with limit={limit}, start_after_doc_id='{start_after_doc_id}', search_term='{search_term}', status_filter='{status_filter}'")
    logger.info(f"get_upload_history called for limit: {limit}, start_after: {start_after_doc_id}, search: '{search_term}', status: '{status_filter}'")
    try:
        # STRATEGY: Secure In-Memory Filtering
        # If allowed_doc_ids is provided, we MUST fetch all candidates and filter in memory
        # to ensure accurate 'total_items' count and correct pagination skipping.
        # Firestore COUNT queries cannot handle 'IN' lists > 30 items efficiently.
        if allowed_doc_ids is not None:
            logger.info(f"Processing secure history for {len(allowed_doc_ids)} allowed IDs (In-Memory Filtering)")
            
            # 1. Build Base Query (Status + Search + Sort)
            # We fetch all matching metadata to filter locally.
            # Note: For massive datasets (100k+), this will need a 'user_accessible' subcollection or array.
            base_query = docs_ref.order_by('upload_timestamp', direction=firestore.Query.DESCENDING)

            if search_term:
                base_query = base_query.where('search_keywords', 'array_contains', search_term.lower())
            
            if status_filter:
                if isinstance(status_filter, list) and status_filter:
                    base_query = base_query.where('status', 'in', status_filter)
                elif isinstance(status_filter, str):
                    base_query = base_query.where('status', '==', status_filter)

            # 2. Fetch All Candidates
            debug_log("DEBUG: Executing base_query.stream() for in-memory filtering...")
            all_snapshots = list(base_query.stream())
            
            # 3. Filter & Intersect
            allowed_set = set(allowed_doc_ids)
            filtered_data = []
            
            for doc in all_snapshots:
                if doc.id in allowed_set:
                    data = doc.to_dict()
                    data['id'] = doc.id
                    # Convert timestamps
                    if 'upload_timestamp' in data and hasattr(data['upload_timestamp'], 'isoformat'):
                        data['upload_timestamp'] = data['upload_timestamp'].isoformat()
                    # Remove 'search_keywords' field if it exists
                    data.pop('search_keywords', None)
                    filtered_data.append(data)
            
            total_items = len(filtered_data)
            logger.info(f"In-Memory Filter Result: {total_items} items allowed out of {len(all_snapshots)} candidates.")

            # 4. Paginate (Slice)
            start_index = 0
            if start_after_doc_id:
                for i, item in enumerate(filtered_data):
                    if item['id'] == start_after_doc_id:
                        start_index = i + 1
                        break
            
            page_slice = filtered_data[start_index : start_index + limit]
            last_doc_id = page_slice[-1]['id'] if page_slice else None
            
            return page_slice, last_doc_id, total_items, None

        # --- STANDARD FIRESTORE LOGIC (Superadmin / No Filter) ---
        debug_log("DEBUG: Executing Standard Firestore Query (No allowed_doc_ids filter)...")

        # Base query for counting total items matching filters (excluding pagination)
        debug_log("DEBUG: Building count_query...")
        count_query = docs_ref # Start with the base collection reference
        
        if search_term:
            count_query = count_query.where('search_keywords', 'array_contains', search_term.lower())
            logger.info(f"COUNT_QUERY: Added search_term filter: '{search_term}'")
        
        if status_filter:
            if isinstance(status_filter, list) and status_filter:
                count_query = count_query.where('status', 'in', status_filter)
            elif isinstance(status_filter, str):
                count_query = count_query.where('status', '==', status_filter)

        # Get total count of matching documents
        debug_log("DEBUG: Executing count_query...")
        total_items_snapshot = count_query.count().get()
        total_items = total_items_snapshot[0][0].value if total_items_snapshot else 0
        logger.info(f"COUNT_QUERY: Calculated total_items: {total_items}")


        # Main query for fetching paginated data
        debug_log("DEBUG: Building main_query...")
        query = docs_ref # Start with the base collection reference
        
        if search_term:
            query = query.where('search_keywords', 'array_contains', search_term.lower())
            
        if status_filter:
            if isinstance(status_filter, list) and status_filter:
                query = query.where('status', 'in', status_filter)
            elif isinstance(status_filter, str):
                query = query.where('status', '==', status_filter)
            query = query.order_by('original_filename', direction=firestore.Query.ASCENDING)
            # upload_timestamp is a secondary sort key for consistent pagination within filename matches
            query = query.order_by('upload_timestamp', direction=firestore.Query.DESCENDING)
        elif status_filter: 
            # If only status_filter is active (no search_term), order by timestamp
            query = query.order_by('upload_timestamp', direction=firestore.Query.DESCENDING)
        else: 
            # Default order if no search_term and no status_filter
            query = query.order_by('upload_timestamp', direction=firestore.Query.DESCENDING)
            logger.info("MAIN_QUERY: Ordering by upload_timestamp DESC (default or status_filter only)")
            debug_log(f"DEBUG: MAIN_QUERY after ordering (default/status_filter only) applied.")

        # if search_term:
        #     # We're doing substring search using search_keywords — order by upload_timestamp only
        #     query = query.order_by('upload_timestamp', direction=firestore.Query.DESCENDING)
        #     logger.info("MAIN_QUERY: search_term active — ordering by upload_timestamp only.")
        #     debug_log("DEBUG: MAIN_QUERY after ordering by upload_timestamp (search_term case).")

        # elif status_filter:
        #     query = query.order_by('upload_timestamp', direction=firestore.Query.DESCENDING)
        #     logger.info("MAIN_QUERY: status_filter active — ordering by upload_timestamp.")
        #     debug_log("DEBUG: MAIN_QUERY after ordering by upload_timestamp (status_filter case).")

        # else:
        #     query = query.order_by('upload_timestamp', direction=firestore.Query.DESCENDING)
        #     logger.info("MAIN_QUERY: default ordering by upload_timestamp DESC.")
        #     debug_log("DEBUG: MAIN_QUERY after ordering (default case).")
        
        if start_after_doc_id:
            debug_log(f"DEBUG: Attempting to fetch start_after_doc: {start_after_doc_id}")
            start_after_doc = docs_ref.document(start_after_doc_id).get()
            if start_after_doc.exists:
                query = query.start_after(start_after_doc)
                logger.info(f"MAIN_QUERY: Applied start_after_doc_id: {start_after_doc_id}")
                debug_log(f"DEBUG: MAIN_QUERY after start_after applied.")
            else:
                logger.warning(f"Pagination cursor document ID '{start_after_doc_id}' not found.")
                debug_log(f"DEBUG: Pagination cursor doc ID '{start_after_doc_id}' not found. Returning empty.")
                return [], None, 0, f"Pagination cursor document ID '{start_after_doc_id}' not found."

        # if start_after_doc_id:
        #     debug_log(f"DEBUG: Attempting to fetch start_after_doc: {start_after_doc_id}")
        #     start_after_doc = docs_ref.document(start_after_doc_id).get()
        #     if start_after_doc.exists:
        #         start_data = start_after_doc.to_dict()

        #         if not start_data or 'upload_timestamp' not in start_data:
        #             logger.warning(f"Document '{start_after_doc_id}' missing required field for pagination.")
        #             debug_log(f"DEBUG: Document '{start_after_doc_id}' missing 'upload_timestamp'.")
        #             return [], None, 0, f"Pagination error: document '{start_after_doc_id}' missing upload_timestamp"

        #         # Supply value(s) for start_after based on current order_by fields
        #         query = query.start_after({
        #             'upload_timestamp': start_data['upload_timestamp']
        #         })

        #         logger.info(f"MAIN_QUERY: Applied start_after with upload_timestamp from doc: {start_after_doc_id}")
        #         debug_log("DEBUG: MAIN_QUERY after start_after applied.")
        # else:
        #     logger.warning(f"Pagination cursor document ID '{start_after_doc_id}' not found.")
        #     debug_log(f"DEBUG: Pagination cursor doc ID '{start_after_doc_id}' not found. Returning empty.")
        #     return [], None, 0, f"Pagination cursor document ID '{start_after_doc_id}' not found."
        
        query = query.limit(limit)
        logger.info(f"MAIN_QUERY: Applied limit: {limit}")
        debug_log(f"DEBUG: MAIN_QUERY after limit applied.")
        
        debug_log("DEBUG: Executing main_query.stream()...")
        docs_stream = query.stream()
        docs = list(docs_stream) # Execute the query
        logger.info(f"MAIN_QUERY: Fetched {len(docs)} documents.")
        debug_log(f"DEBUG: MAIN_QUERY fetched {len(docs)} documents.")

        history = []
        last_doc_id = None
        debug_log("DEBUG: Processing fetched documents...")
        for i, doc in enumerate(docs):
            doc_data = doc.to_dict()
            if not doc_data:
                continue
            # if i < 3: # Log first 3 fetched docs for inspection
            #     debug_log(f"DEBUG: Fetched doc {i+1} ID: {doc.id}, Data: {doc_data}")
                
            if 'upload_timestamp' in doc_data and isinstance(doc_data['upload_timestamp'], datetime.datetime):
                doc_data['upload_timestamp'] = doc_data['upload_timestamp'].isoformat()
            doc_data['id'] = doc.id
            # Remove 'search_keywords' field if it exists
            doc_data.pop('search_keywords', None)
            history.append(doc_data)
            last_doc_id = doc.id
        
        if history:
            logger.info(f"Returning {len(history)} history items. First item ID (preview): {history[0]['id']}, Last item ID (for next_cursor): {last_doc_id}")
            debug_log(f"DEBUG: Processed {len(history)} items. First ID: {history[0]['id'] if history else 'N/A'}, Last ID for cursor: {last_doc_id}")
        else:
            logger.info("Returning 0 history items.")
            debug_log("DEBUG: Processed 0 history items.")
        
        debug_log(f"DEBUG: FINAL RETURN: history_len={len(history)}, last_doc_id='{last_doc_id}', total_items={total_items}, error=None")
        return history, last_doc_id, total_items, None

    except Exception as e:
        logger.error(f"ERROR: Failed to retrieve upload history: {e}", exc_info=True) # Removed user_email from log
        debug_log(f"DEBUG: EXCEPTION in get_upload_history: {e}") # Updated function name
        if "query requires an index" in str(e).lower():
             error_msg = "Query requires a Firestore index. Please use the Diagnosis page to create/update indexes. Details: " + str(e)
        else:
             error_msg = f"Failed to retrieve upload history: {e}"
        return [], None, 0, error_msg

def get_user_chat_sessions(user_email: str, limit: int = 100):
    """Retrieves chat session summaries for a specific user, ordered by last_updated time descending."""
    try:
        query = sessions_ref.where('user_email', '==', user_email)\
                            .order_by('created_at', direction=firestore.Query.DESCENDING)\
                            .limit(limit)
        docs = query.stream()

        # for doc in docs:
        #     print("======================================================>>>> DEBUG: Fetched chat session documents: ")
        #     debug_log(f"DEBUG: Fetched chat session document ID: {doc.id}")


        sessions = []
        for doc in docs:
            session_data = doc.to_dict()
            title = session_data.get('title')
            if not title and session_data.get('messages') and len(session_data['messages']) > 0:
                 first_user_message = next((msg.get('text') for msg in session_data['messages'] if msg.get('sender') == 'user'), None)
                 if first_user_message:
                      title = first_user_message[:50] + ('...' if len(first_user_message) > 50 else '')
            if not title:
                 title = "Chat Session"

            last_updated_ts = session_data.get('last_updated', session_data.get('created_at'))
            sessions.append({
                'id': doc.id,
                'title': title,
                'last_updated': last_updated_ts.isoformat() if last_updated_ts else None
            })
        return sessions, None
    except Exception as e:
        logger.error(f"ERROR: Failed to retrieve chat sessions for {user_email}: {e}", exc_info=True)
        if "query requires an index" in str(e).lower():
             error_msg = "Query requires a Firestore index for chat sessions (user_email == ?, last_updated DESC). Please create it."
        else:
             error_msg = f"Failed to retrieve chat sessions: {e}"
        return [], error_msg

def get_session_messages(session_id: str, user_email: str):
    """Retrieves all messages for a specific chat session, verifying user ownership."""
    try:
        session_ref = sessions_ref.document(session_id)
        session_doc = session_ref.get()

        if not session_doc.exists:
            return None, "Chat session not found."

        session_data = session_doc.to_dict()
        if session_data.get('user_email') != user_email:
            return None, "Access denied to this chat session."

        messages = session_data.get('messages', [])
        for msg in messages:
            if 'timestamp' in msg and isinstance(msg['timestamp'], datetime.datetime):
                msg['timestamp'] = msg['timestamp'].isoformat()
        
        # Messages are stored in an array, already ordered by append.
        # If explicit sort is needed: messages.sort(key=lambda x: x.get('timestamp', ''))
        return messages, None
    except Exception as e:
        logger.error(f"ERROR: Failed to retrieve messages for session {session_id}: {e}", exc_info=True)
        return None, f"Failed to retrieve messages: {e}"



def rename_chat_session(session_id: str, user_email: str, new_title: str) -> Tuple[bool, str]:
    """Renames a chat session if the user owns it."""
    if not new_title or len(new_title) > 100:
        return False, "New title is invalid or too long."
    try:
        session_ref = sessions_ref.document(session_id)
        session_doc = session_ref.get()

        if not session_doc.exists:
            return False, "Chat session not found."

        session_data = session_doc.to_dict()
        if session_data.get('user_email') != user_email:
            return False, "Access denied."

        session_ref.update({
            'title': new_title,
            'last_updated': datetime.datetime.now(tz=datetime.timezone.utc)
        })
        logger.info(f"Renamed session {session_id} for user {user_email}")
        return True, "Session renamed successfully."
    except Exception as e:
        logger.error(f"Error renaming session {session_id}: {e}", exc_info=True)
        return False, f"Failed to rename session: {e}"

def delete_chat_session(session_id: str, user_email: str) -> Tuple[bool, str]:
    """Deletes a chat session if the user owns it."""
    try:
        session_ref = sessions_ref.document(session_id)
        session_doc = session_ref.get()

        if not session_doc.exists:
            return True, "Chat session not found or already deleted."

        session_data = session_doc.to_dict()
        if session_data.get('user_email') != user_email:
            return False, "Access denied."

        session_ref.delete()
        logger.info(f"Deleted session {session_id} for user {user_email}")
        return True, "Session deleted successfully."
    except Exception as e:
        logger.error(f"Error deleting session {session_id}: {e}", exc_info=True)
        return False, f"Failed to delete session: {e}"

def get_chunks_for_document(original_doc_id: str, limit: int = 20, start_after_chunk_id: Optional[str] = None) -> Tuple[list, Optional[str], Optional[str]]:
    """
    Retrieves a paginated list of chunk documents associated with a specific original document ID.
    """
    try:
        query = chunks_ref.where('original_doc_firestore_id', '==', original_doc_id)\
                          .order_by('start_page', direction=firestore.Query.ASCENDING)

        if start_after_chunk_id:
            start_after_doc = chunks_ref.document(start_after_chunk_id).get()
            if start_after_doc.exists:
                query = query.start_after(start_after_doc)
            else:
                logger.warning(f"Pagination cursor chunk ID '{start_after_chunk_id}' not found for doc {original_doc_id}.")
                return [], None, f"Pagination cursor chunk ID '{start_after_chunk_id}' not found."

        query = query.limit(limit)
        docs = list(query.stream())

        chunks_list = []
        last_chunk_id = None
        for doc in docs:
            chunk_data = doc.to_dict()
            chunk_data['id'] = doc.id
            if 'created_at' in chunk_data and isinstance(chunk_data['created_at'], datetime.datetime):
                 chunk_data['created_at'] = chunk_data['created_at'].isoformat()
            if 'last_updated' in chunk_data and isinstance(chunk_data['last_updated'], datetime.datetime):
                 chunk_data['last_updated'] = chunk_data['last_updated'].isoformat()

            selected_data = {
                'id': doc.id,
                'chunk_id': chunk_data.get('chunk_id'),
                'original_doc_firestore_id': chunk_data.get('original_doc_firestore_id'),
                'start_page': chunk_data.get('start_page'),
                'end_page': chunk_data.get('end_page'),
                'status': chunk_data.get('status'),
                'status_message': chunk_data.get('status_message'),
                'created_at': chunk_data.get('created_at'),
                'ocr_confidence_score': chunk_data.get('ocr_confidence_score'),
                'has_embedding': chunk_data.get('has_embedding'),
                'extraction_method': chunk_data.get('extracted_entities', {}).get('_extraction_method') or chunk_data.get('extracted_entities', {}).get('extraction_method'),
                # Fetch actual Firestore field names
                'classified_document_type': chunk_data.get('classified_document_type'), 
                'selected_parser_processor_id': chunk_data.get('selected_parser_processor_id')
            }
            # Note: The mapping to frontend-expected names ('classified_chunk_document_type_label', 'used_parser_processor_id')
            # is handled in document_routes.py before sending the response.
            chunks_list.append(selected_data)
            last_chunk_id = doc.id

        return chunks_list, last_chunk_id, None

    except Exception as e:
        logger.error(f"Error retrieving chunks for document {original_doc_id}: {e}", exc_info=True)
        if "query requires an index" in str(e).lower():
             error_msg = "Query requires a Firestore index for chunks (original_doc_firestore_id == ?, start_page ASC). Please create it."
        else:
             error_msg = f"Failed to retrieve chunks: {e}"
        return [], None, error_msg

def get_all_chunk_ids() -> Tuple[list[str], Optional[str]]:
    """
    Retrieves all document IDs from the 'document_chunks' collection.
    """
    all_ids = []
    try:
        logger.info("Attempting to retrieve all chunk IDs from Firestore...")
        docs_stream = chunks_ref.select([]).stream()
        for doc in docs_stream:
            all_ids.append(doc.id)
        logger.info(f"Successfully retrieved {len(all_ids)} chunk IDs.")
        return all_ids, None
    except Exception as e:
        logger.error(f"Error retrieving all chunk IDs: {e}", exc_info=True)
        return [], f"Failed to retrieve all chunk IDs: {e}"

def get_processing_stats(doc_id: str) -> Tuple[Optional[dict], Optional[str]]:
    """
    Aggregates processing statistics for a document's chunks.
    Counts total chunks, Gemini-processed chunks, and DocAI-processed chunks.
    """
    try:
        logger.info(f"Aggregating processing stats for doc {doc_id}")
        
        # Get parent document status
        doc_snap = docs_ref.document(doc_id).get()
        doc_status = "unknown"
        if doc_snap.exists:
            doc_status = doc_snap.to_dict().get('status', 'unknown')

        chunks_stream = chunks_ref.where('original_doc_firestore_id', '==', doc_id).stream()
        
        total_chunks = 0
        gemini_chunks = 0
        docai_chunks = 0
        
        # Define active statuses consistently with global stats
        active_statuses = ['completed', 'pending_embedding', 'processing_embedding']
        
        # Base query for this document's active chunks
        base_query = chunks_ref.where('original_doc_firestore_id', '==', doc_id).where('status', 'in', active_statuses)
        
        # 1. Total Count
        total_res = base_query.count().get()
        total_chunks = total_res[0][0].value
        
        # 2. LLM Count
        llm_res = base_query.where('extraction_source', '==', 'llm').count().get()
        gemini_chunks = llm_res[0][0].value
        
        # 3. Legacy Count
        legacy_res = base_query.where('extraction_source', '==', 'legacy').count().get()
        docai_chunks = legacy_res[0][0].value
        
        stats = {
            "total_chunks": int(total_chunks),
            "gemini_chunks": int(gemini_chunks),
            "docai_chunks": int(docai_chunks),
            "status": doc_status
        }
        
        logger.info(f"Processing stats for doc {doc_id}: {stats}")
        return stats, None
        
    except Exception as e:
        logger.error(f"Error aggregating processing stats for doc {doc_id}: {e}", exc_info=True)
        return None, str(e)

def get_overall_processing_stats() -> Tuple[Optional[dict], Optional[str]]:
    """
    Calculates aggregate statistics across all processed chunks in the system.
    Returns (stats_dict, error_msg).
    """
    try:
        # Define statuses
        reprocess_statuses = ['pending_reprocess', 'processing_content', 'processing_reprocess']
        active_statuses = ['completed', 'pending_embedding', 'processing_embedding'] + reprocess_statuses
        
        # 1. Reprocess Queue Count
        # Efficient count query for queue
        queue_query = chunks_ref.where('status', 'in', reprocess_statuses)
        queue_res = queue_query.count().get()
        reprocessing_queue = queue_res[0][0].value

        # 2. Total Active Chunks (All processed chunks)
        total_query = chunks_ref.where('status', 'in', active_statuses)
        total_res = total_query.count().get()
        total = total_res[0][0].value
        
        # 3. LLM Chunks
        # Use simple count query for chunks explicitly marked as 'llm'
        llm_query = chunks_ref.where('extraction_source', '==', 'llm')
        llm_res = llm_query.count().get()
        llm_count = llm_res[0][0].value
        
        # 4. Legacy Chunks
        # Use simple count query for chunks explicitly marked as 'legacy'
        legacy_query = chunks_ref.where('extraction_source', '==', 'legacy')
        legacy_res = legacy_query.count().get()
        legacy_count = legacy_res[0][0].value
                    
        return {
            "total_chunks": int(total),
            "gemini_chunks": int(llm_count),
            "docai_chunks": int(legacy_count),
            "reprocessing_queue": int(reprocessing_queue)
        }, None
    except Exception as e:
        logger.error(f"Error fetching global stats: {e}", exc_info=True)
        return None, str(e)



def get_documents_with_legacy_chunks_batch(
    limit: int = 1000, 
    last_doc = None,
    min_created_at: Optional[datetime.datetime] = None
) -> Tuple[Set[str], Any, Optional[str]]:
    """
    Finds unique document IDs with legacy chunks (specifically failed extractions) in batches.
    
    Args:
        limit: Maximum number of chunks to fetch per batch.
        last_doc: Firestore snapshot to start pagination after.
        min_created_at: Optional datetime to filter chunks - only returns chunks 
                        created ON OR AFTER this timestamp. Used by Cloud Scheduler
                        to skip aged chunks (e.g., chunks older than 7 days).
    
    Returns (doc_ids_set, last_evaluated_doc_snapshot, error_msg).
    """
    try:
        # Strategy: Query for chunks explicitly marked as having "legacy" source.
        # This relies on the migration script having been run to tag existing failed chunks.
        # This is a robust, clean query that avoids depending on nested "failed" flags.
        query = chunks_ref.where(filter=FieldFilter('extraction_source', '==', 'legacy'))
        
        # Apply date filter if provided (for Cloud Scheduler optimization)
        if min_created_at:
            query = query.where(filter=FieldFilter('created_at', '>=', min_created_at))
            logger.info(f"Filtering legacy chunks with created_at >= {min_created_at.isoformat()}")
        
        if last_doc:
            query = query.start_after(last_doc)
            
        query = query.limit(limit)
        
        # Select minimum fields needed to identify the parent
        chunks_stream = query.select(['parent_doc_id', 'original_doc_firestore_id']).stream()
        
        doc_ids = set()
        last_snapshot = None
        
        for doc in chunks_stream:
            last_snapshot = doc
            data = doc.to_dict()
            doc_id = data.get('parent_doc_id') or data.get('original_doc_firestore_id')
            if doc_id:
                doc_ids.add(doc_id)
                    
        return doc_ids, last_snapshot, None
        
    except Exception as e:
        logger.error(f"Error fetching legacy chunks batch: {e}", exc_info=True)
        return set(), None, str(e)

