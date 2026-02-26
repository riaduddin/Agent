# backend/app/services/doc_processing_helpers/bulk_processing_utils.py
import os
import uuid
import datetime
import logging
from typing import Optional

from google.cloud import firestore, storage
from app import config, db
from app.utils.redis_client import get_redis_client, QUEUE_NAME
from app.services import vertex_ai_service # Import vertex_ai_service
from app.utils.debug_logger import debug_log

logger = logging.getLogger(__name__)
storage_client = storage.Client(project=config.PROJECT_ID) # Initialize GCS client for this module
redis_client = get_redis_client() # Initialize Redis client for this module

# --- Helper Functions ---
def is_superadmin(user_email: str) -> bool:
    """
    Checks if the user has superadmin privileges.
    Uses the SUPERADMIN_EMAILS environment variable if available.
    """
    import os
    superadmin_emails = os.getenv("SUPERADMIN_EMAILS", "").split(",")
    superadmin_emails = [e.strip() for e in superadmin_emails if e.strip()]
    
    if superadmin_emails:
        return user_email in superadmin_emails
    
    # Fallback to True if no emails configured (dev mode safe-ish)
    # but log a warning.
    logger.warning(f"No SUPERADMIN_EMAILS configured in .env. Defaulting to True for {user_email}.")
    return True

# --- Task Queue Interaction ---
def enqueue_process_document_task(doc_firestore_id: str, gcs_uri: str):
    """Enqueues a single task to process an entire document (split, OCR, vectorize)."""
    # This function was originally in document_processing_service.py
    # It's moved here as it's primarily used by bulk operations.
    # The task name "process_document::" might need to be updated to "split_and_process_pdf::"
    # if this is the entry point for the new worker flow.
    # For now, keeping original logic. If worker expects "split_and_process_pdf::", this needs change.
    # Based on current worker.py, it expects "split_and_process_pdf::" from the API,
    # but bulk processing might still enqueue an older "process_document::" task if not updated.
    # Let's assume for now this is for the OLD bulk flow, or it needs an update.
    # For the purpose of refactoring, I'm moving the function as is.
    # If this function is intended to trigger the NEW 17-step flow, the task_data string
    # should be "split_and_process_pdf::{doc_firestore_id}::{gcs_uri}"
    try:
        task_data = f"split_and_process_pdf::{doc_firestore_id}::{gcs_uri}"
        redis_client.rpush(QUEUE_NAME, task_data)
        logger.info(f"Enqueued task for document {doc_firestore_id}: {task_data}")
    except Exception as e:
        logger.error(f"Failed to enqueue task for {doc_firestore_id}: {e}", exc_info=True)

def add_task_to_queue(task_name: str, payload: dict):
    """
    Generic helper to add tasks to the Redis queue.
    Formats the task string based on the task name to match worker expectations.
    """
    try:
        task_data = ""
        if task_name == "reprocess_legacy_doc_chunks":
            # Worker expects: reprocess_legacy_doc_chunks::doc_id
            doc_id = payload.get("doc_id")
            if not doc_id:
                raise ValueError("doc_id is required for reprocess_legacy_doc_chunks")
            task_data = f"{task_name}::{doc_id}"
            
        elif task_name == "initiate_bulk_reprocess_legacy":
             # Worker expects: initiate_bulk_reprocess_legacy::{user_email}::{trigger_source}
             user_email = payload.get("user_email")
             trigger_source = payload.get("trigger_source", "MANUAL")
             task_data = f"{task_name}::{user_email}::{trigger_source}"
             
        # Add other definitions here as needed
        else:
             # Default fallback or error
             raise ValueError(f"Unknown task type for auto-formatting: {task_name}")

        redis_client.rpush(QUEUE_NAME, task_data)
        logger.info(f"Enqueued task: {task_data}")
        
    except Exception as e:
        logger.error(f"Failed to enqueue task {task_name}: {e}", exc_info=True)
        raise e

# --- Main Orchestration (Triggered by API) ---
def initiate_bulk_processing(gcs_prefix_override: Optional[str] = None):
    """
    Scans GCS for PDFs within a specified prefix, tags them in Firestore,
    and enqueues initial splitting tasks. Uses prefix from config if not overridden.
    """
    prefix_to_use = gcs_prefix_override if gcs_prefix_override is not None else config.GCS_BULK_PROCESSING_PREFIX
    prefix_to_use = prefix_to_use.lstrip('/')
    if prefix_to_use and not prefix_to_use.endswith('/'):
        prefix_to_use += '/'

    logger.info(f"Initiating bulk processing scan for gs://{config.BUCKET_NAME}/{prefix_to_use}")
    run_id = str(uuid.uuid4())
    start_time = datetime.datetime.now(datetime.timezone.utc)
    run_doc_ref = db.collection("bulk_process_runs").document(run_id)

    initial_stats = {
        "run_id": run_id, "start_time": start_time, "end_time": None,"completed": 0,
        "status": "scanning_gcs", "gcs_prefix_used": prefix_to_use,
        "files_found": 0, "files_tagged_for_split": 0, "errors": []
    }
    try:
        run_doc_ref.set(initial_stats)
        logger.info(f"Created initial stats record for bulk run {run_id}")

        blobs = storage_client.list_blobs(config.BUCKET_NAME, prefix=prefix_to_use)
        files_found_count = 0
        files_enqueued_count = 0
        blob_list = list(blobs)
        files_found_count = sum(1 for blob in blob_list if blob.name.lower().endswith(".pdf") and blob.name != prefix_to_use)
        logger.info(f"Found {files_found_count} PDF files in GCS scan.")
        run_doc_ref.update({"files_found": files_found_count, "status": "tagging_and_enqueueing"})

        for blob in blob_list:
            if blob.name == prefix_to_use or not blob.name.lower().endswith(".pdf"):
                continue

            doc_meta_ref = db.collection("document_metadata").document()
            doc_firestore_id = doc_meta_ref.id
            raw_blob_size = blob.size
            file_size_to_store = raw_blob_size if isinstance(raw_blob_size, int) and raw_blob_size >= 0 else None
            if file_size_to_store is None:
                 logger.warning(f"Invalid size detected for blob {blob.name}: {raw_blob_size}. Storing null.")
            
            metadata = {
                "original_filename": os.path.basename(blob.name),
                "gcs_uri": f"gs://{config.BUCKET_NAME}/{blob.name}",
                "gcs_blob_name": blob.name,
                "file_size_bytes": file_size_to_store,
                "content_type": blob.content_type,
                "status": "pending", # This status will be picked up by enqueue_pending_documents or the new flow
                "upload_timestamp": datetime.datetime.now(datetime.timezone.utc),
                "last_status_update": datetime.datetime.now(datetime.timezone.utc),
                "source": "gcs_bulk", "bulk_run_id": run_id,
                # For new flow, initialize these:
                "total_chunks": 0,
                "completed_chunks": 0,
                "processing_events": []
            }
            doc_meta_ref.set(metadata)
            logger.info(f"Tagged document {doc_firestore_id} for GCS file {metadata['gcs_uri']}")
            
            # Enqueue for the new 17-step flow
            task_data_new_flow = f"split_and_process_pdf::{doc_firestore_id}::{metadata['gcs_uri']}"
            redis_client.rpush(QUEUE_NAME, task_data_new_flow)
            logger.info(f"Enqueued NEW FLOW task for document {doc_firestore_id}: {task_data_new_flow}")
            files_enqueued_count += 1

        logger.info(f"Finished GCS scan. Enqueued tasks for {files_enqueued_count} PDF documents.")
        run_doc_ref.update({"files_tagged_for_processing": files_enqueued_count, "status": "enqueued"})
       
    except Exception as e:
        error_message = f"Error during bulk processing initiation: {e}"
        logger.error(error_message, exc_info=True)
        run_doc_ref.update({
            "status": "error", "errors": firestore.ArrayUnion([error_message]),
            "end_time": datetime.datetime.now(datetime.timezone.utc)
         })

def enqueue_pending_documents():
    """
    Queries Firestore for documents with status 'pending' (typically from gcs_bulk source)
    and enqueues the 'split_and_process_pdf::' task for them if not already processed by initiate_bulk_processing.
    """
    logger.info("Starting enqueue process for 'pending' documents...")
    enqueued_count = 0
    error_count = 0
    # Query for documents that are from 'gcs_bulk' and still 'pending'
    # This assumes initiate_bulk_processing sets them to 'pending' and then enqueues.
    # This function could serve as a catch-all or for re-enqueueing if initial enqueue failed.
    # MODIFIED: Query for any document with status "pending", regardless of source.
    # Use 'in' to catch both 'pending' and 'Pending' statuses.
    pending_docs_query = db.collection("document_metadata").where("status", "in", ["pending", "Pending", "queued_for_splitting"]).stream()

    # Convert stream to list to get length and allow iteration.
    # The original code had a bug where list() would exhaust the stream before the loop.
    pending_docs = list(pending_docs_query)

    # Debug message if no pending documents are found
    debug_log(f"Found {len(pending_docs)} pending documents to process.")

    for doc in pending_docs:
        doc_id = doc.id
        doc_data = doc.to_dict()
        gcs_uri = doc_data.get("gcs_uri")

        # Remove its Chunks
        delete_all_chunks_for_parent(doc_id)

        if gcs_uri:
            try:
                # Enqueue for the new 17-step flow
                task_data_new_flow = f"split_and_process_pdf::{doc_id}::{gcs_uri}"
                redis_client.rpush(QUEUE_NAME, task_data_new_flow)
                logger.info(f"Enqueued NEW FLOW task for pending document {doc_id}: {task_data_new_flow}")
                # Update status to 'queued_for_splitting' to prevent re-enqueueing before processing
                db.collection("document_metadata").document(doc_id).update({
                    "status": "queued_for_splitting",
                    "last_status_update": firestore.SERVER_TIMESTAMP,
                    "status_message": "Re-enqueued for processing by system."
                })
                logger.info(f"Updated status to 'queued_for_splitting' for document {doc_id}")
                enqueued_count += 1
            except Exception as e:
                logger.error(f"Failed to enqueue task or update status for pending doc {doc_id}: {e}", exc_info=True)
                error_count += 1
        else:
            logger.warning(f"Pending document {doc_id} is missing gcs_uri. Cannot enqueue.")
            error_count += 1

    logger.info(f"Finished enqueueing pending documents. Enqueued tasks: {enqueued_count}, Errors/Skipped: {error_count}")
    return enqueued_count, error_count

def delete_all_chunks_for_parent(parent_doc_id: str) -> dict:
    """
    Deletes all chunk documents from Firestore and their corresponding blobs from GCS
    for a given parent document ID.
    """
    debug_log(f"delete_all_chunks_for_parent: Starting for parent_doc_id: {parent_doc_id}")
    logger.info(f"Starting deletion of all chunks for parent document: {parent_doc_id}")
    
    # 1. Get all chunk IDs (which are also vector datapoint IDs)
    debug_log(f"delete_all_chunks_for_parent: Querying Firestore for chunks of parent {parent_doc_id}")
    chunks_ref = db.collection("document_chunks").where("parent_doc_id", "==", parent_doc_id)
    
    query1 = db.collection("document_chunks") \
    .where("parent_doc_id", "==", parent_doc_id) \
    .stream()

    query2 = db.collection("document_chunks") \
    .where("original_doc_firestore_id", "==", parent_doc_id) \
    .stream()
    
    all_docs = {}
    for doc in list(query1) + list(query2):
        all_docs[doc.id] = doc

    chunk_docs = list(all_docs.values())

    debug_log(f"delete_all_chunks_for_parent: Found {len(chunk_docs)} chunks in Firestore.")
    
    if not chunk_docs:
        logger.info(f"No existing chunks found in Firestore for parent {parent_doc_id}. Nothing to delete.")
        debug_log(f"delete_all_chunks_for_parent: No chunks found, returning.")
        return {"status": "success", "message": "No chunks found to delete."}

    chunk_ids_to_delete = [doc.id for doc in chunk_docs]
    gcs_paths_to_to_delete = [doc.to_dict().get("gcs_path_chunk") for doc in chunk_docs if doc.to_dict().get("gcs_path_chunk")]
    debug_log(f"delete_all_chunks_for_parent: Chunk IDs to delete from Vector DB: {len(chunk_ids_to_delete)}")
    debug_log(f"delete_all_chunks_for_parent: GCS paths to delete: {len(gcs_paths_to_to_delete)}")

    # 2. Delete embeddings from Vector Database
    vector_removal_success = False
    vector_removal_error = None
    if chunk_ids_to_delete:
        debug_log(f"delete_all_chunks_for_parent: Calling vertex_ai_service.remove_vector_datapoints for {len(chunk_ids_to_delete)} IDs.")
        try:
            logger.info(f"Attempting to remove {len(chunk_ids_to_delete)} datapoints from Vector Search for parent {parent_doc_id}.")
            vector_removal_success, vector_removal_error = vertex_ai_service.remove_vector_datapoints(chunk_ids_to_delete)
            if vector_removal_success:
                logger.info(f"Successfully initiated removal of {len(chunk_ids_to_delete)} datapoints from Vector Search for parent {parent_doc_id}.")
                debug_log(f"delete_all_chunks_for_parent: Vector DB removal initiated successfully.")
            else:
                logger.error(f"Failed to initiate removal of datapoints from Vector Search for parent {parent_doc_id}: {vector_removal_error}")
                debug_log(f"delete_all_chunks_for_parent: Vector DB removal failed: {vector_removal_error}")
        except Exception as e:
            logger.error(f"Unexpected error during Vector Search datapoint removal for parent {parent_doc_id}: {e}", exc_info=True)
            debug_log(f"delete_all_chunks_for_parent: Exception during Vector DB removal: {e}")
            vector_removal_error = str(e)
    else:
        vector_removal_success = True # No chunks, so no vectors to remove
        logger.info(f"No chunk IDs to delete from Vector Search for parent {parent_doc_id}.")
        debug_log(f"delete_all_chunks_for_parent: No chunk IDs to delete from Vector DB.")

    # 3. Delete GCS Blobs
    debug_log(f"delete_all_chunks_for_parent: Starting GCS blob deletion for {len(gcs_paths_to_to_delete)} blobs.")
    if gcs_paths_to_to_delete:
        bucket_name = config.BUCKET_NAME
        bucket = storage_client.bucket(bucket_name)
        
        for gcs_path in gcs_paths_to_to_delete:
            try:
                if gcs_path.startswith(f"gs://{bucket_name}/"):
                    blob_name = gcs_path[len(f"gs://{bucket_name}/"):]
                    blob = bucket.blob(blob_name)
                    blob.delete()
                    logger.info(f"Deleted GCS blob: {blob_name}")
                    debug_log(f"delete_all_chunks_for_parent: Deleted GCS blob: {blob_name}")
            except Exception as e:
                logger.error(f"Failed to delete GCS blob {gcs_path}: {e}")
                debug_log(f"delete_all_chunks_for_parent: Failed to delete GCS blob {gcs_path}: {e}")
                # Continue deletion process even if one blob fails
    debug_log(f"delete_all_chunks_for_parent: Finished GCS blob deletion.")

    # 4. Delete Firestore Chunk Documents (batched)
    debug_log(f"delete_all_chunks_for_parent: Starting Firestore chunk document deletion for {len(chunk_docs)} documents.")
    batch = db.batch()
    for doc in chunk_docs:
        batch.delete(doc.reference)
    batch.commit()
    debug_log(f"delete_all_chunks_for_parent: Committed Firestore batch deletion.")
    
    logger.info(f"Deleted {len(chunk_docs)} chunk documents from Firestore for parent {parent_doc_id}.")
    debug_log(f"delete_all_chunks_for_parent: Finished Firestore chunk document deletion.")
    
    return {
        "status": "success", 
        "deleted_count": len(chunk_docs),
        "vector_db_removal_initiated": vector_removal_success,
        "vector_db_removal_error": vector_removal_error
    }


def reprocess_legacy_chunks_for_doc(
    parent_doc_id: str,
    skip_aged_chunks: bool = False,
    max_age_days: Optional[int] = None
) -> dict:
    """
    Identifies 'legacy' chunks (not multimodal) for a given document,
    removes their vectors, resets their status to 'pending_reprocess',
    and enqueues a SINGLE task for the worker to process them in parallel.
    
    This preserves GCS blobs and avoids re-splitting the PDF.
    """
    debug_log(f"reprocess_legacy_chunks: Starting for parent_doc_id: {parent_doc_id}")
    logger.info(f"Starting selective reprocessing setup for document: {parent_doc_id}")
    
    # 1. Identify Legacy Chunks
    # We look for chunks where extracted_entities._extraction_method != 'multimodal'
    # Since we need to support legacy chunks that might only have original_doc_firestore_id,
    # we queried both fields by combining results.
    
    # Query 1: standard parent_doc_id
    q1 = db.collection("document_chunks").where("parent_doc_id", "==", parent_doc_id).stream()
    # Query 2: legacy original_doc_firestore_id
    q2 = db.collection("document_chunks").where("original_doc_firestore_id", "==", parent_doc_id).stream()
    
    all_chunks_map = {}
    for doc in list(q1) + list(q2):
        all_chunks_map[doc.id] = doc
        
    all_chunks = list(all_chunks_map.values())
    legacy_chunks = []
    aged_out_chunks = 0

    def _normalize_timestamp(value: Optional[object]) -> Optional[datetime.datetime]:
        if value is None:
            return None
        if hasattr(value, "to_datetime"):
            value = value.to_datetime()
        if isinstance(value, datetime.datetime):
            if value.tzinfo is None:
                value = value.replace(tzinfo=datetime.timezone.utc)
            return value
        return None

    def _is_chunk_aged_out(chunk_data: dict) -> bool:
        if not skip_aged_chunks or not max_age_days:
            return False
        ts = _normalize_timestamp(chunk_data.get("created_at") or chunk_data.get("last_updated"))
        if not ts:
            return False
        cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=max_age_days)
        return ts <= cutoff
    for doc in all_chunks:
        data = doc.to_dict()
        extracted = data.get("extracted_entities", {})
        
        # NEW: Check the top-level field first, fallback to extraction method
        source = data.get("extraction_source")
        if source:
            is_multimodal = (source == "llm")
        else:
            method = extracted.get("_extraction_method") or extracted.get("extraction_method")
            is_multimodal = (method == "multimodal")
        
        # Also check if it's already pending reprocess (idempotency)
        if not is_multimodal:
            if _is_chunk_aged_out(data):
                aged_out_chunks += 1
                continue
            legacy_chunks.append(doc)

    if not legacy_chunks:
        if aged_out_chunks > 0:
            logger.info(
                f"All legacy chunks for doc {parent_doc_id} are aged >= {max_age_days} days. Skipping."
            )
            return {
                "status": "skipped",
                "message": f"All legacy chunks are aged >= {max_age_days} days.",
                "skipped_count": aged_out_chunks,
                "reason": "aged_out"
            }
        logger.info(f"No legacy chunks found for doc {parent_doc_id}. Nothing to reprocess.")
        return {"status": "skipped", "message": "No legacy chunks found.", "skipped_count": 0}
    
    chunk_ids_to_reprocess = [doc.id for doc in legacy_chunks]
    logger.info(f"Found {len(chunk_ids_to_reprocess)} legacy chunks to reprocess for doc {parent_doc_id}.")

    # 2. Delete Embeddings for these chunks ONLY
    # We must remove old vectors because the new text/entities will generate new vectors.
    vector_removal_success = False
    try:
        validation, errors = vertex_ai_service.remove_vector_datapoints(chunk_ids_to_reprocess)
        vector_removal_success = validation # remove_vector_datapoints returns (bool, error_str)
        if vector_removal_success:
             logger.info(f"Successfully initiated removal of {len(chunk_ids_to_reprocess)} legacy vectors.")
        else:
             logger.warning(f"Vector removal returned failure: {errors}")
    except Exception as e:
        logger.error(f"Failed to remove vectors for legacy chunks: {e}")
        # Proceed anyway? If vectors stay, we might have duplicates or stale data. 
        # But we overwrite the chunk metadata. Vector search might return old chunk ID, 
        # but if we re-embed with same ID, it *should* overwrite in Vector Search (upsert).
        # Vertex AI Search Index supports upsert if ID matches. 
        # However, it's cleaner to remove. We'll proceed.

    # 3. Batch Update Firestore: Reset status to 'pending_reprocess'
    batch = db.batch()
    for doc in legacy_chunks:
        # Reset fields to clean state for new extraction
        update_data = {
            "status": "pending_reprocess",
            "parent_doc_id": parent_doc_id, # Backfill/Ensure this is set for worker to find it
            # Clear old extraction data
            "extracted_entities": {}, 
            "ocr_text": firestore.DELETE_FIELD, # Will be re-populated
            "ocr_text_preview": firestore.DELETE_FIELD,
            "entities": [], # Clear searchable entities
            "docai_extracted_entities": firestore.DELETE_FIELD,
            "error_message": firestore.DELETE_FIELD,
            "last_updated": datetime.datetime.now(datetime.timezone.utc)
        }
        batch.update(doc.reference, update_data)
    
    batch.commit()
    logger.info(f"Reset {len(legacy_chunks)} chunks to 'pending_reprocess' status.")

    # 4. Enqueue SINGLE task for the worker
    try:
        # Format: reprocess_legacy_doc_chunks::{doc_id}
        task_str = f"reprocess_legacy_doc_chunks::{parent_doc_id}"
        redis_client.rpush(QUEUE_NAME, task_str)
        logger.info(f"Enqueued reprocessing task: {task_str}")
        
        # Update Parent Status
        # We NO LONGER update the status to processing_reprocess to avoid clustering the UI file list.
        # The user requested to see stats only in the dashboard card.
        # db.collection("document_metadata").document(parent_doc_id).update({
        #     "status": "processing_reprocess",
        #     "last_status_update": datetime.datetime.now(datetime.timezone.utc),
        #     "status_message": f"Reprocessing {len(legacy_chunks)} legacy chunks."
        # })
        
    except Exception as e:
        logger.error(f"Failed to enqueue reprocess task for {parent_doc_id}: {e}")
        return {"status": "error", "message": str(e)}

    return {
        "status": "success",
        "reprocessed_count": len(legacy_chunks),
        "skipped_count": aged_out_chunks,
        "queued_task": task_str
    }

def start_legacy_reprocess_job(
    user_email: str,
    skip_aged_chunks: bool = False,
    max_age_days: Optional[int] = None,
    trigger_source: str = "MANUAL"
):
    """
    Orchestrator function running in the worker.
    Finds all documents with legacy chunks and queues a reprocess task for each.
    """
    from app.models import metadata_model
    
    run_id = str(uuid.uuid4())
    start_time = datetime.datetime.now(datetime.timezone.utc)
    run_doc_ref = db.collection("bulk_process_runs").document(run_id)
    
    logger.info(f"Starting bulk legacy reprocess job {run_id} for user {user_email}")
    
    initial_stats = {
        "run_id": run_id, 
        "start_time": start_time,
        "end_time": None,
        "type": "legacy_reprocess_selective",
        "initiated_by": user_email,
        "trigger_source": trigger_source,
        "trigger_details": {
            "user_email": user_email if trigger_source == "MANUAL" else None,
            "scheduler_job": "legacy-reprocess-job" if trigger_source == "CLOUD_SCHEDULER" else None
        },
        "skip_aged_chunks": skip_aged_chunks,
        "max_age_days": max_age_days,
        "status": "scanning",
        "docs_found": 0,
        "docs_queued": 0,
        "docs_skipped": 0,
        "errors": []
    }
    run_doc_ref.set(initial_stats)
    
    try:
        # Loop through batches
        last_cursor = None
        total_docs_processed = 0
        batch_size = 1000 
        
        # Initialize counters
        processed_count = 0
        error_count = 0
        skipped_count = 0 
        
        # Calculate cutoff date for Firestore query-level filtering
        # This is a major optimization: instead of fetching ALL legacy chunks
        # and filtering in Python, we push the date filter to Firestore.
        min_created_at = None
        if skip_aged_chunks and max_age_days:
            min_created_at = datetime.datetime.now(datetime.timezone.utc) - \
                             datetime.timedelta(days=max_age_days)
            logger.info(f"Query-level date filter: only chunks with created_at >= {min_created_at.isoformat()}")
        
        while True:
            # Fetch Batch (with optional date filter pushed to Firestore)
            batch_doc_ids, new_cursor, error = metadata_model.get_documents_with_legacy_chunks_batch(
                limit=batch_size, 
                last_doc=last_cursor,
                min_created_at=min_created_at  # Pass cutoff to Firestore query
            )
            
            if error:
                raise Exception(f"Batch query failed: {error}")
            
            # Queue valid docs from this batch
            if batch_doc_ids:
                doc_count = len(batch_doc_ids)
                total_docs_processed += doc_count
                logger.info(f"Batch found {doc_count} unique legacy docs. Queueing...")
                
                # Update "Found" count incrementally
                run_doc_ref.update({
                    "status": "queuing",
                    "docs_found": total_docs_processed
                })

                for doc_id in batch_doc_ids:
                    try:
                        # CRITICAL: We must call the setup function for each doc
                        # which deletes embeddings, resets status to pending_reprocess,
                        # AND enqueues the individual reprocess task.
                        result = reprocess_legacy_chunks_for_doc(
                            doc_id,
                            skip_aged_chunks=skip_aged_chunks,
                            max_age_days=max_age_days
                        )

                        if result.get("status") == "skipped":
                            skipped_count += 1
                        else:
                            processed_count += 1
                        
                        # Update progress every 100 docs
                        if processed_count % 100 == 0:
                            run_doc_ref.update({
                                "docs_queued": processed_count
                            })
                        if skipped_count % 100 == 0 and skipped_count > 0:
                            run_doc_ref.update({
                                "docs_skipped": skipped_count
                            })
                            
                    except Exception as e:
                        logger.error(f"Failed to queue doc {doc_id}: {e}")
                        error_count += 1
                        run_doc_ref.update({
                            "errors": firestore.ArrayUnion([f"Failed to queue {doc_id}: {str(e)}"])
                        })
            elif not new_cursor:
                 # If no docs found and no cursor returned, we are truly done
                 logger.info("No more documents found in batch.")

            # Move cursor forward
            if not new_cursor:
                break
                
            last_cursor = new_cursor
            
        # Final update
        run_doc_ref.update({
            "status": "completed",
            "docs_queued": processed_count,
            "docs_skipped": skipped_count, # skipped_count is effectively 0 in this new flow as we filtered in query
            "end_time": datetime.datetime.now(datetime.timezone.utc)
        })
        
        logger.info(f"Bulk legacy reprocess job finished. Queued {processed_count} docs.")
        
    except Exception as e:
        logger.error(f"Job {run_id} failed: {e}", exc_info=True)
        run_doc_ref.update({
            "status": "error",
            "error_message": str(e),
            "end_time": datetime.datetime.now(datetime.timezone.utc)
        })
