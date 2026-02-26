# backend/worker.py
import logging
import time
import traceback
import uuid # Import uuid for worker ID
from redis import RedisError
import os
from dotenv import load_dotenv
from google.api_core import exceptions as api_core_exceptions # Import Google API exceptions
import concurrent.futures # For parallel chunk processing
import datetime # For logging timestamps
from typing import Optional # Import Optional for type hinting

# --- Configure logging FIRST (before any app imports) ---
import sys
from app.utils.logging_utils import setup_cloud_logging

# Configure root logger with structured JSON on stdout
logger = setup_cloud_logging()

# Suppress noisy internal loggers BEFORE imports - only show WARNINGS and above
# This keeps terminal clean, showing only our [DEBUG] pattern logs and critical errors
_loggers_to_suppress = [
    'app.models.log_model',
    'app.services.doc_processing_helpers.firestore_ops',
    'app.services.doc_processing_helpers.ocr_utils',
    'app.services.doc_processing_helpers.classification_utils',
    'app.services.document_processing_service',
    'app.services.gcs_service',
    'app.services.vertex_ai_service',
    'app.services.metadata_extraction_service',
    'app.services.type_specific_extraction_service',
    'app.services.category_api_service',
    'app.services.categorization_service',
    'app.utils.redis_client',
    'google',
    'urllib3',
    'google.cloud',
    'google.auth',
    'grpc',
]
for logger_name in _loggers_to_suppress:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

# We can get a specific logger for this module if we want, 
# although root logger config covers it.
logger = logging.getLogger(__name__)

# --- Load .env file at the very start ---
# Determine the path to the .env file relative to this script's location
backend_dir = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(backend_dir, '.env')
print(f"WORKER: Attempting to load .env file from: {dotenv_path}")
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)
    print("WORKER: .env file loaded.")
else:
    print(f"WORKER: .env file not found at {dotenv_path}. Relying on system environment variables.")

# --- DEBUG: Check env var immediately after load_dotenv ---
print(f"WORKER DEBUG: os.getenv('VECTOR_INDEX_NAME') after load_dotenv: {os.getenv('VECTOR_INDEX_NAME')}")
# --- END DEBUG ---

# Import config and services AFTER loading .env (logging already configured)
from app import config, db
from app.services import document_processing_service as dps
from app.services import vertex_ai_service as vais
from app.services import gcs_service # For download_blob_to_bytes
from app.services import categorization_service # Import the new service
from app.services.metadata_extraction_service import MetadataExtractionService # Import metadata service (legacy, still used for parent-level extraction)
from app.services.type_specific_extraction_service import TypeSpecificExtractionService # NEW: Per-chunk entity extraction
from app.services.doc_processing_helpers import pdf_utils # For split_pdf
from app.services.doc_processing_helpers import firestore_ops # For status updates
from app.services.doc_processing_helpers import ocr_utils # For perform_ocr
from app.services.doc_processing_helpers import classification_utils # For get_parser_for_chunk_via_gemini
from app.services.doc_processing_helpers import bulk_processing_utils # For legacy reprocess orchestration
from app.utils.redis_client import get_redis_client, QUEUE_NAME
from app.models.log_model import add_log_entry
from google.cloud import firestore # For FieldValue
from app.utils.debug_logger import debug_log, set_debug_patterns  # DEBUG: For tracing extraction

# --- Redis Lock Configuration ---
LOCK_KEY_PREFIX = "doc_process_lock:"
LOCK_TIMEOUT_SECONDS = 900 # 15 minutes

# --- Main Worker Loop ---
def main():
    worker_instance_id = str(uuid.uuid4())
    logger.info(f"Starting worker process (ID: {worker_instance_id})...")
    redis_client = get_redis_client()
    if not redis_client:
        logger.error(f"Worker {worker_instance_id}: Failed to connect to Redis. Worker cannot start.")
        return

    logger.info(f"Worker {worker_instance_id} connected to Redis, listening on queue: {QUEUE_NAME}")
    add_log_entry("INFO", "Worker started and connected to Redis.", step="worker_startup", worker_id=worker_instance_id)

    while True:
        task_str = None
        log_details = {"worker_id": worker_instance_id} # Initialize local variable to avoid UnboundLocalError
        try:
            task_data_bytes = redis_client.blpop(QUEUE_NAME, timeout=0)
            if not task_data_bytes:
                continue
            task_str = task_data_bytes[1]
            print(f"\n--- WORKER: Received Task ---\n{task_str}\n--------------------------")

            log_details = {"raw_task": task_str, "worker_id": worker_instance_id}

            if task_str.startswith("split_and_process_pdf::"):
                handle_split_and_process_pdf_task(task_str, log_details, redis_client, worker_instance_id)
            elif task_str.startswith("backfill_category::") or task_str.startswith("categorize_new_document::"):
                handle_categorization_task(task_str, log_details, worker_instance_id)
            elif task_str.startswith("reprocess_legacy_doc_chunks::"):
                handle_reprocess_legacy_doc_chunks(task_str, log_details, redis_client, worker_instance_id)
            elif task_str.startswith("initiate_bulk_reprocess_legacy::"):
                handle_initiate_bulk_reprocess_legacy(task_str, log_details, worker_instance_id)
            elif task_str.startswith("process_document::"):
                logger.warning(f"Worker {worker_instance_id}: Received deprecated 'process_document::' task: {task_str}. New flow uses 'split_and_process_pdf::'. Skipping.")
                add_log_entry("WARNING", "Received deprecated 'process_document::' task", step="deprecated_task", details=log_details)
            else:
                logger.warning(f"Worker {worker_instance_id}: Unknown task format received: {task_str}")
                add_log_entry("WARNING", f"Unknown/deprecated task format", step="unknown_task", details=log_details)

        except RedisError as e:
            logger.error(f"Worker {worker_instance_id}: Redis error: {e}. Attempting to reconnect...", exc_info=True)
            add_log_entry("ERROR", "Redis connection error", step="redis_connect", details={"error": str(e), **log_details})
            time.sleep(5)
            redis_client = get_redis_client()
            if not redis_client:
                logger.error(f"Worker {worker_instance_id}: Failed to reconnect to Redis. Worker stopping.")
                add_log_entry("CRITICAL", "Failed to reconnect to Redis. Worker stopping.", step="redis_connect_failed", details=log_details)
                break
        except Exception as e:
            logger.error(f"Worker {worker_instance_id}: Unhandled error processing task: {task_str if task_str else 'N/A'}", exc_info=True)
            add_log_entry("ERROR", f"Unhandled error processing task", step="unhandled_worker_error", details={"task": task_str if task_str else 'N/A', "error": str(e), "traceback": traceback.format_exc(), **log_details})
            time.sleep(1)


def update_parent_doc_event(parent_doc_id: str, event_name: str, status: Optional[str] = None, details: Optional[dict] = None, original_filename: Optional[str] = None):
    """Helper to add an event to the parent document's processing_events array and main logs."""
    try:
        parent_ref = db.collection("document_metadata").document(parent_doc_id)
        now_iso = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()

        event_log_for_array = {
            "timestamp": now_iso,
            "event": event_name,
        }
        if status:
            event_log_for_array["status"] = status
        if details:
            event_log_for_array["details"] = details

        parent_ref.update({"processing_events": firestore.ArrayUnion([event_log_for_array])})
        logger.info(f"Logged event '{event_name}' to parent doc {parent_doc_id}'s array.")

        log_level = "INFO"
        if "fail" in event_name.lower() or "error" in event_name.lower() or (status and "error" in status.lower()):
            log_level = "ERROR"
        elif "warn" in event_name.lower():
            log_level = "WARNING"

        log_message = f"Parent Doc Event: {event_name}"
        if status:
            log_message += f" (Status: {status})"

        add_log_entry(
            level=log_level,
            message=log_message,
            document_id=parent_doc_id,
            step=event_name,
            original_filename=original_filename,
            details=details
        )
        logger.info(f"Also logged event '{event_name}' for parent doc {parent_doc_id} to main processing_logs.")

    except Exception as e:
        logger.error(f"Failed to log event '{event_name}' for parent doc {parent_doc_id}: {e}", exc_info=True)

def handle_split_and_process_pdf_task(task_str, log_details_base, redis_client, worker_id):
    parent_doc_id = None
    gcs_path_original_parent = None
    original_parent_filename = "unknown_parent_filename"
    lock_key = None
    lock_acquired = False
    log_details = {**log_details_base}

    try:
        parts = task_str.split('::', 2)
        if len(parts) != 3 or parts[0] != 'split_and_process_pdf':
            err_msg = f"Invalid split_and_process_pdf task format: {task_str}"
            logger.error(err_msg)
            add_log_entry("ERROR", err_msg, step="split_task_parse", details=log_details, worker_id=worker_id, original_filename=original_parent_filename)
            return
        _, parent_doc_id, gcs_path_original_parent = parts
        log_details.update({"parent_doc_id": parent_doc_id, "gcs_path_original_parent": gcs_path_original_parent})
        logger.info(f"Worker {worker_id}: Starting task 'split_and_process_pdf' for parent_doc_id: {parent_doc_id}")
        update_parent_doc_event(parent_doc_id, "worker_task_received", status="processing_split", details={"gcs_path": gcs_path_original_parent, "worker_id": worker_id}, original_filename=original_parent_filename)

        lock_key = f"{LOCK_KEY_PREFIX}{parent_doc_id}"
        lock_acquired = redis_client.set(lock_key, worker_id, nx=True, ex=LOCK_TIMEOUT_SECONDS)

        if not lock_acquired:
            existing_lock_holder = redis_client.get(lock_key)
            logger.warning(f"Worker {worker_id}: Parent document {parent_doc_id} is already being processed by worker {existing_lock_holder}. Skipping task.")
            add_log_entry("WARNING", "Parent document lock already held, skipping task", document_id=parent_doc_id, step="parent_doc_lock_failed", details={"lock_key": lock_key, "current_holder": str(existing_lock_holder), **log_details}, worker_id=worker_id, original_filename=original_parent_filename)
            return

        logger.info(f"Worker {worker_id} acquired lock for parent document {parent_doc_id} ({lock_key})")
        update_parent_doc_event(parent_doc_id, "worker_lock_acquired", details={"lock_key": lock_key, "worker_id": worker_id}, original_filename=original_parent_filename)

        parent_doc_ref = db.collection("document_metadata").document(parent_doc_id)
        parent_doc_snap = parent_doc_ref.get()
        if not parent_doc_snap.exists:
            logger.error(f"Worker {worker_id}: Parent metadata document {parent_doc_id} not found. Critical error.")
            add_log_entry("ERROR", "Parent metadata not found by worker", document_id=parent_doc_id, step="fetch_parent_meta_failed", details=log_details, worker_id=worker_id, original_filename=original_parent_filename)
            return

        original_parent_filename = parent_doc_snap.to_dict().get("original_filename", original_parent_filename)
        log_details["original_parent_filename"] = original_parent_filename
        firestore_ops.update_document_status(parent_doc_id, "splitting_in_progress", f"Worker {worker_id} starting to split.")
        update_parent_doc_event(parent_doc_id, "splitting_started", status="splitting_in_progress", details={"worker_id": worker_id}, original_filename=original_parent_filename)

        logger.info(f"Worker {worker_id}: Downloading parent PDF {gcs_path_original_parent} for doc {parent_doc_id}")
        parent_pdf_bytes = gcs_service.download_blob_to_bytes(gcs_path_original_parent)

        logger.info(f"Worker {worker_id}: Splitting parent PDF {parent_doc_id} into 2-page chunks.")
        raw_pdf_chunks = pdf_utils.split_pdf(parent_pdf_bytes, chunk_size=dps.CHUNK_SIZE)

        if not raw_pdf_chunks:
            logger.error(f"Worker {worker_id}: No chunks generated for parent PDF {parent_doc_id}. Marking parent as error.")
            firestore_ops.update_document_status(parent_doc_id, "error_splitting", "Failed to split PDF or PDF has no pages.")
            update_parent_doc_event(parent_doc_id, "splitting_failed", status="error_splitting", details={"reason": "No chunks generated", "worker_id": worker_id}, original_filename=original_parent_filename)
            return

        # --- Metadata Extraction Phase (Fast Track) ---
        extracted_entity_metadata = {}
        try:
            logger.info(f"Worker {worker_id}: Starting initial metadata extraction on First Chunk for parent {parent_doc_id}")
            # Get first chunk content
            first_chunk_content, _, _ = raw_pdf_chunks[0]

            # 1. OCR First Chunk Synchronously
            first_chunk_text, conf, _ = ocr_utils.perform_ocr(pdf_chunk_content=first_chunk_content)

            if first_chunk_text and len(first_chunk_text.strip()) > 50:
                 # 2. Call Extraction Service
                 extracted_entity_metadata = MetadataExtractionService.extract_identifiers(
                     text_chunk=first_chunk_text,
                     filename=original_parent_filename
                 )
                 logger.info(f"Worker {worker_id}: Extracted Metadata: {extracted_entity_metadata}")
                 update_parent_doc_event(parent_doc_id, "metadata_extracted", details={"metadata": extracted_entity_metadata, "worker_id": worker_id}, original_filename=original_parent_filename)
            else:
                 logger.warning(f"Worker {worker_id}: First chunk OCR text empty or too short. Skipping metadata extraction.")

        except Exception as e_meta:
            logger.error(f"Worker {worker_id}: Metadata extraction failed (non-blocking): {e_meta}", exc_info=True)
            # We continue processing even if extraction fails
        # --- End Metadata Extraction ---

        created_chunk_info_list = []
        chunk_processing_errors = []

        logger.info(f"Worker {worker_id}: Processing {len(raw_pdf_chunks)} raw chunks for parent {parent_doc_id}.")
        for i, (chunk_content, start_page, end_page) in enumerate(raw_pdf_chunks):
            chunk_number = i + 1
            try:
                logger.info(f"Worker {worker_id}: Uploading chunk {chunk_number} for parent {parent_doc_id} (pages {start_page}-{end_page}).")
                chunk_id, chunk_gcs_uri, error_msg = dps.upload_chunk_to_gcs_and_create_initial_entry(
                    chunk_content=chunk_content, parent_doc_id=parent_doc_id, original_parent_filename=original_parent_filename,
                    chunk_number=chunk_number, start_page=start_page, end_page=end_page
                )
                if error_msg:
                    err = f"Failed to process raw chunk {chunk_number} for parent {parent_doc_id}: {error_msg}"
                    logger.error(f"Worker {worker_id}: {err}")
                    chunk_processing_errors.append(err)
                else:
                    created_chunk_info_list.append({
                        "chunk_id": chunk_id, "gcs_path_chunk": chunk_gcs_uri, "chunk_number": chunk_number,
                        "start_page": start_page, "end_page": end_page
                    })
                    logger.info(f"Worker {worker_id}: Successfully created chunk {chunk_id} (GCS: {chunk_gcs_uri}) for parent {parent_doc_id}.")
            except Exception as e_chunk_creation:
                err = f"Critical error creating/uploading chunk {chunk_number} for parent {parent_doc_id}: {e_chunk_creation}"
                logger.error(f"Worker {worker_id}: {err}", exc_info=True)
                chunk_processing_errors.append(err)

        num_successfully_created_chunks = len(created_chunk_info_list)
        parent_update_data = {"total_chunks": num_successfully_created_chunks, "last_status_update": datetime.datetime.now(tz=datetime.timezone.utc)}
        if extracted_entity_metadata:
             parent_update_data["extracted_metadata"] = extracted_entity_metadata # Store on parent too

        if chunk_processing_errors:
            parent_update_data["status"] = "error_creating_chunks"
            parent_update_data["error_message"] = f"Failed to create/upload one or more chunks: {'; '.join(chunk_processing_errors)}"
            firestore_ops.update_document_status(parent_doc_id, "error_creating_chunks", parent_update_data["error_message"])
            update_parent_doc_event(parent_doc_id, "chunk_creation_failed", status="error_creating_chunks", details={"errors": chunk_processing_errors, "created_count": num_successfully_created_chunks, "worker_id": worker_id}, original_filename=original_parent_filename)
            logger.error(f"Worker {worker_id}: Parent {parent_doc_id} failed during chunk creation phase.")
            return

        parent_update_data["status"] = "pending_chunk_processing"
        parent_doc_ref.update(parent_update_data)
        logger.info(f"Worker {worker_id}: Parent {parent_doc_id} updated with total_chunks: {num_successfully_created_chunks}, status: pending_chunk_processing.")
        update_parent_doc_event(parent_doc_id, "splitting_completed_all_chunks_created", status="pending_chunk_processing", details={"total_chunks": num_successfully_created_chunks, "worker_id": worker_id}, original_filename=original_parent_filename)

        if num_successfully_created_chunks > 0:
            logger.info(f"Worker {worker_id}: Starting parallel processing of {num_successfully_created_chunks} chunks for parent {parent_doc_id}.")
            update_parent_doc_event(parent_doc_id, "parallel_chunk_processing_started", details={"chunk_count": num_successfully_created_chunks, "worker_id": worker_id}, original_filename=original_parent_filename)

            completed_chunk_count_for_parent = 0
            failed_chunk_ids_for_parent = []
            max_concurrent_chunk_tasks = getattr(config, 'MAX_CONCURRENT_CHUNK_TASKS', 5)

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent_chunk_tasks) as executor:
                future_to_chunk_info = {
                    executor.submit(_process_single_chunk_task, chunk_info, parent_doc_id, original_parent_filename, worker_id, extracted_entity_metadata): chunk_info
                    for chunk_info in created_chunk_info_list
                }
                for future in concurrent.futures.as_completed(future_to_chunk_info):
                    chunk_info_processed = future_to_chunk_info[future]
                    processed_chunk_id = chunk_info_processed['chunk_id']
                    try:
                        chunk_succeeded = future.result()
                        if chunk_succeeded:
                            completed_chunk_count_for_parent += 1
                            parent_doc_ref.update({"completed_chunks": firestore.Increment(1)})
                            logger.info(f"Worker {worker_id}: Successfully processed chunk {processed_chunk_id} for parent {parent_doc_id}.")
                        else:
                            failed_chunk_ids_for_parent.append(processed_chunk_id)
                            logger.error(f"Worker {worker_id}: Failed to process chunk {processed_chunk_id} for parent {parent_doc_id}.")
                    except Exception as exc:
                        failed_chunk_ids_for_parent.append(processed_chunk_id)
                        logger.error(f"Worker {worker_id}: Chunk {processed_chunk_id} generated an exception during parallel processing: {exc}", exc_info=True)
                        _log_chunk_event(parent_doc_id, processed_chunk_id, worker_id, "chunk_processing_exception_in_executor", status="error_worker_exception", details={"error": str(exc), "traceback": traceback.format_exc()}, original_filename=original_parent_filename)

            logger.info(f"Worker {worker_id}: All parallel chunk tasks completed for parent {parent_doc_id}. Successfully processed: {completed_chunk_count_for_parent}/{num_successfully_created_chunks}")

            final_parent_status = "completed" if completed_chunk_count_for_parent == num_successfully_created_chunks else "incomplete"
            final_status_message = (f"All {num_successfully_created_chunks} chunks processed successfully by worker {worker_id}."
                                   if final_parent_status == "completed"
                                   else f"Processing incomplete. {completed_chunk_count_for_parent}/{num_successfully_created_chunks} chunks completed. Failed chunks: {len(failed_chunk_ids_for_parent)}.")
            if final_parent_status == "incomplete":
                 logger.warning(f"Worker {worker_id}: Parent {parent_doc_id} processing incomplete. Failed chunks: {failed_chunk_ids_for_parent}")

            firestore_ops.update_document_status(parent_doc_id, final_parent_status, final_status_message)
            update_parent_doc_event(parent_doc_id, "all_chunk_processing_finished", status=final_parent_status, details={"completed_chunks": completed_chunk_count_for_parent, "total_chunks": num_successfully_created_chunks, "failed_chunk_ids": failed_chunk_ids_for_parent, "worker_id": worker_id}, original_filename=original_parent_filename)
            logger.info(f"Worker {worker_id}: Parent {parent_doc_id} final status: {final_parent_status}. Message: {final_status_message}")

            # Enqueue categorization task after main processing is complete
            try:
                categorization_task_str = f"categorize_new_document::{parent_doc_id}"
                redis_client.rpush(QUEUE_NAME, categorization_task_str)
                logger.info(f"Worker {worker_id}: Enqueued post-processing categorization task for parent {parent_doc_id}.")
                update_parent_doc_event(parent_doc_id, "categorization_task_enqueued", details={"worker_id": worker_id}, original_filename=original_parent_filename)
            except Exception as e_cat_queue:
                logger.error(f"Worker {worker_id}: Failed to enqueue categorization task for parent {parent_doc_id}: {e_cat_queue}", exc_info=True)
                update_parent_doc_event(parent_doc_id, "categorization_task_enqueue_failed", status="error", details={"error": str(e_cat_queue), "worker_id": worker_id}, original_filename=original_parent_filename)

        else:
            logger.warning(f"Worker {worker_id}: No chunks were available to process for parent {parent_doc_id}. Parent status should already reflect an error.")
    except Exception as e:
        err_msg = f"Unhandled error in handle_split_and_process_pdf_task for parent {parent_doc_id}"
        logger.error(f"Worker {worker_id}: {err_msg}: {e}", exc_info=True)
        add_log_entry("ERROR", err_msg, document_id=parent_doc_id, step="split_task_unhandled_error", details={"error": str(e), "traceback": traceback.format_exc(), **log_details}, worker_id=worker_id, original_filename=original_parent_filename)
        if parent_doc_id:
            try:
                firestore_ops.update_document_status(parent_doc_id, "error_worker_failure", f"Core worker error: {e}")
                update_parent_doc_event(parent_doc_id, "worker_unhandled_exception", status="error_worker_failure", details={"error": str(e), "worker_id": worker_id}, original_filename=original_parent_filename)
            except Exception as update_e:
                logger.error(f"Worker {worker_id}: Failed to update error status for parent {parent_doc_id} after unhandled task error: {update_e}")
    finally:
        if lock_acquired and lock_key and redis_client:
            try:
                current_lock_holder = redis_client.get(lock_key)
                if current_lock_holder == worker_id:
                    deleted_count = redis_client.delete(lock_key)
                    if deleted_count > 0:
                        logger.info(f"Worker {worker_id} released lock for parent document {parent_doc_id} ({lock_key})")
                        add_log_entry("INFO", "Released parent document processing lock", document_id=parent_doc_id, step="parent_doc_lock_released", details={"lock_key": lock_key, **log_details}, worker_id=worker_id, original_filename=original_parent_filename)
                    else:
                        logger.warning(f"Worker {worker_id} attempted to release lock {lock_key} but it was not found (or not held by this worker).")
                        add_log_entry("WARNING", "Lock release attempt failed (not found or not owner)", document_id=parent_doc_id, step="parent_doc_lock_release_failed_not_found", details={"lock_key": lock_key, **log_details}, worker_id=worker_id)
                elif current_lock_holder is None:
                    logger.info(f"Worker {worker_id}: Lock {lock_key} for parent {parent_doc_id} was already released/expired.")
                    add_log_entry("INFO", "Lock already released/expired before explicit release", document_id=parent_doc_id, step="parent_doc_lock_already_released", details={"lock_key": lock_key, **log_details}, worker_id=worker_id)
                else:
                    logger.error(f"Worker {worker_id} detected lock {lock_key} for parent {parent_doc_id} is held by another worker ({current_lock_holder}) at time of release. This should not happen if locking logic is correct.")
                    add_log_entry("ERROR", "Lock held by another worker at release time", document_id=parent_doc_id, step="parent_doc_lock_held_by_other_at_release", details={"lock_key": lock_key, "expected_holder": worker_id, "actual_holder": str(current_lock_holder), **log_details}, worker_id=worker_id)
            except Exception as release_e:
                logger.error(f"Worker {worker_id}: Error releasing Redis lock {lock_key} for parent {parent_doc_id}: {release_e}", exc_info=True)
                add_log_entry("ERROR", "Error releasing lock", document_id=parent_doc_id, step="parent_doc_lock_release_error", details={"lock_key": lock_key, "error": str(release_e), **log_details}, worker_id=worker_id)
        elif lock_key and not redis_client:
             logger.error(f"Worker {worker_id}: Cannot release lock {lock_key} because Redis client is not available.")


def handle_reprocess_legacy_doc_chunks(task_str: str, log_details_base: dict, redis_client, worker_id: str):
    """
    Handles selective processing of legacy chunks for a document.
    """
    doc_id = None
    log_details = {**log_details_base}
    try:
        _, doc_id = task_str.split('::', 1)
        log_details["doc_id"] = doc_id
        logger.info(f"Worker {worker_id}: Received reprocess request for doc {doc_id}")
        
        # 1. Fetch chunks pending reprocess
        # Some chunks store the parent reference in `original_doc_firestore_id` instead of `parent_doc_id`.
        # Query both and merge to avoid missing records.
        chunks_to_process = []
        seen_chunk_ids = set()

        chunks_queries = [
            db.collection("document_chunks")
            .where("parent_doc_id", "==", doc_id)
            .where("status", "==", "pending_reprocess"),
            db.collection("document_chunks")
            .where("original_doc_firestore_id", "==", doc_id)
            .where("status", "==", "pending_reprocess"),
        ]

        for query in chunks_queries:
            for doc in query.stream():
                if doc.id in seen_chunk_ids:
                    continue
                seen_chunk_ids.add(doc.id)
                chunk_data = doc.to_dict()
                chunks_to_process.append({
                    "chunk_id": doc.id,
                    "gcs_path_chunk": chunk_data.get("gcs_path_chunk"),
                    "start_page": chunk_data.get("start_page"),
                    "end_page": chunk_data.get("end_page")
                })
            
        if not chunks_to_process:
            logger.info(f"Worker {worker_id}: No chunks found with status 'pending_reprocess' for doc {doc_id}.")
            # Check if all chunks are in fact completed
            # ...
            return

        logger.info(f"Worker {worker_id}: Found {len(chunks_to_process)} chunks to reprocess for doc {doc_id}.")

        # Get original filename for logging
        parent_doc = db.collection("document_metadata").document(doc_id).get()
        original_filename = parent_doc.to_dict().get("original_filename", "unknown") if parent_doc.exists else "unknown"

        # 2. Process in parallel
        max_concurrent = getattr(config, 'MAX_CONCURRENT_CHUNK_TASKS', 5)
        completed_count = 0
        failed_chunks = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            future_to_chunk = {
                executor.submit(_process_single_chunk_task, chunk_info, doc_id, original_filename, worker_id, None): chunk_info
                for chunk_info in chunks_to_process
            }
            
            for future in concurrent.futures.as_completed(future_to_chunk):
                chunk_info = future_to_chunk[future]
                c_id = chunk_info['chunk_id']
                try:
                    success = future.result()
                    if success:
                        completed_count += 1
                        logger.info(f"Worker {worker_id}: Reprocessed chunk {c_id} successfully.")
                    else:
                        failed_chunks.append(c_id)
                        logger.error(f"Worker {worker_id}: Failed to reprocess chunk {c_id}.")
                except Exception as e:
                    failed_chunks.append(c_id)
                    logger.error(f"Worker {worker_id}: Exception reprocessing chunk {c_id}: {e}")

        # 3. Finalize
        logger.info(f"Worker {worker_id}: Finished reprocessing for doc {doc_id}. Success: {completed_count}, Failed: {len(failed_chunks)}.")
        
        # Check overall status
        all_chunks_completed = True
        # Simple check: query if any chunks are NOT completed
        not_completed_query = db.collection("document_chunks") \
            .where("parent_doc_id", "==", doc_id) \
            .where("status", "!=", "completed").limit(1).stream()
        
        if any(not_completed_query):
             all_chunks_completed = False

        if all_chunks_completed:
            firestore_ops.update_document_status(doc_id, "completed", "All chunks reprocessed and up to date.")
            logger.info(f"Worker {worker_id}: Document {doc_id} marked as completed.")
        else:
             firestore_ops.update_document_status(doc_id, "incomplete", f"Reprocessing finished but some chunks failed or are pending. Failed this run: {len(failed_chunks)}.")

    except Exception as e:
        logger.error(f"Worker {worker_id}: Error in handle_reprocess_legacy_doc_chunks for {doc_id}: {e}", exc_info=True)


def _log_chunk_event(parent_doc_id: str, chunk_id: str, worker_id: str, event_name: str, level: str = "INFO", status: Optional[str] = None, details: Optional[dict] = None, message_override: Optional[str] = None, original_filename: Optional[str] = None):
    if not chunk_id:
        logger.warning(f"Attempted to log event '{event_name}' but chunk_id is None.")
        return
    log_message = message_override if message_override else f"Chunk Event: {event_name}"
    if status:
        log_message += f" (Status after: {status})"
    log_details_to_save = { "event_name": event_name, "status_after_event": status, **(details or {}) }
    try:
        add_log_entry( level=level.upper(), message=log_message, document_id=parent_doc_id, chunk_id=chunk_id, step=event_name, original_filename=original_filename, details=log_details_to_save, worker_id=worker_id )
        db.collection("document_chunks").document(chunk_id).update({"last_updated": datetime.datetime.now(tz=datetime.timezone.utc)})
    except Exception as e:
        logger.error(f"Failed to add chunk log event '{event_name}' for chunk {chunk_id} to processing_logs: {e}", exc_info=True)

def _process_single_chunk_task(chunk_info: dict, parent_doc_id: str, original_parent_filename: str, worker_id: str, entity_metadata: Optional[dict] = None) -> bool:
    """
    Process a single chunk with PER-CHUNK entity extraction.

    This is the core processing function that:
    1. Classifies the chunk's document type (Gemini)
    2. Performs OCR with the appropriate parser (Document AI)
    3. Extracts entities SPECIFIC TO THIS CHUNK using type-specific schema (Gemini)
    4. Generates embedding and upserts to vector index with chunk-specific restrictions

    Args:
        chunk_info: Dictionary with chunk_id, gcs_path_chunk, start_page, end_page
        parent_doc_id: Parent document Firestore ID
        original_parent_filename: Original filename for context
        worker_id: Worker identifier for logging
        entity_metadata: DEPRECATED - No longer used. Each chunk extracts its own entities.

    Returns:
        True if processing succeeded, False otherwise
    """
    chunk_id = chunk_info['chunk_id']
    chunk_gcs_path = chunk_info['gcs_path_chunk']
    start_page = chunk_info['start_page']
    end_page = chunk_info['end_page']
    log_prefix = f"Worker {worker_id}, Parent {parent_doc_id}, Chunk {chunk_id} (p{start_page}-{end_page}):"
    logger.info(f"{log_prefix} Starting processing with per-chunk entity extraction.")

    # This will hold the chunk's own extracted entities
    chunk_entity_metadata = {}

    try:
        # Step 8: Gemini Parser Identification (using GCS URI)
        firestore_ops.update_chunk_status(chunk_id, "processing_classification", "Starting Gemini classification using GCS URI")
        _log_chunk_event(parent_doc_id, chunk_id, worker_id, "classification_started_gcs_uri", status="processing_classification", original_filename=original_parent_filename)
        logger.info(f"{log_prefix} Calling Gemini for parser selection using GCS URI: {chunk_gcs_path}.")

        # Unpack three values now
        classified_doc_type, selected_parser_id, raw_gemini_response = classification_utils.get_parser_for_chunk_via_gemini(
            chunk_gcs_uri=chunk_gcs_path,
            chunk_id=chunk_id,
            parent_doc_id=parent_doc_id,
            worker_id=worker_id,
            original_filename=original_parent_filename
        )
        # If classification fails or defaults, selected_parser_id will be the default from classification_utils

        db.collection("document_chunks").document(chunk_id).update({
            "selected_parser_processor_id": selected_parser_id,
            "classified_document_type": classified_doc_type,
            "status": "pending_ocr",
            "last_updated": datetime.datetime.now(tz=datetime.timezone.utc)
        })
        _log_chunk_event(
            parent_doc_id, chunk_id, worker_id,
            "classification_completed_gcs_uri",
            status="pending_ocr",
            details={
                "selected_parser": selected_parser_id,
                "classified_type": classified_doc_type,
                "gemini_raw_choice": raw_gemini_response
            },
            original_filename=original_parent_filename
        )
        logger.info(f"{log_prefix} Classification via GCS URI complete. Selected parser: {selected_parser_id}, Type: {classified_doc_type}, Gemini Raw: {raw_gemini_response}. Status set to pending_ocr.")

        # Step 9 & 10: Multimodal Content Extraction & Entity Recognition (Gemini Vision)
        # We perform BOTH transcription and entity extraction in one call for efficiency.
        firestore_ops.update_chunk_status(chunk_id, "processing_content", "Extracting full content and entities with Gemini Vision")
        _log_chunk_event(parent_doc_id, chunk_id, worker_id, "content_extraction_started", status="processing_content", original_filename=original_parent_filename)

        logger.info(f"{log_prefix} Downloading chunk content from {chunk_gcs_path}")
        chunk_pdf_bytes = gcs_service.download_blob_to_bytes(chunk_gcs_path)
        
        # === DEBUG: Track extraction flow ===
        debug_log("[EXTRACTION] ========== WORKER: Starting extraction for chunk ==========")
        debug_log("[EXTRACTION] Chunk info:",
                  chunk_id=chunk_id,
                  parent_doc_id=parent_doc_id,
                  classified_doc_type=classified_doc_type,
                  pdf_bytes_size=len(chunk_pdf_bytes) if chunk_pdf_bytes else 0)

        gemini_exception = None
        chunk_entity_metadata = {}
        for attempt in range(1, 4):
            try:
                logger.info(f"{log_prefix} Starting MULTIMODAL entity extraction and transcription for doc_type: {classified_doc_type} (attempt {attempt})")
                debug_log("[EXTRACTION] Calling TypeSpecificExtractionService.extract_entities_multimodal...", attempt=attempt)

                # Use MULTIMODAL (Vision) extraction - sends PDF directly to Gemini
                # This now returns BOTH extracted entities and a _full_transcription field
                chunk_entity_metadata = TypeSpecificExtractionService.extract_entities_multimodal(
                    pdf_bytes=chunk_pdf_bytes,  # Send the actual PDF chunk!
                    doc_type=classified_doc_type or "OTHER",
                    filename=original_parent_filename
                )
                
                debug_log("[EXTRACTION] extract_entities_multimodal returned",
                          metadata_keys=list(chunk_entity_metadata.keys()) if chunk_entity_metadata else [],
                          has_transcription="_full_transcription" in chunk_entity_metadata if chunk_entity_metadata else False,
                          attempt=attempt)
                
                # Check if extraction actually succeeded (has meaningful data)
                if chunk_entity_metadata and len(chunk_entity_metadata) > 0:
                    gemini_exception = None
                    logger.info(f"{log_prefix} Gemini extraction succeeded on attempt {attempt}")
                    break
                else:
                    # Empty result, treat as failure and retry
                    raise Exception("Gemini extraction returned empty result")
                    
            except Exception as e_extract:
                gemini_exception = e_extract
                logger.warning(f"{log_prefix} Gemini Multimodal extraction attempt {attempt} failed: {e_extract}")
                if attempt < 3:  # Only sleep if we're going to retry
                    wait_time = 2 if attempt == 1 else 5
                    logger.info(f"{log_prefix} Waiting {wait_time} seconds before retry attempt {attempt + 1}...")
                    time.sleep(wait_time)

        if gemini_exception:
            logger.warning(f"{log_prefix} Gemini Multimodal extraction failed after 3 attempts, falling back to Document AI: {gemini_exception}")
            # Fallback to standard OCR if the vision call fails entirely
            ocr_text, confidence_score, docai_extracted_entities = ocr_utils.perform_ocr(
                pdf_chunk_content=chunk_pdf_bytes,
                processor_name_override=selected_parser_id
            )
            final_doc_type = classified_doc_type or "OTHER"
            chunk_entity_metadata = {"_doc_type": final_doc_type, "_extraction_failed": True, "extraction_source": "legacy"}

            _log_chunk_event(
                parent_doc_id, chunk_id, worker_id,
                "gemini_extraction_failed_using_docai_fallback",
                level="WARNING",
                status="processing_entity_extraction",
                details={"error": str(gemini_exception), "attempts": 3},
                original_filename=original_parent_filename
            )
        else:
            # Use the DETECTED category from vision extraction
            final_doc_type = chunk_entity_metadata.get("_detected_category") or chunk_entity_metadata.get("_doc_type") or classified_doc_type or "OTHER"
            debug_log("[EXTRACTION] Final detected category:", final_doc_type=final_doc_type)

            # Update metadata
            chunk_entity_metadata["_doc_type"] = final_doc_type
            chunk_entity_metadata["_chunk_id"] = chunk_id
            chunk_entity_metadata["_initial_classification"] = classified_doc_type
            chunk_entity_metadata["extraction_source"] = "llm"

            # Get Gemini transcription for vector search
            ocr_text = chunk_entity_metadata.get("_full_transcription", "")
            confidence_score = 0.98 if ocr_text else None # High confidence for Gemini Vision
            docai_extracted_entities = {} # Document AI skipped unless fallback used

            # Robust check for extraction method field
            extraction_method = chunk_entity_metadata.get("_extraction_method") or chunk_entity_metadata.get("extraction_method", "text")
            entity_count = len([k for k in chunk_entity_metadata.keys() if not k.startswith("_")])
            logger.info(f"{log_prefix} Gemini Vision extraction complete ({extraction_method}). Category: {final_doc_type}. Extracted {entity_count} entities. Transcription length: {len(ocr_text)}")

            # Fallback to Document AI if Gemini provided no text (rare but possible)
            if not ocr_text or len(ocr_text.strip()) < 10:
                logger.warning(f"{log_prefix} Gemini transcription empty or too short, falling back to Document AI OCR")
                ocr_text, confidence_score, docai_extracted_entities = ocr_utils.perform_ocr(
                    pdf_chunk_content=chunk_pdf_bytes,
                    processor_name_override=selected_parser_id
                )

            _log_chunk_event(
                parent_doc_id, chunk_id, worker_id,
                "content_extraction_completed",
                status="processing_entity_extraction",
                details={
                    "initial_doc_type": classified_doc_type,
                    "final_doc_type": final_doc_type,
                    "entity_count": entity_count,
                    "extraction_method": extraction_method,
                    "transcription_source": "gemini" if ocr_text and not docai_extracted_entities else "docai"
                },
                original_filename=original_parent_filename
            )

        # Step 10b: Save results and entities to chunk entry in Firestore
        # NEW: Flatten entity values for deterministic search (array-contains queries)
        searchable_entities = set() # Use a set to avoid duplicates
        
        def collect_searchable_values(data_dict):
            if not isinstance(data_dict, dict):
                return
            for key, value in data_dict.items():
                if key.startswith('_'):
                    continue
                if value is None:
                    continue
                if isinstance(value, (str, int, float)):
                    val_str = str(value).strip()
                    if val_str:
                        searchable_entities.add(val_str)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, (str, int, float)):
                            val_str = str(item).strip()
                            if val_str:
                                searchable_entities.add(val_str)

        if chunk_entity_metadata:
            collect_searchable_values(chunk_entity_metadata)
            
            # Special case for multi-document chunks: also index sub-documents
            if "_documents" in chunk_entity_metadata and isinstance(chunk_entity_metadata["_documents"], list):
                for doc in chunk_entity_metadata["_documents"]:
                    collect_searchable_values(doc)

        searchable_entities_list = list(searchable_entities)


        extraction_source = chunk_entity_metadata.get("extraction_source", "legacy")

        chunk_update_data_ocr = {
            "ocr_text_preview": ocr_text if ocr_text else "", # Limit preview size
            "extracted_text_preview": ocr_text if ocr_text else "", # Dual write for UI consistency
            "full_text_gcs_path": None,
            "ocr_confidence_score": confidence_score,
            "docai_extracted_entities": docai_extracted_entities if docai_extracted_entities else {},
            "extraction_source": extraction_source, # NEW: Top-level field for clean querying
            "extracted_entities": chunk_entity_metadata,
            "entities": searchable_entities_list, # NEW: Searchable array for hard-match
            "classified_document_type": final_doc_type,
            "status": "pending_embedding",
            "last_updated": datetime.datetime.now(tz=datetime.timezone.utc)
        }
        db.collection("document_chunks").document(chunk_id).update(chunk_update_data_ocr)

        # --- DUAL WRITE: Save heavy details to separate collection ---
        try:
            from app.models.metadata_model import save_chunk_details
            
            debug_log("[EXTRACTION] Building vector restrictions for chunk:", chunk_id=chunk_id)

            # Build vector restrictions from chunk's OWN entities using DETECTED category
            chunk_restricts = TypeSpecificExtractionService.get_vector_restrictions(
                entities=chunk_entity_metadata,
                doc_type=final_doc_type,  # Use detected category
                doc_id=parent_doc_id
            )
            
            debug_log("[EXTRACTION] Vector restrictions built:", 
                      restriction_count=len(chunk_restricts) if chunk_restricts else 0)

            details_payload = {
                "chunk_id": chunk_id,
                "full_text": ocr_text,
                "entity_metadata": chunk_entity_metadata,  # Chunk's own entities
                "docai_entities": docai_extracted_entities if docai_extracted_entities else {},
                "vector_restricts": chunk_restricts,
                "initial_classified_doc_type": classified_doc_type,  # Original classification
                "classified_doc_type": final_doc_type,  # Final detected category
                "processing_timestamp": datetime.datetime.now(tz=datetime.timezone.utc),
                "worker_id": worker_id,
                "original_filename": original_parent_filename,
                "extraction_version": "2.4"  # v2.4: Structured Markdown transcription with strict 11 categories
            }
            save_chunk_details(chunk_id, details_payload)
            logger.info(f"{log_prefix} Saved rich chunk details to 'document_chunk_details' collection.")
            debug_log("[EXTRACTION] Chunk details saved to document_chunk_details collection")
        except Exception as e_details:
            logger.error(f"{log_prefix} Failed to save chunk details: {e_details}")
            debug_log("[EXTRACTION] ERROR saving chunk details:", error=str(e_details))
        # -------------------------------------------------------------

        _log_chunk_event(parent_doc_id, chunk_id, worker_id, "ocr_completed", status="pending_embedding", details={"text_length": len(ocr_text) if ocr_text else 0, "confidence": confidence_score}, original_filename=original_parent_filename)
        logger.info(f"{log_prefix} OCR results saved. Status set to pending_embedding.")

        if not ocr_text or not ocr_text.strip():
            logger.warning(f"{log_prefix} No text extracted from OCR. Marking as completed without embedding.")
            debug_log("[EXTRACTION] No text extracted, skipping embedding")
            firestore_ops.update_chunk_status(
                chunk_id=chunk_id, status="completed_no_text_for_embedding",
                status_message="No text from OCR to embed.", has_embedding=False, embedding_status="skipped"
            )
            _log_chunk_event(parent_doc_id, chunk_id, worker_id, "embedding_skipped_no_text", status="completed_no_text_for_embedding", original_filename=original_parent_filename)
            return True

        # Step 11: Embed text and store in vector database
        firestore_ops.update_chunk_status(chunk_id, "processing_embedding", "Generating text embedding.")
        _log_chunk_event(parent_doc_id, chunk_id, worker_id, "embedding_started", status="processing_embedding", original_filename=original_parent_filename)
        logger.info(f"{log_prefix} Generating embedding.")
        embedding_vector = vais.get_text_embedding(ocr_text)

        # Use the CHUNK's OWN entity metadata for vector restrictions
        logger.info(f"{log_prefix} Embedding generated. Upserting to vector index with CHUNK-SPECIFIC metadata: {list(chunk_entity_metadata.keys())}")
        vais.add_embedding_to_index(
            datapoint_id=chunk_id,
            embedding=embedding_vector,
            filename=original_parent_filename,
            start_page=start_page,
            end_page=end_page,
            gcs_uri=chunk_gcs_path,
            text_preview=ocr_text,
            doc_id=parent_doc_id,
            entity_metadata=chunk_entity_metadata  # CHUNK's OWN entities!
        )
        _log_chunk_event(parent_doc_id, chunk_id, worker_id, "embedding_vector_upserted", status="processing_embedding", details={"entity_fields": list(chunk_entity_metadata.keys())}, original_filename=original_parent_filename)

        # Step 12: Mark chunk as completed
        firestore_ops.update_chunk_status(
            chunk_id=chunk_id, status="completed", status_message="Chunk processing successful with per-chunk entity extraction.",
            has_embedding=True, embedding_status="completed"
        )
        _log_chunk_event(parent_doc_id, chunk_id, worker_id, "chunk_completed_successfully", status="completed", original_filename=original_parent_filename)
        logger.info(f"{log_prefix} Successfully processed and vectorized with per-chunk entities.")
        return True

    except Exception as e_chunk_proc:
        error_message = f"Error processing chunk {chunk_id} for parent {parent_doc_id}: {str(e_chunk_proc)}"
        logger.error(f"{log_prefix} {error_message}", exc_info=True)
        final_chunk_status = "error_processing_chunk"
        if "embedding" in str(e_chunk_proc).lower() or isinstance(e_chunk_proc, vais.EmbeddingError):
            final_chunk_status = "failed_embedding_dlq"
            try:
                dlq_ref = db.collection("failed_embedding_chunks").document(chunk_id)
                dlq_ref.set({
                    "chunk_id": chunk_id, "parent_doc_id": parent_doc_id, "gcs_path_chunk": chunk_gcs_path,
                    "original_parent_filename": original_parent_filename, "error_message": str(e_chunk_proc),
                    "traceback": traceback.format_exc(), "failed_at_timestamp": datetime.datetime.now(tz=datetime.timezone.utc),
                    "worker_id": worker_id
                })
                logger.info(f"{log_prefix} Added chunk to embedding DLQ.")
                _log_chunk_event(parent_doc_id, chunk_id, worker_id, "added_to_embedding_dlq", level="ERROR", status=final_chunk_status, details={"error": str(e_chunk_proc)}, original_filename=original_parent_filename)
            except Exception as e_dlq:
                logger.error(f"{log_prefix} Failed to add chunk to DLQ: {e_dlq}", exc_info=True)
                _log_chunk_event(parent_doc_id, chunk_id, worker_id, "dlq_add_failed", level="ERROR", status="error_processing_chunk", details={"dlq_error": str(e_dlq), "original_error": str(e_chunk_proc)}, original_filename=original_parent_filename)
        try:
            firestore_ops.update_chunk_status(chunk_id, final_chunk_status, error_message)
            _log_chunk_event(parent_doc_id, chunk_id, worker_id, "chunk_processing_failed", level="ERROR", status=final_chunk_status, details={"error": str(e_chunk_proc)}, original_filename=original_parent_filename)
        except Exception as e_status_update:
            logger.error(f"{log_prefix} Failed to update chunk status to error: {e_status_update}", exc_info=True)
            add_log_entry("ERROR", f"Failed to update error status for chunk {chunk_id}", document_id=parent_doc_id, chunk_id=chunk_id, step="update_chunk_error_status_failed", details={"original_error": str(e_chunk_proc), "update_error": str(e_status_update)}, worker_id=worker_id, original_filename=original_parent_filename)
        return False

def check_and_update_root_doc_completion_wrapper(doc_id: str, worker_id: str):
     logger.debug(f"Worker {worker_id}: Skipping redundant completion check for doc {doc_id}.")
     pass

def handle_categorization_task(task_str: str, log_details_base: dict, worker_id: str):
    """Handles both backfill and new document categorization tasks."""
    doc_id = None
    log_details = {**log_details_base}
    try:
        task_type, doc_id = task_str.split('::', 1)
        log_details.update({"doc_id": doc_id, "task_type": task_type})

        logger.info(f"Worker {worker_id}: Starting task '{task_type}' for doc_id: {doc_id}")
        add_log_entry("INFO", f"Starting categorization task: {task_type}", document_id=doc_id, step="categorization_started", details=log_details, worker_id=worker_id)

        # Choose categorization path based on presence of destination_path (scheduled batch files)
        try:
            doc_ref = db.collection("document_metadata").document(doc_id)
            doc_snap = doc_ref.get()
            doc_data = doc_snap.to_dict() if doc_snap.exists else {}
            destination_path = doc_data.get("destination_path")

            if destination_path:
                # For batch-processed docs, use strict/destination-based categorization first
                categorization_service.categorize_document_strict(doc_id)
            else:
                # Default path (manual uploads or no destination available): full categorization
                categorization_service.categorize_document(doc_id)
        except Exception:
            # Fallback to existing behavior to avoid breaking flows
            categorization_service.categorize_document(doc_id)

        logger.info(f"Worker {worker_id}: Finished task '{task_type}' for doc_id: {doc_id}")
        add_log_entry("SUCCESS", f"Finished categorization task: {task_type}", document_id=doc_id, step="categorization_finished", details=log_details, worker_id=worker_id)

    except ValueError:
        err_msg = f"Invalid categorization task format: {task_str}"
        logger.error(err_msg)
        add_log_entry("ERROR", err_msg, step="categorization_task_parse_error", details=log_details, worker_id=worker_id)
    except Exception as e:
        err_msg = f"Unhandled error in handle_categorization_task for doc {doc_id}"
        logger.error(f"Worker {worker_id}: {err_msg}: {e}", exc_info=True)
        add_log_entry("ERROR", err_msg, document_id=doc_id, step="categorization_unhandled_error", details={"error": str(e), "traceback": traceback.format_exc(), **log_details}, worker_id=worker_id)


def handle_initiate_bulk_reprocess_legacy(task_str: str, log_details_base: dict, worker_id: str):
    """
    Handles the orchestration task for bulk legacy reprocessing.
    Supports both NEW format (3rd param is trigger_source) and OLD format (3rd param is skip_aged)
    NEW: initiate_bulk_reprocess_legacy::{user_email}::{trigger_source}[::{skip_aged_chunks}::{max_age_days}]
    OLD: initiate_bulk_reprocess_legacy::{user_email}[::{skip_aged_chunks}::{max_age_days}]
    """
    try:
        parts = task_str.split('::')
        user_email = parts[1] if len(parts) > 1 else "unknown"
        
        trigger_source = "MANUAL"
        skip_aged_chunks = False
        max_age_days = None

        if len(parts) > 2:
            p2 = str(parts[2]).strip()
            # If p2 looks like a boolean, it's the OLD format
            if p2.lower() in {"true", "false", "1", "0", "yes", "no", "y", "n"}:
                trigger_source = "MANUAL" # Default for old format
                skip_aged_chunks = p2.lower() in {"true", "1", "yes", "y"}
                if len(parts) > 3:
                    try:
                        max_age_days = int(parts[3])
                    except ValueError:
                        max_age_days = None
            else:
                # NEW format
                trigger_source = p2
                if len(parts) > 3:
                    skip_val = str(parts[3]).strip().lower()
                    skip_aged_chunks = skip_val in {"true", "1", "yes", "y"}
                if len(parts) > 4:
                    try:
                        max_age_days = int(parts[4])
                    except ValueError:
                        max_age_days = None
        
        logger.info(f"Worker {worker_id}: Initiating bulk legacy reprocess job for {user_email} (trigger: {trigger_source}, skip_aged: {skip_aged_chunks})")
        bulk_processing_utils.start_legacy_reprocess_job(
            user_email,
            skip_aged_chunks=skip_aged_chunks,
            max_age_days=max_age_days,
            trigger_source=trigger_source
        )
        logger.info(f"Worker {worker_id}: Bulk legacy reprocess job finished.")
        
    except Exception as e:
        logger.error(f"Worker {worker_id}: Error in initiate_bulk_reprocess_legacy: {e}", exc_info=True)


if __name__ == "__main__":
    main()
