import uuid
import hashlib
import io
import os
from datetime import datetime, timezone
from google.cloud import storage, firestore
from PyPDF2 import PdfReader, errors as PyPDF2Errors
from app.utils.redis_client import get_redis_client, QUEUE_NAME
from app import db, config
from app.services import gcs_service
from app.models.metadata_model import create_doc_metadata
from app.utils.utils import update_search_keywords_by_doc_id
# No longer importing from routes
# from app.routes.document_routes import force_reprocess_document
import logging
from threading import Thread
from app.utils.debug_logger import debug_log, debug_warn, debug_error

logger = logging.getLogger(__name__)

def add_ignored_folder(folder_name):
    """Adds a folder to the ignored list in Firestore."""
    try:
        # Ensure folder_name ends with a '/' for consistent matching
        if not folder_name.endswith('/'):
            folder_name += '/'
        
        ignored_ref = db.collection('ignored_folders').document(folder_name.replace('/', '_')) # Use a sanitized name as doc ID
        ignored_ref.set({
            'folder_path': folder_name,
            'added_timestamp': firestore.SERVER_TIMESTAMP
        })
        logger.info(f"Added ignored folder: {folder_name}")
    except Exception as e:
        logger.error(f"Error adding ignored folder {folder_name}: {e}", exc_info=True)
        raise

def get_ignored_folders():
    """Retrieves all ignored folders from Firestore."""
    try:
        ignored_folders = []
        for doc in db.collection('ignored_folders').stream():
            ignored_folders.append(doc.to_dict().get('folder_path'))
        return ignored_folders
    except Exception as e:
        logger.error(f"Error getting ignored folders: {e}", exc_info=True)
        return []

def remove_ignored_folder(folder_name):
    """Removes a folder from the ignored list in Firestore."""
    try:
        # Ensure folder_name ends with a '/' for consistent matching
        if not folder_name.endswith('/'):
            folder_name += '/'
        
        doc_id = folder_name.replace('/', '_')
        ignored_ref = db.collection('ignored_folders').document(doc_id)
        
        # Check if the document exists before deleting
        if ignored_ref.get().exists:
            ignored_ref.delete()
            logger.info(f"Removed ignored folder: {folder_name}")
            return True
        else:
            logger.warning(f"Ignored folder not found: {folder_name}")
            return False
    except Exception as e:
        logger.error(f"Error removing ignored folder {folder_name}: {e}", exc_info=True)
        raise

def list_gcs_buckets():
    """Lists all buckets in the current project."""
    try:
        storage_client = storage.Client()
        buckets = storage_client.list_buckets()
        return [bucket.name for bucket in buckets]
    except Exception as e:
        debug_error(f"Error listing GCS buckets: {e}")
        return []

def save_default_bucket(bucket_name, user_id):
    """Saves the default bucket name to Firestore."""
    try:
        config_ref = db.collection('batch_processing_config').document('default_config')
        config_ref.set({
            'default_bucket_name': bucket_name,
            'last_updated_by': user_id,
            'last_updated_timestamp': firestore.SERVER_TIMESTAMP
        }, merge=True)
    except Exception as e:
        debug_error(f"Error saving default bucket: {e}")
        raise

def get_default_bucket():
    """Retrieves the default bucket name from Firestore."""
    try:
        config_ref = db.collection('batch_processing_config').document('default_config')
        config_doc = config_ref.get()
        if config_doc.exists:
            return config_doc.to_dict().get('default_bucket_name')
        return None
    except Exception as e:
        debug_error(f"Error getting default bucket: {e}")
        return None

def set_legacy_reprocess_enabled(enabled: bool, user_id: str):
    """Enable or disable nightly legacy reprocess runs."""
    try:
        config_ref = db.collection('batch_processing_config').document('default_config')
        config_ref.set({
            'legacy_reprocess_enabled': bool(enabled),
            'legacy_reprocess_updated_by': user_id,
            'legacy_reprocess_updated_at': firestore.SERVER_TIMESTAMP
        }, merge=True)
    except Exception as e:
        debug_error(f"Error saving legacy reprocess setting: {e}")
        raise

def get_legacy_reprocess_enabled(default: bool = True) -> bool:
    """Gets whether nightly legacy reprocess is enabled. Defaults to True."""
    try:
        config_ref = db.collection('batch_processing_config').document('default_config')
        config_doc = config_ref.get()
        if not config_doc.exists:
            return default
        value = config_doc.to_dict().get('legacy_reprocess_enabled')
        if value is None:
            return default
        return bool(value)
    except Exception as e:
        debug_error(f"Error getting legacy reprocess setting: {e}")
        return default

def get_legacy_reprocess_config() -> dict:
    """Returns legacy reprocess config details for admin UIs."""
    try:
        config_ref = db.collection('batch_processing_config').document('default_config')
        config_doc = config_ref.get()
        if not config_doc.exists:
            return {
                "legacy_reprocess_enabled": True,
                "legacy_reprocess_updated_by": None,
                "legacy_reprocess_updated_at": None
            }
        data = config_doc.to_dict()
        return {
            "legacy_reprocess_enabled": bool(data.get("legacy_reprocess_enabled", True)),
            "legacy_reprocess_updated_by": data.get("legacy_reprocess_updated_by"),
            "legacy_reprocess_updated_at": data.get("legacy_reprocess_updated_at")
        }
    except Exception as e:
        debug_error(f"Error getting legacy reprocess config: {e}")
        return {
            "legacy_reprocess_enabled": True,
            "legacy_reprocess_updated_by": None,
            "legacy_reprocess_updated_at": None
        }

def get_runs():
    """Retrieves all batch processing runs from Firestore."""
    try:
        runs_ref = db.collection('batch_process_runs').order_by('start_timestamp', direction=firestore.Query.DESCENDING).limit(10)
        runs_data = []
        for doc in runs_ref.stream():
            run = doc.to_dict()
            # Convert datetime objects to the expected format for JSON serialization
            if 'start_timestamp' in run and isinstance(run['start_timestamp'], datetime):
                run['start_timestamp'] = {
                    '_seconds': int(run['start_timestamp'].timestamp()),
                    '_nanoseconds': run['start_timestamp'].microsecond * 1000
                }
            if 'end_timestamp' in run and run.get('end_timestamp') and isinstance(run['end_timestamp'], datetime):
                run['end_timestamp'] = {
                    '_seconds': int(run['end_timestamp'].timestamp()),
                    '_nanoseconds': run['end_timestamp'].microsecond * 1000
                }
            runs_data.append(run)
        return runs_data
    except Exception as e:
        debug_error(f"Error getting batch runs: {e}")
        return []

def get_files_for_run(run_id, limit=100, start_after_doc_id=None):
    """Retrieves a paginated list of files for a specific batch run."""
    try:
        query = db.collection('batch_processed_files').where('parent_run_id', '==', run_id).order_by('first_seen_timestamp').limit(limit)
        
        if start_after_doc_id:
            start_after_doc = db.collection('batch_processed_files').document(start_after_doc_id).get()
            if start_after_doc.exists:
                query = query.start_after(start_after_doc)

        files_data = []
        for doc in query.stream():
            file_doc = doc.to_dict()
            if 'first_seen_timestamp' in file_doc and isinstance(file_doc['first_seen_timestamp'], datetime):
                file_doc['first_seen_timestamp'] = {
                    '_seconds': int(file_doc['first_seen_timestamp'].timestamp()),
                    '_nanoseconds': file_doc['first_seen_timestamp'].microsecond * 1000
                }
            if 'last_checked_timestamp' in file_doc and isinstance(file_doc['last_checked_timestamp'], datetime):
                file_doc['last_checked_timestamp'] = {
                    '_seconds': int(file_doc['last_checked_timestamp'].timestamp()),
                    '_nanoseconds': file_doc['last_checked_timestamp'].microsecond * 1000
                }
            files_data.append(file_doc)

        files = files_data
        
        next_cursor = None
        if len(files) == limit:
            last_doc_id = files[-1].get('file_gcs_path') # Assuming this is unique enough for cursor
            if last_doc_id:
                file_hash = hashlib.sha256(last_doc_id.encode('utf-8')).hexdigest()
                next_cursor = file_hash

        return files, next_cursor, None
    except Exception as e:
        logger.error(f"Error getting files for run {run_id}: {e}", exc_info=True)
        return [], None, str(e)

def start_run(bucket_name):
    """
    Starts a new batch processing run for the specified GCS bucket.
    This is an async-style operation that returns quickly.
    """
    run_id = f"run_{uuid.uuid4()}"
    start_time = datetime.now(timezone.utc)

    runs_ref = db.collection('batch_process_runs').document(run_id)
    runs_ref.set({
        'run_id': run_id,
        'bucket_name': bucket_name,
        'start_timestamp': start_time,
        'end_timestamp': None,
        'status': 'running',
        'total_files_scanned': 0,
        'new_files_found': 0,
        'previously_processed_files': 0,
        'error_message': None
    })

    def background_task():
        try:
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            blobs = bucket.list_blobs()

            total_files_scanned = 0
            new_files_found = 0
            previously_processed_files = 0
            
            # Fetch ignored folders once per run
            ignored_folders = get_ignored_folders()
            logger.info(f"Batch processing will ignore folders: {ignored_folders}")

            for blob in blobs:
                # Check if the blob is in an ignored folder
                is_ignored = False
                for ignored_path in ignored_folders:
                    if blob.name.startswith(ignored_path):
                        logger.info(f"Skipping blob '{blob.name}' as it is in an ignored folder: '{ignored_path}'")
                        is_ignored = True
                        break
                if is_ignored:
                    total_files_scanned += 1 # Still count it as scanned, but not processed
                    continue

                if not blob.name.lower().endswith(('.pdf', '.doc', '.csv', '.xlsx')):
                    continue

                total_files_scanned += 1
                file_gcs_path = f"gs://{bucket_name}/{blob.name}"
                
                # Create a unique hash based on path, size, and updated time
                unique_string = f"{file_gcs_path}:{blob.size}:{blob.updated}"
                file_version_hash = hashlib.sha256(unique_string.encode('utf-8')).hexdigest()
                
                processed_files_ref = db.collection('batch_processed_files').document(file_version_hash)
                doc = processed_files_ref.get()

                if doc.exists:
                    previously_processed_files += 1
                else:
                    # New file or updated version found, process it
                    try:
                        file_content = blob.download_as_bytes()
                        file_stream = io.BytesIO(file_content)
                        
                        # Use the original upload logic
                        pdf_reader = PdfReader(file_stream)
                        if pdf_reader.is_encrypted:
                            if config.DEFAULT_PDF_PASSWORD:
                                try:
                                    pdf_reader.decrypt(config.DEFAULT_PDF_PASSWORD)
                                except Exception as e:
                                    logger.warning(f"Skipping encrypted file (wrong password?): {file_gcs_path}")
                                    continue
                            else:
                                logger.warning(f"Skipping encrypted file (no default password set): {file_gcs_path}")
                                continue
                        if not (config.MIN_PDF_PAGE_COUNT <= len(pdf_reader.pages) <= config.MAX_PDF_PAGE_COUNT):
                            logger.warning(f"Skipping file with invalid page count: {file_gcs_path}")
                            continue
                        
                        file_stream.seek(0)
                        base_filename = os.path.basename(blob.name)
                        gcs_uri, blob_name_new, file_size, gcs_error = gcs_service.upload_original_to_gcs(file_stream, base_filename, blob.content_type, "batch_process")
                        if gcs_error:
                            logger.error(f"GCS Upload Failed for {file_gcs_path}: {gcs_error}")
                            continue

                        # # Extract category from immediate parent folder name (night batch only)
                        # # Example: "Finance Reports/2024/document.pdf" -> category = "FINANCE_REPORTS"
                        # blob_dir = os.path.dirname(blob.name)
                        # categories = []
                        # if blob_dir:
                        #     # Get only the immediate parent folder (last directory in path)
                        #     folder_name = os.path.basename(blob_dir)
                        #     if folder_name:
                        #         # Convert to uppercase and replace spaces with underscores
                        #         category = folder_name.upper().replace(' ', '_')
                        #         categories = [category]
                        #         logger.info(f"Night batch: Extracted category '{category}' from folder '{folder_name}' for file {base_filename}")

                        # Store the original GCS path where file was picked from (for scheduled batch process)
                        original_gcs_path = f"gs://{bucket_name}/{blob.name}"
                        # doc_id, metadata_error = create_doc_metadata("batch_process", base_filename, gcs_uri, blob_name_new, blob.content_type, file_size, "Pending", 0, 0, categories=categories, destination_path=original_gcs_path)
                        doc_id, metadata_error = create_doc_metadata("batch_process", base_filename, gcs_uri, blob_name_new, blob.content_type, file_size, "Pending", 0, 0, destination_path=original_gcs_path)
                        if metadata_error:
                            logger.error(f"Metadata Creation Failed for {file_gcs_path}: {metadata_error}")
                            continue

                        redis_conn = get_redis_client()
                        if redis_conn:
                            task_string = f"split_and_process_pdf::{doc_id}::{gcs_uri}"
                            redis_conn.rpush(QUEUE_NAME, task_string)
                        
                        Thread(target=update_search_keywords_by_doc_id, args=(doc_id,)).start()
                        
                        new_files_found += 1
                        processed_files_ref.set({
                            'file_gcs_path': file_gcs_path,
                            'file_size': blob.size,
                            'file_last_updated': blob.updated,
                            'parent_run_id': run_id,
                            'status': 'processed',
                            'document_metadata_id': doc_id,
                            'first_seen_timestamp': firestore.SERVER_TIMESTAMP,
                            'last_checked_timestamp': firestore.SERVER_TIMESTAMP
                        })
                    except Exception as e:
                        logger.error(f"Error processing file {file_gcs_path} from batch: {e}")

            runs_ref.update({
                'end_timestamp': firestore.SERVER_TIMESTAMP,
                'status': 'completed',
                'total_files_scanned': total_files_scanned,
                'new_files_found': new_files_found,
                'previously_processed_files': previously_processed_files
            })

        except Exception as e:
            runs_ref.update({
                'end_timestamp': firestore.SERVER_TIMESTAMP,
                'status': 'failed',
                'error_message': str(e)
            })
            logger.error(f"Error during batch run {run_id}: {e}", exc_info=True)

    thread = Thread(target=background_task)
    thread.start()

    return run_id
