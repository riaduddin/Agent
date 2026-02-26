from app import db
from google.cloud import firestore
import datetime
import logging

logger = logging.getLogger(__name__)

logs_ref = db.collection('processing_logs')

def add_log_entry(
    level: str,
    message: str,
    document_id: str = None,
    chunk_id: str = None,
    step: str = None,
    worker_id: str = None,
    original_filename: str = None, # Added original_filename
    details: dict = None
):
    """Adds a new log entry to the Firestore 'processing_logs' collection."""
    try:
        log_data = {
            'timestamp': datetime.datetime.now(tz=datetime.timezone.utc),
            'level': level.upper(),
            'message': message,
            'document_id': document_id,
            'chunk_id': chunk_id,
            'step': step,
            'worker_id': worker_id,
            'original_filename': original_filename, # Added original_filename
            'details': details
        }
        # Filter out None values
        log_data = {k: v for k, v in log_data.items() if v is not None}
        logs_ref.document().set(log_data)
        # Note: Removed verbose 'Log saved' message to keep terminal clean
    except Exception as e:
        logger.error(f"Failed to add log entry: {e}", exc_info=True)

def get_logs(
    limit: int = 50,
    start_after_doc_id: str = None,
    level: str = None,
    step: str = None,
    document_id: str = None,
    worker_id: str = None,
    original_filename: str = None,
    chunk_id: str = None # Added chunk_id
):
    try:
        query = logs_ref

        if level:
            query = query.where('level', '==', level.upper())
        if step:
            query = query.where('step', '==', step)
        if document_id:
            query = query.where('document_id', '==', document_id)
        if chunk_id: # Added chunk_id filter
            query = query.where('chunk_id', '==', chunk_id)
        if worker_id:
            query = query.where('worker_id', '==', worker_id)
        if original_filename:
            query = query.where('original_filename', '==', original_filename)

        query = query.order_by('timestamp', direction=firestore.Query.DESCENDING)

        if start_after_doc_id:
            start_after_doc = logs_ref.document(start_after_doc_id).get()
            if start_after_doc.exists:
                query = query.start_after(start_after_doc)
            else:
                logger.warning(f"Log pagination cursor document ID '{start_after_doc_id}' not found.")
                return [], None, f"Pagination cursor document ID '{start_after_doc_id}' not found."

        query = query.limit(limit)
        docs = list(query.stream())

        log_entries = []
        last_doc_id = None
        for doc in docs:
            log_data = doc.to_dict()
            if 'timestamp' in log_data and isinstance(log_data['timestamp'], datetime.datetime):
                log_data['timestamp'] = log_data['timestamp'].isoformat()
            log_data['id'] = doc.id
            log_entries.append(log_data)
            last_doc_id = doc.id

        return log_entries, last_doc_id, None

    except Exception as e:
        logger.error(f"Failed to retrieve logs: {e}", exc_info=True)
        return [], None, f"Failed to retrieve logs: {e}"

def get_all_logs_for_export(
    level: str = None,
    step: str = None,
    document_id: str = None,
    worker_id: str = None,
    original_filename: str = None
):
    try:
        query = logs_ref

        if level:
            query = query.where('level', '==', level.upper())
        if step:
            query = query.where('step', '==', step)
        if document_id:
            query = query.where('document_id', '==', document_id)
        if worker_id:
            query = query.where('worker_id', '==', worker_id)
        if original_filename:
            query = query.where('original_filename', '==', original_filename)

        query = query.order_by('timestamp', direction=firestore.Query.ASCENDING)

        docs = list(query.stream())

        log_entries = []
        for doc in docs:
            log_data = doc.to_dict()
            if 'timestamp' in log_data and isinstance(log_data['timestamp'], datetime.datetime):
                log_data['timestamp'] = log_data['timestamp'].isoformat()
            log_data['id'] = doc.id
            log_entries.append(log_data)

        return log_entries, None

    except Exception as e:
        logger.error(f"Failed to retrieve all logs for export: {e}", exc_info=True)
        return [], f"Failed to retrieve logs for export: {e}"
