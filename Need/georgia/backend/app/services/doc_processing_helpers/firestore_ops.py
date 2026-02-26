# backend/app/services/doc_processing_helpers/firestore_ops.py
import logging
import datetime
from typing import List, Optional, Tuple

from google.api_core import exceptions as api_core_exceptions
from tenacity import retry, stop_after_attempt, wait_exponential

from app import db, config

logger = logging.getLogger(__name__)

# Firestore collection references
chunks_ref = db.collection('document_chunks')
docs_ref = db.collection('document_metadata') # For parent documents

@retry(stop=stop_after_attempt(config.FIRESTORE_SAVE_MAX_RETRY), wait=wait_exponential(multiplier=1, min=1, max=5), reraise=True)
def create_initial_chunk_firestore_entry(
    parent_doc_id: str,
    original_parent_filename: str,
    chunk_number: int,
    chunk_gcs_uri: str,
    start_page: int,
    end_page: int
) -> Tuple[Optional[str], Optional[str]]:
    """
    Creates an initial Firestore document for a new chunk.
    Returns (chunk_id, error_message).
    """
    logger.info(f"Creating initial Firestore entry for chunk {chunk_number} of parent {parent_doc_id}")
    try:
        chunk_doc_ref = chunks_ref.document() # Let Firestore generate chunk_id
        chunk_id = chunk_doc_ref.id
        
        chunk_data = {
            "chunk_id": chunk_id,
            "original_doc_firestore_id": parent_doc_id, # Changed key name
            "original_parent_filename": original_parent_filename,
            "chunk_number": chunk_number,
            "gcs_path_chunk": chunk_gcs_uri,
            "start_page": start_page,
            "end_page": end_page,
            "status": "pending_classification",
            "ocr_text_preview": None,
            "ocr_confidence_score": None,
            "selected_parser_processor_id": None,
            "classified_document_type": None,
            "embedding_status": "pending",
            "has_embedding": False,
            "processing_log": [
                {
                    "timestamp": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
                    "event": "chunk_metadata_created",
                    "status": "pending_classification"
                }
            ],
            "created_at": datetime.datetime.now(tz=datetime.timezone.utc),
            "last_updated": datetime.datetime.now(tz=datetime.timezone.utc)
        }
        chunk_doc_ref.set(chunk_data)
        logger.info(f"Successfully created initial Firestore entry for chunk {chunk_id} (parent: {parent_doc_id})")
        return chunk_id, None
    except Exception as e:
        logger.error(f"Error creating initial Firestore entry for chunk (parent {parent_doc_id}): {e}", exc_info=True)
        return None, str(e)

@retry(stop=stop_after_attempt(config.FIRESTORE_UPDATE_MAX_RETRY), wait=wait_exponential(multiplier=1, min=1, max=5), reraise=True)
def update_document_status(doc_firestore_id: str, status: str, status_message: Optional[str] = None):
    """Updates the status of the main document metadata in Firestore with retry logic."""
    logger.info(f"Attempting to update status for document {doc_firestore_id} to '{status}'")
    try:
        doc_ref = docs_ref.document(doc_firestore_id)
        update_data = {
            "status": status,
            "last_status_update": datetime.datetime.now(datetime.timezone.utc)
        }
        if status_message:
            update_data["status_message"] = status_message
        doc_ref.update(update_data)
        logger.info(f"Successfully updated status for document {doc_firestore_id} to '{status}'.")
    except (api_core_exceptions.GoogleAPICallError, api_core_exceptions.RetryError, TimeoutError) as e:
        logger.warning(f"Retrying Firestore status update for document {doc_firestore_id} due to potentially transient error: {e}")
        raise 
    except Exception as e:
        logger.error(f"Unexpected error updating Firestore status for document {doc_firestore_id}: {e}", exc_info=True)
        raise

@retry(stop=stop_after_attempt(config.FIRESTORE_SAVE_MAX_RETRY), wait=wait_exponential(multiplier=1, min=1, max=5), reraise=True)
def store_chunk_metadata(
    chunk_id: str,
    original_doc_firestore_id: str,
    original_filename: str,
    chunk_gcs_uri: str,
    start_page: int,
    end_page: int,
    extracted_text: Optional[str],
    embedding: Optional[list],
    ocr_confidence_score: Optional[float], 
    extracted_entities: Optional[dict] = None,
    classified_chunk_document_type_label: Optional[str] = None,
    used_parser_processor_id: Optional[str] = None,
    status: str = "pending_vectorization"
) -> None:
    """Stores metadata for a processed chunk in Firestore with retry logic."""
    logger.info(f"Attempting to store metadata for chunk {chunk_id}")
    metadata = {
        "chunk_id": chunk_id,
        "original_doc_firestore_id": original_doc_firestore_id,
        "original_filename": original_filename,
        "chunk_gcs_uri": chunk_gcs_uri,
        "start_page": start_page,
        "end_page": end_page,
        "ocr_text_preview": extracted_text,
        "has_text": bool(extracted_text and extracted_text.strip()),
        "embedding_model": config.EMBEDDING_MODEL_NAME if embedding else None,
        "has_embedding": bool(embedding),
        "ocr_confidence_score": ocr_confidence_score,
        "extracted_entities": extracted_entities, 
        "entities": [], # NEW: Searchable array for hard-match
        "classified_chunk_document_type_label": classified_chunk_document_type_label,
        "used_parser_processor_id": used_parser_processor_id,
        "status": status,
        "created_at": datetime.datetime.now(datetime.timezone.utc),
        "last_updated": datetime.datetime.now(datetime.timezone.utc),
    }

    # Populate entities array if we have extracted_entities
    if extracted_entities:
        searchable_entities = set()
        def collect_values(data):
            if isinstance(data, dict):
                for k, v in data.items():
                    # Skip strictly internal tracking fields that aren't useful for search
                    if k in ['_chunk_id', '_category_confidence', '_extraction_version', '_original_doc_type', '_extraction_method', 'extraction_method', '_initial_classification', '_detected_category', '_doc_type']:
                        continue
                    collect_values(v)
            elif isinstance(data, list):
                for item in data:
                    collect_values(item)
            elif data is not None and not isinstance(data, bool):
                val_str = str(data).strip()
                if val_str:
                    searchable_entities.add(val_str)
        
        collect_values(extracted_entities)
        metadata["entities"] = list(searchable_entities)

    try:
        doc_ref = chunks_ref.document(chunk_id)
        doc_ref.set(metadata)
        logger.info(f"Successfully stored metadata for chunk {chunk_id} in Firestore.")
    except (api_core_exceptions.GoogleAPICallError, api_core_exceptions.RetryError, TimeoutError) as e:
        logger.warning(f"Retrying Firestore save for chunk {chunk_id} due to potentially transient error: {e}")
        raise 
    except Exception as e:
        logger.error(f"Unexpected error storing Firestore metadata for chunk {chunk_id}: {e}", exc_info=True)
        raise

@retry(stop=stop_after_attempt(config.FIRESTORE_UPDATE_MAX_RETRY), wait=wait_exponential(multiplier=1, min=1, max=5), reraise=True)
def update_chunk_status(
    chunk_id: str, 
    status: str, 
    status_message: Optional[str] = None, 
    has_embedding: Optional[bool] = None,
    embedding_status: Optional[str] = None  # New parameter
):
    """Updates the status, optionally has_embedding, and optionally embedding_status of a chunk metadata document."""
    logger.info(f"Attempting to update status for chunk {chunk_id} to '{status}'")
    try:
        doc_ref = chunks_ref.document(chunk_id)
        update_data = {
            "status": status,
            "last_updated": datetime.datetime.now(datetime.timezone.utc)
        }
        if status_message:
            update_data["status_message"] = status_message
        if has_embedding is not None:
            update_data["has_embedding"] = has_embedding
            logger.info(f"Updating has_embedding for chunk {chunk_id} to {has_embedding}.")
        if embedding_status is not None: # Add new field to update_data
            update_data["embedding_status"] = embedding_status
            logger.info(f"Updating embedding_status for chunk {chunk_id} to {embedding_status}.")

        doc_ref.update(update_data)
        log_msg = f"Updated status for chunk {chunk_id} to '{status}'."
        if has_embedding is not None:
            log_msg += f" Set has_embedding to {has_embedding}."
        if embedding_status is not None:
            log_msg += f" Set embedding_status to {embedding_status}."
        logger.info(log_msg)
    except (api_core_exceptions.GoogleAPICallError, api_core_exceptions.RetryError, TimeoutError) as e:
        logger.warning(f"Retrying Firestore status update for chunk {chunk_id} due to potentially transient error: {e}")
        raise 
    except Exception as e:
        logger.error(f"Unexpected error updating Firestore status for chunk {chunk_id}: {e}", exc_info=True)
        raise

def get_all_chunk_statuses(original_doc_firestore_id: str) -> List[str]:
    """Retrieves the status for all chunks of a given document."""
    statuses = []
    try:
        chunks_query = chunks_ref.where("original_doc_firestore_id", "==", original_doc_firestore_id).stream()
        for chunk in chunks_query:
            chunk_data = chunk.to_dict()
            statuses.append(chunk_data.get("status"))
        return statuses
    except Exception as e:
        logger.error(f"Failed to retrieve chunk statuses for doc {original_doc_firestore_id}: {e}", exc_info=True)
        return []

@retry(stop=stop_after_attempt(config.FIRESTORE_UPDATE_MAX_RETRY), wait=wait_exponential(multiplier=1, min=1, max=5), reraise=True)
def update_root_doc_completion(doc_firestore_id: str, final_status: str = "completed"):
    """Updates the root document status (completed or error) with retry logic."""
    logger.info(f"Attempting to mark document {doc_firestore_id} as {final_status}")
    try:
        doc_ref = docs_ref.document(doc_firestore_id)
        update_data = {
            "status": final_status,
            "last_status_update": datetime.datetime.now(datetime.timezone.utc),
            "processing_end_time": datetime.datetime.now(datetime.timezone.utc)
        }
        doc_ref.update(update_data)
        logger.info(f"Successfully marked document {doc_firestore_id} as {final_status}.")
    except (api_core_exceptions.GoogleAPICallError, api_core_exceptions.RetryError, TimeoutError) as e:
        logger.warning(f"Retrying root document completion update for {doc_firestore_id} due to potentially transient error: {e}")
        raise 
    except Exception as e:
        logger.error(f"Unexpected error updating root document {doc_firestore_id} completion status: {e}", exc_info=True)
        raise
