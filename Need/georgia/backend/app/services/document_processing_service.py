# backend/app/services/document_processing_service.py
import os
import io
import logging
from typing import List, Optional, Tuple
from PyPDF2 import PdfReader, PdfWriter, errors as PyPDF2Errors
from google.cloud import firestore, storage, documentai # Add documentai import
from google.api_core.client_options import ClientOptions # Add client_options import
from google.api_core import exceptions as api_core_exceptions # For specific exception handling
from tenacity import retry, stop_after_attempt, wait_exponential # Import tenacity
import uuid
import datetime
import statistics # For calculating average - Keep for now if needed elsewhere, remove if not
from threading import Thread

# Import other necessary services/modules
from .doc_processing_helpers import pdf_utils
from .doc_processing_helpers import firestore_ops
from .doc_processing_helpers import ocr_utils
from .doc_processing_helpers import classification_utils # Import new classification_utils
from . import vertex_ai_service
from . import gcs_service
from app import config
from app import db # Import the initialized db instance from __init__
from app.utils.redis_client import get_redis_client, QUEUE_NAME # Import Redis client utility and QUEUE_NAME
from app.models.processor_routing_rule_model import ProcessorRoutingRuleModel # Import the new model
from app.models.metadata_model import create_doc_metadata
from app.utils.utils import update_search_keywords_by_doc_id
from app.services.metadata_extraction_service import MetadataExtractionService


# Firestore collection references
chunks_ref = db.collection('document_chunks')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') # Added format
logger = logging.getLogger(__name__)

# --- Client Initialization (using config) ---
# Initialize Document AI client
docai_client = None
DOCAI_PROCESSOR_NAME = None
if config.PROJECT_ID and config.DOCAI_LOCATION and config.DOCUMENT_API_PROCESSOR_ID:
    try:
        # You must set the api_endpoint if you use a location other than 'us'.
        opts = ClientOptions(api_endpoint=f"{config.DOCAI_LOCATION}-documentai.googleapis.com") if config.DOCAI_LOCATION != "us" else None
        docai_client = documentai.DocumentProcessorServiceClient(client_options=opts)
        DOCAI_PROCESSOR_NAME = docai_client.processor_path(
            config.PROJECT_ID, config.DOCAI_LOCATION, config.DOCUMENT_API_PROCESSOR_ID
        )
        logger.info(f"Document AI client initialized. Processor Name: {DOCAI_PROCESSOR_NAME}")
    except Exception as e:
        logger.error(f"Failed to initialize Document AI client: {e}", exc_info=True)
        # Depending on requirements, might want to raise an error or handle gracefully
else:
    logger.warning("Document AI configuration missing (PROJECT_ID, LOCATION, or PROCESSOR_ID). Client not initialized.")

storage_client = storage.Client(project=config.PROJECT_ID)
redis_client = get_redis_client() # Initialize Redis client

# --- Constants ---
CHUNK_SIZE = 2 # Pages per chunk
CHUNK_GCS_PREFIX = "chunks/" # Prefix for storing chunks in GCS
# QUEUE_NAME is imported from redis_client

# --- Helper Functions ---

def _generate_chunk_id() -> str:
    """Generates a unique ID for a document chunk."""
    return str(uuid.uuid4())

# download_blob_to_bytes moved to gcs_service.py

# Remove _calculate_average_confidence as it's specific to Document AI
# def _calculate_average_confidence(document: documentai.Document) -> Optional[float]: ...

# --- Core Processing Step Functions ---

# split_pdf function moved to doc_processing_helpers/pdf_utils.py
# create_initial_chunk_firestore_entry moved to doc_processing_helpers/firestore_ops.py

@retry(stop=stop_after_attempt(config.GCS_UPLOAD_MAX_RETRY), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
def upload_chunk_to_gcs_and_create_initial_entry(
    chunk_content: bytes, 
    parent_doc_id: str, 
    original_parent_filename: str,
    chunk_number: int,
    start_page: int, 
    end_page: int
) -> Tuple[Optional[str], Optional[str], Optional[str]]: # Returns (chunk_id, chunk_gcs_uri, error_message)
    """
    Uploads a single PDF chunk to GCS and creates its initial Firestore entry.
    This combines part of Step 4 (saving individual chunk PDF) and Step 5.
    """
    temp_chunk_id_for_gcs = _generate_chunk_id() # Temporary ID for GCS path, actual ID from Firestore
    logger.info(f"Processing chunk {chunk_number} for parent doc {parent_doc_id} (temp GCS ID: {temp_chunk_id_for_gcs})")
    
    chunk_gcs_uri = None
    firestore_chunk_id = None
    
    try:
        # 1. Upload chunk PDF to GCS
        bucket = storage_client.bucket(config.BUCKET_NAME)
        # Use parent_doc_id in the GCS path for better organization
        blob_name = f"{CHUNK_GCS_PREFIX}{parent_doc_id}/chunk_{chunk_number}_{temp_chunk_id_for_gcs}.pdf"
        blob = bucket.blob(blob_name)
        blob.upload_from_string(chunk_content, content_type="application/pdf")
        chunk_gcs_uri = f"gs://{config.BUCKET_NAME}/{blob_name}"
        logger.info(f"Successfully uploaded chunk {chunk_number} (temp GCS ID: {temp_chunk_id_for_gcs}) for parent {parent_doc_id} to {chunk_gcs_uri}")

        # 2. Create initial Firestore entry for this chunk
        firestore_chunk_id, firestore_error = firestore_ops.create_initial_chunk_firestore_entry( # Updated call
            parent_doc_id=parent_doc_id,
            original_parent_filename=original_parent_filename,
            chunk_number=chunk_number,
            chunk_gcs_uri=chunk_gcs_uri,
            start_page=start_page,
            end_page=end_page
        )

        if firestore_error:
            # If Firestore entry fails, we might have an orphaned GCS file.
            # This could be handled by a cleanup process or by attempting to delete the GCS file here.
            logger.error(f"Failed to create Firestore entry for chunk {chunk_number} (GCS: {chunk_gcs_uri}): {firestore_error}. GCS file might be orphaned.")
            # For now, just return the error.
            return None, chunk_gcs_uri, f"Firestore entry creation failed: {firestore_error}"

        logger.info(f"Successfully created Firestore entry {firestore_chunk_id} for chunk {chunk_number} (GCS: {chunk_gcs_uri})")
        return firestore_chunk_id, chunk_gcs_uri, None

    except (api_core_exceptions.GoogleAPICallError, api_core_exceptions.RetryError, TimeoutError) as e_gcs:
        logger.warning(f"Retrying GCS upload for chunk {chunk_number} (parent {parent_doc_id}) due to: {e_gcs}")
        raise # Re-raise to trigger tenacity retry for GCS operation
    except Exception as e:
        logger.error(f"Unexpected error uploading chunk {chunk_number} (parent {parent_doc_id}) or creating its initial Firestore entry: {e}", exc_info=True)
        # Return None for chunk_id if a non-GCS error occurs after GCS upload might have succeeded.
        return None, chunk_gcs_uri, str(e)

# --- Firestore Update Functions ---
# Moved to doc_processing_helpers/firestore_ops.py:
# - update_document_status
# - store_chunk_metadata
# - update_chunk_status
# - get_all_chunk_statuses
# - update_root_doc_completion

# --- Classification and Processor Selection ---
# get_parser_for_chunk_via_gemini moved to doc_processing_helpers/classification_utils.py
