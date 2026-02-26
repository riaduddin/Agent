# backend/app/services/doc_processing_helpers/ocr_utils.py
import logging
from typing import Optional, Tuple
import statistics

from google.cloud import documentai
from google.api_core.client_options import ClientOptions
from google.api_core import exceptions as api_core_exceptions

from app import config

logger = logging.getLogger(__name__)

# Initialize Document AI client specifically for this module if it's self-contained
# Or, assume it's passed or accessed from a shared context if dps becomes very thin.
# For now, re-initializing here for clarity, similar to how it was in dps.
docai_client = None
DOCAI_PROCESSOR_NAME = None # This is the default processor
if config.PROJECT_ID and config.DOCAI_LOCATION and config.DOCUMENT_API_PROCESSOR_ID:
    try:
        opts = ClientOptions(api_endpoint=f"{config.DOCAI_LOCATION}-documentai.googleapis.com") if config.DOCAI_LOCATION != "us" else None
        docai_client = documentai.DocumentProcessorServiceClient(client_options=opts)
        DOCAI_PROCESSOR_NAME = docai_client.processor_path(
            config.PROJECT_ID, config.DOCAI_LOCATION, config.DOCUMENT_API_PROCESSOR_ID
        )
        logger.info(f"OCR Utils: Document AI client initialized. Default Processor Name: {DOCAI_PROCESSOR_NAME}")
    except Exception as e:
        logger.error(f"OCR Utils: Failed to initialize Document AI client: {e}", exc_info=True)
else:
    logger.warning("OCR Utils: Document AI configuration missing. Client not initialized.")


def perform_ocr(pdf_chunk_content: bytes, mime_type: str = "application/pdf", processor_name_override: Optional[str] = None) -> Tuple[str, Optional[float], Optional[dict]]:
    """
    Performs OCR on a PDF chunk using a specified or default Google Document AI processor.

    Args:
        pdf_chunk_content: Bytes of the PDF chunk.
        mime_type: Mime type of the content (should be 'application/pdf').
        processor_name_override (Optional[str]): Specific processor name (short ID or full path) to use.
                                                 If None, uses the default DOCAI_PROCESSOR_NAME.

    Returns:
        A tuple containing:
            - The extracted text as a string.
            - The average token layout confidence score as a float (or None if unavailable).
            - Extracted entities as a dict (or None if not applicable/available).
    """
    final_processor_path = None

    if not docai_client:
        err_msg = "OCR Utils: Document AI client not initialized. Cannot perform OCR."
        logger.error(err_msg)
        raise ValueError(err_msg)

    if processor_name_override:
        if not processor_name_override.startswith("projects/"): 
            if config.PROJECT_ID and config.DOCAI_LOCATION:
                try:
                    final_processor_path = docai_client.processor_path(
                        config.PROJECT_ID,
                        config.DOCAI_LOCATION,
                        processor_name_override 
                    )
                    logger.info(f"OCR Utils: Constructed full processor path for override ID '{processor_name_override}': {final_processor_path}")
                except Exception as e_path:
                    err_msg = f"OCR Utils: Failed to construct processor path for override short ID '{processor_name_override}': {e_path}"
                    logger.error(err_msg)
                    raise ValueError(err_msg)
            else:
                err_msg = f"OCR Utils: PROJECT_ID or DOCAI_LOCATION missing, cannot construct full path for override ID '{processor_name_override}'."
                logger.error(err_msg)
                raise ValueError(err_msg)
        else: 
             final_processor_path = processor_name_override
             logger.info(f"OCR Utils: Using provided full path override: {final_processor_path}")
    else:
        final_processor_path = DOCAI_PROCESSOR_NAME
        if not final_processor_path:
             err_msg = "OCR Utils: Default DOCAI_PROCESSOR_NAME not initialized. Cannot perform OCR."
             logger.error(err_msg)
             raise ValueError(err_msg)
        logger.info(f"OCR Utils: Using default processor path: {final_processor_path}")

    if not final_processor_path:
        err_msg = "OCR Utils: Critical error: final_processor_path could not be determined."
        logger.error(err_msg)
        raise ValueError(err_msg)

    try:
        logger.info(f"OCR Utils: Performing Document AI processing using final processor path: {final_processor_path}")
        raw_document = documentai.RawDocument(content=pdf_chunk_content, mime_type=mime_type)
        request = documentai.ProcessRequest(
            name=final_processor_path,
            raw_document=raw_document,
            skip_human_review=True
        )
        result = docai_client.process_document(request=request)
        
        document = result.document
        extracted_text = document.text if document.text else ""
        extracted_entities = None

        if document.entities:
            extracted_entities = {}
            for entity in document.entities:
                entity_type = entity.type_
                mention_text = entity.mention_text
                confidence = entity.confidence
                if entity_type not in extracted_entities:
                    extracted_entities[entity_type] = []
                extracted_entities[entity_type].append({
                    "text": mention_text,
                    "confidence": round(confidence, 4) if confidence else None
                })
            logger.info(f"OCR Utils: Extracted {len(document.entities)} entities using processor {final_processor_path}.")

        token_confidences = []
        if document.pages:
            for page in document.pages:
                if page.tokens:
                    for token in page.tokens:
                        if hasattr(token, 'layout') and token.layout and hasattr(token.layout, 'confidence') and isinstance(token.layout.confidence, (float, int)):
                             token_confidences.append(token.layout.confidence)

        avg_confidence = statistics.mean(token_confidences) if token_confidences else None

        log_msg = f"OCR Utils: Document AI processing successful with {final_processor_path}. Extracted {len(extracted_text)} characters."
        if avg_confidence is not None:
            log_msg += f" Average token layout confidence: {avg_confidence:.4f}."
        else:
             log_msg += " (Layout confidence score unavailable)."
        if extracted_entities:
            log_msg += f" Found {len(extracted_entities)} entity types."
        logger.info(log_msg)

        return extracted_text, avg_confidence, extracted_entities

    except api_core_exceptions.GoogleAPICallError as e:
        logger.error(f"OCR Utils: Document AI API call failed with processor {final_processor_path}: {e}", exc_info=True)
        raise 
    except Exception as e:
        logger.error(f"OCR Utils: Unexpected error during Document AI processing with {final_processor_path}: {e}", exc_info=True)
        raise
