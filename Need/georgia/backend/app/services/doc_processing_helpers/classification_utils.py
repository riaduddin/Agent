# backend/app/services/doc_processing_helpers/classification_utils.py
import logging
from typing import Optional, Tuple

from app import config
from app.services import gemini_gcs_classification_service
from app.models.processor_routing_rule_model import ProcessorRoutingRuleModel
from app.models.log_model import add_log_entry # Added import

logger = logging.getLogger(__name__)

def get_parser_for_chunk_via_gemini(
    chunk_gcs_uri: str, 
    chunk_id: Optional[str] = None,
    parent_doc_id: Optional[str] = None, # Added parent_doc_id
    worker_id: Optional[str] = None, # Added worker_id
    original_filename: Optional[str] = None # Added original_filename
) -> Tuple[str, str, Optional[str]]:
    """
    Selects an appropriate Document AI parser processor for a given chunk by its GCS URI using Gemini.

    Args:
        chunk_gcs_uri (str): The GCS URI of the document chunk.
        chunk_id (Optional[str]): The ID of the chunk, for logging purposes.
        parent_doc_id (Optional[str]): The ID of the parent document, for logging.
        worker_id (Optional[str]): The ID of the worker, for logging.

    Returns:
        A tuple containing (classified_document_type_label, selected_parser_processor_id, raw_gemini_response_label).
        Defaults to DEFAULT_CLASSIFIED_DOCUMENT_TYPE and DEFAULT_PARSER_PROCESSOR_ID if no specific rule matches or in case of error.
        The raw_gemini_response_label is the direct output from Gemini.
    """
    log_prefix = f"Chunk {chunk_id if chunk_id else 'Unknown'} (Classification):"
    logger.info(f"{log_prefix} Starting Gemini classification for parser selection.")
    
    selected_parser_id = config.DEFAULT_PARSER_PROCESSOR_ID
    classified_type_label = config.DEFAULT_CLASSIFIED_DOCUMENT_TYPE
    raw_gemini_response_label: Optional[str] = None # Initialize

    try:
        rules = ProcessorRoutingRuleModel.get_all_rules(enabled_only=True, order_by_priority=False)

        if not rules:
            logger.info(f"{log_prefix} No enabled processor routing rules found. Using default parser.")
        # Removed: elif not chunk_text_content:
        elif not chunk_gcs_uri: # Check if GCS URI is provided
            logger.info(f"{log_prefix} No GCS URI provided for chunk. Using default parser.")
        else:
            # Extract just the labels for Gemini, keep the full rules for matching later
            parser_labels_for_gemini = []
            valid_rules_for_matching = [] # Store rules that have both label and ID

            for rule in rules:
                label = rule.get('documentTypeLabel')
                parser_id = rule.get('targetParserProcessorId')
                if label and parser_id:
                    parser_labels_for_gemini.append(label)
                    valid_rules_for_matching.append(rule) # Moved inside the if block
            
            if parser_labels_for_gemini: # If there are any valid labels to send to Gemini
                raw_gemini_response_label = gemini_gcs_classification_service.classify_document_type_with_gemini_from_gcs_uri(
                    chunk_gcs_uri=chunk_gcs_uri, 
                    available_parser_labels=parser_labels_for_gemini
                )

                # Add a new log entry for the raw Gemini response
                if parent_doc_id and chunk_id and worker_id: # Ensure IDs are available for logging
                    add_log_entry(
                        level="INFO",
                        message=f"Gemini raw classification output: {raw_gemini_response_label}",
                        document_id=parent_doc_id,
                        chunk_id=chunk_id,
                        step="gemini_raw_classification_output",
                        original_filename=original_filename,
                        details={"raw_output": raw_gemini_response_label, "gcs_uri": chunk_gcs_uri},
                        worker_id=worker_id
                    )
                else:
                    logger.info(f"{log_prefix} Received raw Gemini response: '{raw_gemini_response_label}' (parent/worker IDs not available for detailed log entry).")


                # Removed print statements for production
                # print(f"raw_gemini_response_label: {raw_gemini_response_label}", )
                # print(f"chunk_gcs_uri: {chunk_gcs_uri}", )
                # print(f"available_parser_labels: {parser_labels_for_gemini}", )

                if raw_gemini_response_label and raw_gemini_response_label != "NONE":
                    # Match the chosen label against the valid_rules_for_matching
                    matched_rule = next((r for r in valid_rules_for_matching if r.get('documentTypeLabel') == raw_gemini_response_label), None)
                    if matched_rule:
                        selected_parser_id = matched_rule.get('targetParserProcessorId', config.DEFAULT_PARSER_PROCESSOR_ID)
                        classified_type_label = raw_gemini_response_label # Use Gemini's choice as the classified type
                        logger.info(f"{log_prefix} Gemini classified GCS URI '{chunk_gcs_uri}' as '{classified_type_label}'. Selected parser: {selected_parser_id}. Raw Gemini response: '{raw_gemini_response_label}'")
                    else:
                        logger.warning(f"{log_prefix} Gemini chose '{raw_gemini_response_label}' for GCS URI '{chunk_gcs_uri}' but no matching rule found. Using default parser. Raw Gemini response: '{raw_gemini_response_label}'")
                elif raw_gemini_response_label == "NONE":
                    logger.info(f"{log_prefix} Gemini indicated no specific type for GCS URI '{chunk_gcs_uri}'. Using default parser. Raw Gemini response: '{raw_gemini_response_label}'")
                else: # This case includes if raw_gemini_response_label is None (service error)
                    logger.error(f"{log_prefix} Gemini classification for GCS URI '{chunk_gcs_uri}' failed or returned unexpected value ('{raw_gemini_response_label}'). Using default parser.")
            else:
                logger.info(f"{log_prefix} No valid parsers with type labels and IDs found in rules for Gemini. Using default processor.")
        
        return classified_type_label, selected_parser_id, raw_gemini_response_label

    except Exception as e:
        logger.error(f"{log_prefix} Critical error in get_parser_for_chunk_via_gemini: {e}", exc_info=True)
        return config.DEFAULT_CLASSIFIED_DOCUMENT_TYPE, config.DEFAULT_PARSER_PROCESSOR_ID, None
