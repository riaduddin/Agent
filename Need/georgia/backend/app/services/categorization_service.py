# backend/app/services/categorization_service.py
import logging
import json
from typing import List
from app import db
from app.services import vertex_ai_service
from app.services import category_api_service
from app.utils.debug_logger import debug_log

logger = logging.getLogger(__name__)

def _extract_filename_from_gcs_path(gcs_path: str) -> str:
    """
    Extracts just the filename from a GCS path.
    Handles both full GCS paths (gs://bucket/path/to/file.pdf) and regular paths.
    Returns the filename portion only.
    """
    if not gcs_path:
        return gcs_path
    
    if gcs_path.startswith('gs://'):
        # Remove 'gs://' prefix and get the last part after '/'
        path_without_prefix = gcs_path[5:]  # Remove 'gs://'
        return path_without_prefix.split('/')[-1]
    
    # If it's already a filename or relative path, return the last part
    return gcs_path.split('/')[-1]

def _extract_categories_from_destination_path(destination_path: str, available_categories: List[str]) -> List[str]:
    """
    Extracts categories from destination_path by removing the GCS bucket part and matching path elements.
    
    Example:
        destination_path: "gs://georgia-doc-storage-2/Georgia 14/INVOICES/inv_demo_22.pdf"
        After removing bucket: "Georgia 14/INVOICES/inv_demo_22.pdf"
        After split: ["Georgia 14", "INVOICES", "inv_demo_22.pdf"]
        After removing filename: ["Georgia 14", "INVOICES"]
        Match against available_categories and return matches
    
    Args:
        destination_path: Full GCS path
        available_categories: List of valid category codes (e.g., ["INVOICES", "CHECKS", ...])
    
    Returns:
        List of matching categories found in the path
    """
    if not destination_path or not available_categories:
        return []
    
    try:
        # Remove the bucket part (gs://bucket-name/)
        if destination_path.startswith('gs://'):
            # Remove 'gs://' prefix
            path_without_prefix = destination_path[5:]
            # Find the first '/' and remove everything before and including it (bucket name)
            first_slash = path_without_prefix.find('/')
            if first_slash != -1:
                path_without_bucket = path_without_prefix[first_slash + 1:]
            else:
                # No path elements after bucket name
                return []
        else:
            path_without_bucket = destination_path
        
        # Split the path by '/'
        path_elements = path_without_bucket.split('/')
        
        # Remove the filename (last element if it has a file extension)
        if path_elements:
            last_element = path_elements[-1]
            # Check if last element is a filename (contains a dot for extension)
            if '.' in last_element:
                path_elements = path_elements[:-1]
                debug_log(f"Removed filename '{last_element}' from path elements")
        
        # Match path elements with available categories (case-insensitive)
        matched_categories = []
        available_categories_upper = [cat.upper() for cat in available_categories]
        
        for element in path_elements:
            if not element:  # Skip empty strings from consecutive slashes
                continue
            
            element_upper = element.upper().replace(' ', '_')  # Handle spaces in folder names
            
            # Check for exact match
            if element_upper in available_categories_upper:
                matched_categories.append(element_upper)
                debug_log(f"Matched path element '{element}' to category '{element_upper}'")
            # Also check without replacing spaces (in case categories have underscores)
            elif element_upper.replace('_', ' ') in available_categories_upper:
                matched_categories.append(element_upper.replace('_', ' '))
                debug_log(f"Matched path element '{element}' to category (with space handling)")
        
        # Return unique matches
        return list(set(matched_categories))
    
    except Exception as e:
        debug_log(f"Error extracting categories from destination path: {e}")
        logger.error(f"Error extracting categories from destination path '{destination_path}': {e}")
        return []

def categorize_document(doc_id: str):
    """
    Categorizes a document based on its filename or content using Gemini.
    This function will be called by the worker.
    """
    debug_log(f"Starting categorization for doc_id: {doc_id}")
    logger.info(f"Starting categorization for doc_id: {doc_id}")
    try:
        doc_ref = db.collection("document_metadata").document(doc_id)
        doc_snap = doc_ref.get()


        # Query batch_processed_files for a document where document_metadata_id == doc_id
        batch_processed_query = db.collection("batch_processed_files").where("document_metadata_id", "==", doc_id).limit(1)
        batch_processed_snapshots = batch_processed_query.get()

        batch_process_gs_path = None

        # Check if any document matched the query
        if batch_processed_snapshots:
            batch_process_data = batch_processed_snapshots[0].to_dict()
            batch_process_gs_path = batch_process_data.get("file_gcs_path")
            debug_log(f"Found batch processed file for {doc_id} with GCS path: {batch_process_gs_path}")
        else:
            debug_log(f"No batch processed file found for {doc_id}. Proceeding with original document.")


        if not doc_snap.exists:
            debug_log(f"Document {doc_id} not found in Firestore.")
            logger.error(f"Categorization failed: Document {doc_id} not found in Firestore.")
            return

        debug_log(f"Document {doc_id} found. Fetching data.")
        doc_data = doc_snap.to_dict()
        debug_log(f"Document data retrieved: {doc_data}")
        
        # Check if categories are already set (e.g., from night batch folder-based categorization)
        existing_categories = doc_data.get("categories", [])
        if existing_categories and any(cat for cat in existing_categories if cat.upper() != "UNCATEGORIZED"):
            debug_log(f"Document {doc_id} already has categories: {existing_categories}. Skipping AI categorization.")
            logger.info(f"Skipping categorization for {doc_id} as it already has categories: {existing_categories}")
            return
        
        filename = doc_data.get("original_filename")
        gcs_uri = doc_data.get("gcs_uri")

        if not filename:
            debug_log(f"Document {doc_id} has no original_filename.")
            logger.error(f"Categorization failed: Document {doc_id} has no original_filename.")
            return

        # Get dynamic categories and short code map from API (with fallback to VALID_CATEGORIES)
        # API uses X-API-Key header with JWT_SECRET_KEY from config
        valid_categories, short_code_map, full_categories = category_api_service.get_categories_with_fallback()
        debug_log(f"Using {len(valid_categories)} categories from API/fallback")
        if short_code_map:
            debug_log(f"Short code map contains {len(short_code_map)} mappings")
        
        # NEW: Check short codes in filename first (before Gemini analysis)
        # This uses the full categories list to find category name by short code
        # Extract just the filename from GCS path if it's a full path
        file_path_to_check = filename
        if batch_process_gs_path:
            file_path_to_check = _extract_filename_from_gcs_path(batch_process_gs_path)
        
        category_name_from_short_code = category_api_service.get_categories_from_filename(
            file_path_to_check, 
            full_categories
        )
        
        if category_name_from_short_code and category_name_from_short_code != "UNCATEGORIZED":
            debug_log(f"Category determined from short code: '{category_name_from_short_code}'")
            logger.info(f"Categorized {doc_id} as '{category_name_from_short_code}' based on short code in filename.")
            doc_ref.update({"categories": category_name_from_short_code})
            debug_log(f"Finished categorization for doc_id: {doc_id} (SUCCESS via short code)")
            return

        # Attempt 1: Categorize by filename (using dynamic categories)
        debug_log(f"Attempt 1 - Categorizing by filename: '{filename}'")

        file_path_to_use = filename
        # Extract just the filename from GCS path if batch_process_gs_path is available
        if batch_process_gs_path:
            file_path_to_use = _extract_filename_from_gcs_path(batch_process_gs_path)
            debug_log(f"Using filename extracted from batch processed GCS path: {file_path_to_use}")
        
        categories_list_str = ", ".join(valid_categories)
        prompt_filename = f"Analyze the following filename and extract its primary category. The filename is: '{file_path_to_use}'. Based on the name, classify it into one of the following categories: [{categories_list_str}]. Return only the category name as a single string. If you cannot determine a category from the filename, return 'None'."
        debug_log(f"Filename prompt: {prompt_filename}")
        
        debug_log("Sending filename prompt to Gemini...")
        category_response = vertex_ai_service.generate_plain_chat_response(prompt_filename)

        debug_log("Gemini response received for filename categorization.")
        # Extract category from Gemini response
        debug_log(f"Processing Gemini response for filename: {category_response}")
        
        category = ""
        if hasattr(category_response, 'parts'):
             category = ''.join(part.text for part in category_response.parts if hasattr(part, 'text')).strip()
        else: # Fallback for different response structures
             category = str(category_response).strip()
        
        debug_log(f"Gemini response for filename: '{category}'")

        debug_log(f"Checking if response '{category}' is valid and not 'None'.")
        if category and category != 'None' and category in valid_categories:
            debug_log(f"Category is valid. Updating Firestore with categories: ['{category}']")
            logger.info(f"Categorized {doc_id} as '{category}' based on filename.")
            doc_ref.update({"categories": [category]})
            debug_log(f"Finished categorization for doc_id: {doc_id} (SUCCESS via filename)")
            return
        else:
            debug_log(f"Category '{category}' is not valid or is 'None'. Proceeding to content analysis.")

        # Attempt 2: Categorize by full content (if filename fails)
        debug_log(f"Attempt 2 - Filename-based categorization failed for {doc_id}. Trying full content analysis.")
        logger.info(f"Filename-based categorization failed for {doc_id}. Trying full content analysis.")
        if not gcs_uri:
            debug_log(f"GCS URI is missing for doc_id: {doc_id}. Cannot categorize by content.")
            logger.error(f"Cannot categorize by content: GCS URI is missing for doc_id: {doc_id}.")
            doc_ref.update({"categories": ["Uncategorized"]})
            return

        categories_list_str = ", ".join(valid_categories)
        prompt_content = f"Analyze the document at the following GCS path and return a JSON array of one or more relevant categories from the following list: [{categories_list_str}]. GCS Path: {gcs_uri}"
        debug_log(f"Content prompt: {prompt_content}")
        
        debug_log(f"Sending content prompt to Gemini for GCS URI: {gcs_uri}")
        content_category_response = vertex_ai_service.generate_chat_response(prompt_content, [], [], stream=False)
        
        content_category_str = ""
        if hasattr(content_category_response, 'parts'):
             content_category_str = ''.join(part.text for part in content_category_response.parts if hasattr(part, 'text')).strip()
        else:
             content_category_str = str(content_category_response).strip()
        
        debug_log(f"Gemini response for content: '{content_category_str}'")

        try:
            debug_log("Attempting to parse Gemini response as JSON.")
            categories = json.loads(content_category_str)
            debug_log(f"Successfully parsed JSON: {categories}")
            if isinstance(categories, list) and len(categories) > 0:
                # Validate all categories are in valid_categories list
                valid_cats = [cat for cat in categories if cat in valid_categories]
                if valid_cats:
                    debug_log(f"Parsed content is a valid, non-empty list. Updating Firestore with {len(valid_cats)} valid categories.")
                    logger.info(f"Categorized {doc_id} as {valid_cats} based on content.")
                    doc_ref.update({"categories": valid_cats})
                else:
                    debug_log("No valid categories found in response. Setting as 'Uncategorized'.")
                    logger.warning(f"Gemini returned categories {categories} but none are in valid categories list. Setting as 'Uncategorized'.")
                    doc_ref.update({"categories": ["Uncategorized"]})
            else:
                debug_log("Parsed content is not a list or is empty. Raising ValueError.")
                raise ValueError("Parsed JSON is not a list or is empty.")
        except (json.JSONDecodeError, ValueError) as e:
            debug_log(f"Failed to parse categories from content analysis for {doc_id}. Error: {e}. Setting as 'Uncategorized'.")
            logger.error(f"Failed to parse categories from content analysis for {doc_id}. Response: '{content_category_str}'. Error: {e}")
            doc_ref.update({"categories": ["Uncategorized"]})

    except Exception as e:
        debug_log(f"An unexpected error occurred during categorization for doc_id {doc_id}: {e}")
        logger.error(f"An unexpected error occurred during categorization for doc_id {doc_id}: {e}", exc_info=True)
        # Failsafe update
        db.collection("document_metadata").document(doc_id).set({"categories": ["Uncategorized"]}, merge=True)

def categorize_document_strict(doc_id: str, full_categories: list = None) -> bool:
    """
    Categorizes a document based ONLY on its filename (short codes or category names).
    Does NOT perform full content analysis or download the file.
    Returns True if processed (even if Uncategorized).
    """
    # print(f"\n--- DEBUG: Starting strict name-only categorization for doc_id: {doc_id} ---")
    try:
        doc_ref = db.collection("document_metadata").document(doc_id)
        doc_snap = doc_ref.get()

        # Log the snap data


        # if the doc id Hjyjl99lDimSOfDVdgKP or CEIB85eQycBkv1Pm6iCq then print this doc snap
        if doc_id == "Hjyjl99lDimSOfDVdgKP" or doc_id == "CEIB85eQycBkv1Pm6iCq":
            print("=================================================")
            debug_log(f"Document snapshot data: {doc_snap.to_dict()}")
            print("=================================================")
        #  else print debug_log(f"Skipping Log ==================")
        else:
            debug_log(f"Skipping Log ==================")


        # Query batch_processed_files for a document where document_metadata_id == doc_id
        batch_processed_query = db.collection("batch_processed_files").where("document_metadata_id", "==", doc_id).limit(1)
        batch_processed_snapshots = batch_processed_query.get()

        batch_process_gs_path = None
        if batch_processed_snapshots:
            batch_process_data = batch_processed_snapshots[0].to_dict()
            batch_process_gs_path = batch_process_data.get("file_gcs_path")

        if not doc_snap.exists:
            # print(f"DEBUG: Document {doc_id} not found in Firestore.")
            return False

        doc_data = doc_snap.to_dict()
        filename = doc_data.get("original_filename")

        if not filename:
            # If we have batch_process_gs_path, we might be able to extract a filename from it?
            if batch_process_gs_path:
                filename = _extract_filename_from_gcs_path(batch_process_gs_path)
            else:
                return False

        # Get dynamic categories if not provided
        if full_categories is None:
             _, _, full_categories = category_api_service.get_categories_with_fallback()
        
        # Get valid category codes for path matching
        valid_categories = [cat.get("code") for cat in full_categories if cat.get("code")]
        
        # NEW: Check destination_path first (highest priority for batch processing)
        destination_path = doc_data.get("destination_path")
        if destination_path:
            debug_log(f"Checking destination_path for categories: {destination_path}")
            categories_from_path = _extract_categories_from_destination_path(destination_path, valid_categories)
            if categories_from_path:
                debug_log(f"Found categories from destination_path: {categories_from_path}")
                logger.info(f"Categorized {doc_id} as {categories_from_path} based on destination folder path.")
                doc_ref.update({"categories": categories_from_path})
                return True
            else:
                debug_log(f"No matching categories found in destination_path")
        
        # Determine the file path/name to check
        file_path_to_check = filename
        if batch_process_gs_path:
            file_path_to_check = _extract_filename_from_gcs_path(batch_process_gs_path)
        
        # 1. Check short codes
        found_categories = category_api_service.get_categories_from_filename(
            file_path_to_check, 
            full_categories
        )
        
        # 2. Check full category names/codes in filename
        # Some filenames might contain "Invoice" or "Contract" directly
        file_path_lower = file_path_to_check.lower()
        for cat in full_categories:
            cat_name = cat.get("name", "").lower()
            cat_code = cat.get("code", "").lower()
            
            # Check if name is in filename (if name is significant length, e.g. > 2 chars)
            if len(cat_name) > 2 and cat_name in file_path_lower:
                found_categories.append(cat.get("code"))
            elif len(cat_code) > 2 and cat_code in file_path_lower:
                found_categories.append(cat.get("code"))

        # Deduplicate
        found_categories = list(set(found_categories))
        
        if found_categories:
            doc_ref.update({
                "categories": found_categories
            })
            return True
        else:
            # No category found -> UNCATEGORIZED (Capitalized as requested)
            doc_ref.update({
                "categories": ["UNCATEGORIZED"]
            })
            return True

    except Exception as e:
        logger.error(f"Error in strict name-only categorization for {doc_id}: {e}", exc_info=True)
        return False
