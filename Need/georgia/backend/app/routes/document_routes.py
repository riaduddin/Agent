# backend/app/routes/document_routes.py
import os
import uuid
import json
import time
import re
import threading
import queue
import requests
from flask import Blueprint, request, jsonify, Response
from flask_jwt_extended import jwt_required, get_jwt_identity
from werkzeug.utils import secure_filename
import io
import logging
import datetime
from google.cloud import firestore
from PyPDF2 import PdfReader, errors as PyPDF2Errors
from app import config, db
from app.utils.utils import update_search_keywords_by_doc_id
from threading import Thread
from app.utils.utils import backfill_search_keywords

# Import services
from app.services import gcs_service, vertex_ai_service, document_processing_service
from app.services.doc_processing_helpers import bulk_processing_utils
from app.llm.gemini_api_key_client import GeminiConfigurationError
# Import new chat services
from app.services.chat import vector_search_service, firestore_service
from app.services.chat.response_validator import validate_response, get_validation_disclaimer
from app.services.search_router_service import SearchRouterService
from app.models.log_model import add_log_entry

# Import models and model functions
from app.models.metadata_model import (
    create_doc_metadata,
    save_chat_message,
    get_upload_history,
    get_user_chat_sessions,
    get_session_messages,
    get_chunks_for_document
)
# Add module imports to fix NameError in pagination routes
from app.models import metadata_model
from app.models import log_model
from app.models.system_log_model import SystemLogModel
from app.models.log_model import add_log_entry
from app.utils.redis_client import get_redis_client, QUEUE_NAME
from app.services.doc_processing_helpers.firestore_ops import update_document_status
from app.services.activity_log_service import log_file_activity, log_search_activity, log_navigation_activity, log_admin_activity
from app.models.activity_log_model import ActivityTypes
from app.utils.debug_logger import debug_log, debug_warn, debug_error, debug_perf

logger = logging.getLogger(__name__)
doc_bp = Blueprint('doc_bp', __name__)
ALLOWED_EXTENSIONS = {'pdf'}
MAX_FILENAME_LENGTH = 50

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_filename(filename):
    if not filename:
        return False, "Filename is required."
    if filename != filename.strip():
        return False, "Filename must not have leading or trailing whitespace."
    if len(filename) > MAX_FILENAME_LENGTH:
        return False, f"Filename is too long. Max length is {MAX_FILENAME_LENGTH} characters (including extension)."
    if filename.startswith("."):
        return False, "Filename must not start with a dot."
    if filename.endswith("."):
        return False, "Filename must not end with a dot."
    if ".." in filename:
        return False, "Filename must not contain consecutive dots."
    if "/" in filename or "\\" in filename:
        return False, "Filename must not contain path separators."
    if "\x00" in filename:
        return False, "Filename contains invalid characters."
    for ch in filename:
        if not ch.isascii() or not ch.isprintable() or ord(ch) < 32 or ord(ch) == 127:
            return False, "Filename contains invalid characters."
    forbidden_chars = set('\\/:*?"<>|')
    if any(ch in forbidden_chars for ch in filename):
        return False, "Filename contains invalid characters."

    if "  " in filename:
        return False, "Filename must not contain consecutive spaces."
    if "." not in filename:
        return False, "Filename must have a .pdf extension."
    base, ext = filename.rsplit(".", 1)
    if not base:
        return False, "Filename must not start with a dot."
    if ext.lower() != "pdf":
        return False, "Only .pdf files are allowed."
    return True, ""

@doc_bp.route('/backfill_search_keywords', methods=['POST'])
@jwt_required()
def trigger_backfill_keywords():
    try:
        backfill_search_keywords()
        return jsonify({"msg": "✅ Search keyword backfill completed successfully."}), 200
    except Exception as e:
        return jsonify({"msg": f"❌ Backfill failed: {str(e)}"}), 500

@doc_bp.route('/upload', methods=['POST'])
@jwt_required()
def upload_document():
    current_user_email = get_jwt_identity()
    files = request.files.getlist('file')
    if not files or all(f.filename == '' for f in files):
        return jsonify({"msg": "No selected files"}), 400

    results, errors, warnings = [], [], []
    for file in files:
        if file:
            original_filename = file.filename
            is_valid, validation_msg = validate_filename(original_filename)
            if not is_valid:
                errors.append({"filename": original_filename, "error": validation_msg})
                continue
            if not allowed_file(original_filename):
                errors.append({"filename": original_filename, "error": "File type not allowed"})
                continue
            filename = original_filename
            try:
                if file.content_type != 'application/pdf':
                    errors.append({"filename": original_filename, "error": "Invalid content type. Only PDF is allowed."})
                    continue
                file_content = file.read()
                file_stream = io.BytesIO(file_content)
                file_stream.seek(0)
                pdf_reader = PdfReader(file_stream)
                if pdf_reader.is_encrypted:
                    errors.append({"filename": original_filename, "error": "PDF is password protected."})
                    continue
                if not (config.MIN_PDF_PAGE_COUNT <= len(pdf_reader.pages) <= config.MAX_PDF_PAGE_COUNT):
                    errors.append({"filename": original_filename, "error": f"Page count out of allowed range."})
                    continue
                
                file_stream.seek(0)
                gcs_uri, blob_name, file_size, gcs_error = gcs_service.upload_original_to_gcs(file_stream, filename, file.content_type, current_user_email)
                if gcs_error:
                    errors.append({"filename": original_filename, "error": f"GCS Upload Failed: {gcs_error}"})
                    continue

                # destination_path is None for manual uploads (only used for scheduled batch process)
                doc_id, metadata_error = create_doc_metadata(current_user_email, filename, gcs_uri, blob_name, file.content_type, file_size, "Pending", 0, 0, destination_path=None)
                if metadata_error:
                    errors.append({"filename": original_filename, "error": f"Metadata Creation Failed: {metadata_error}", "gcs_uri": gcs_uri})
                    continue

                redis_conn = get_redis_client()
                if redis_conn:
                    task_string = f"split_and_process_pdf::{doc_id}::{gcs_uri}"
                    redis_conn.rpush(QUEUE_NAME, task_string)
                else:
                    warnings.append({"filename": original_filename, "warning": "Could not connect to Redis to enqueue processing task."})
                
                Thread(target=update_search_keywords_by_doc_id, args=(doc_id,)).start()
                results.append({"filename": original_filename, "gcs_uri": gcs_uri, "firestore_doc_id": doc_id, "status": "Pending"})
                
                # Log successful file upload
                log_file_activity(
                    user_email=current_user_email,
                    activity_type=ActivityTypes.FILE_UPLOAD,
                    filename=filename,
                    file_info={'file_size': file_size, 'doc_id': doc_id, 'status': 'queued_for_processing'},
                    request_obj=request
                )
            except Exception as e:
                errors.append({"filename": original_filename, "error": f"Unexpected error: {str(e)}"})
        elif file:
            errors.append({"filename": file.filename, "error": "File type not allowed"})

    status_code = 207 if errors else 201
    return jsonify({"msg": "Upload process finished.", "successful_uploads": results, "failed_uploads": errors, "upload_warnings": warnings}), status_code





@doc_bp.route('/chat', methods=['POST'])
@jwt_required()
def chat_with_documents():
    """Handles user chat queries by orchestrating various chat services."""
    start_time = time.time()
    current_user_email = get_jwt_identity()
    data = request.get_json()
    query = data.get('query')
    session_id = data.get('session_id', None)
    message_id = uuid.uuid4().hex

    if not query:
        return jsonify({"msg": "Query is required"}), 400

    def generator():
        try:
            # 1. Analyze Query for Entity Filters (Router Agent)
            yield f"data: {json.dumps({'status': 'Analyzing query...'})}\n\n"
            router_result = SearchRouterService.analyze_query(query)
            metadata_filters = router_result.get("filters", {})
            if metadata_filters:
                debug_log(f"Router Agent found filters: {metadata_filters}")

            # 2. Generate Query Embedding
            yield f"data: {json.dumps({'status': 'Understanding your question...'})}\n\n"
            query_embedding = embedding_service.generate_query_embedding(query, session_id, message_id)

            # 3. Find Relevant Chunks (with filters)
            t_vector_start = time.time()
            yield f"data: {json.dumps({'status': 'Looking through your documents...'})}\n\n"
            
            # Create a queue for status messages from vector search fallback
            status_queue = queue.Queue()
            search_result = [None]  # Use list to store result from thread
            search_error = [None]   # Use list to store any errors
            
            def status_callback(message):
                """Callback to send status updates during vector search fallback attempts"""
                status_queue.put(message)
            
            # Run vector search in a separate thread so we can stream status messages
            def search_thread():
                try:
                    search_result[0] = vector_search_service.find_relevant_chunks(
                        query_embedding, query, session_id, message_id, 
                        user_email=current_user_email,
                        metadata_filters=metadata_filters, # Pass the extracted filters
                        status_callback=status_callback
                    )
                except Exception as e:
                    search_error[0] = e
                finally:
                    # Signal that search is complete
                    status_queue.put(None)
            
            # Start the search in a background thread
            thread = threading.Thread(target=search_thread, daemon=True)
            thread.start()
            
            # Stream status messages as they arrive from the search thread
            while True:
                try:
                    # Wait for messages with a timeout to avoid blocking forever
                    message = status_queue.get(timeout=0.5)
                    
                    # None signals the search is complete
                    if message is None:
                        break
                    
                    # Yield status message to client
                    yield f"data: {json.dumps({'status': message})}\n\n"
                except queue.Empty:
                    # No message yet, check if thread is still alive
                    if not thread.is_alive():
                        # Thread finished but might have one final message
                        try:
                            message = status_queue.get_nowait()
                            if message is not None:
                                yield f"data: {json.dumps({'status': message})}\n\n"
                        except queue.Empty:
                            break
            
            # Check if there was an error in the search thread
            if search_error[0]:
                raise search_error[0]
            
            neighbors = search_result[0]
            
            t_vector_end = time.time()
            logger.info(f"PERF_LOG: Step 2 (Vector Search) took: {t_vector_end - t_vector_start:.4f} seconds.")
            debug_perf(f" Step 2 (Vector Search) took: {t_vector_end - t_vector_start:.4f} seconds.", flush=True)
            
            if not neighbors:
                no_info_answer = "Based on the available documents, I could not find specific information to answer your query."
                for chunk in stream_no_grounded_answer_response(no_info_answer, [], "NO_NEIGHBORS", session_id, message_id, query, current_user_email):
                    yield chunk
                return

            # 3. Fetch Data from Firestore
            t_fetch_start = time.time()
            yield f"data: {json.dumps({'status': 'Gathering relevant information...'})}\n\n"
            chunk_ids = [neighbor['id'] for neighbor in neighbors]
            context_chunk_map = firestore_service.fetch_chunk_data(chunk_ids, query, session_id, message_id, user_email=current_user_email)
            parent_metadata_map = firestore_service.fetch_parent_document_metadata(context_chunk_map, query, session_id, message_id)
            t_fetch_end = time.time()

            # print chunk map and parent meta data 
            debug_log(f"context_chunk_map: {context_chunk_map}")
            debug_log(f"parent_metadata_map: {parent_metadata_map}")

            logger.info(f"PERF_LOG: Step 3 (Firestore Fetch for {len(chunk_ids)} chunks) took: {t_fetch_end - t_fetch_start:.4f} seconds.")
            debug_perf(f" Step 3 (Firestore Fetch for {len(chunk_ids)} chunks) took: {t_fetch_end - t_fetch_start:.4f} seconds.", flush=True)
            
            context_texts = [chunk.get("ocr_text_preview") or chunk.get("extracted_text_preview") for chunk in context_chunk_map.values() if chunk.get("ocr_text_preview") or chunk.get("extracted_text_preview")]
            debug_perf(f" Context preparation. Valid text chunks found: {len(context_texts)}", flush=True)
            
            if not context_texts:
                no_context_answer = "I found some potentially relevant document sections, but could not extract sufficient information to answer your query."
                for chunk in stream_no_grounded_answer_response(no_context_answer, [], "NO_CONTEXT_TEXT", session_id, message_id, query, current_user_email):
                    yield chunk
                return

            # Prepare structured context chunks for advanced LLM analysis
            structured_context_chunks = []
            # chunk_ids comes from neighbors, so it's ordered by relevance (distance)
            # User Requirement: Top 10 nearest neighbor chunks
            top_chunk_ids = chunk_ids[:50]  # Increased from 20 to 50 for better context

            debug_log(f"top_chunk_ids: {top_chunk_ids}")
            
            for cid in top_chunk_ids:
                if cid in context_chunk_map:
                    chunk_data = context_chunk_map[cid]
                    # print chunk_data 
                    debug_log(f"chunk_data+++: {chunk_data}")
                    text = chunk_data.get("ocr_text_preview") or chunk_data.get("extracted_text_preview")
                    if text:
                        # Find distance from neighbors list
                        dist = next((n['distance'] for n in neighbors if n['id'] == cid), "N/A")
                        structured_context_chunks.append({
                            'id': cid,
                            'original_doc_id': chunk_data.get('original_doc_firestore_id'),
                            'text': text,
                            'distance': dist
                        })

            # 4. Fetch Chat History
            t_hist_start = time.time()
            chat_history_formatted = firestore_service.fetch_chat_history(session_id, current_user_email, query, message_id)
            t_hist_end = time.time()
            logger.info(f"PERF_LOG: Step 4 (Chat History) took: {t_hist_end - t_hist_start:.4f} seconds.")
            debug_perf(f" Step 4 (Chat History) took: {t_hist_end - t_hist_start:.4f} seconds.", flush=True)

            # 5. Stream LLM Response
            t_llm_start = time.time()
            yield f"data: {json.dumps({'status': 'Preparing your answer...'})}\n\n"
            logger.info(f"PERF_LOG: Starting LLM Stream (Time from start: {t_llm_start - start_time:.4f}s)")
            debug_perf(f" Starting LLM Stream (Time from start: {t_llm_start - start_time:.4f}s)", flush=True)
            for chunk in stream_response(query, structured_context_chunks, chat_history_formatted, context_chunk_map, parent_metadata_map, session_id, message_id, current_user_email, start_time):
                yield chunk

        except Exception as e:
            logger.error(f"Error in chat orchestrator for message_id {message_id}: {e}", exc_info=True)
            
            error_message = f"An unexpected error occurred: {str(e)}"
            if isinstance(e, GeminiConfigurationError):
                 error_message = "The AI service is temporarily unavailable due to a configuration issue. Please contact support."
            
            yield f"data: {json.dumps({'error': error_message})}\n\n"

    return Response(generator(), mimetype='text/event-stream')

def stream_response(query, context_chunks, chat_history, context_chunk_map, parent_metadata_map, session_id, message_id, current_user_email, start_time):
    """Generator function to stream the main LLM response, with parallel reference determination."""
    full_answer = ""
    references_result = []
    reference_time_result = [0.0]
    text_response_time_result = [0.0]

    try:
        # Integrated Citation Strategy: 
        # The model will output [[BEST_MATCH_ANALYSIS: {...}]] at the end.
        
        full_answer = ""
        citation_id = None
        analysis_json = None
        
        # Start the stream
        text_start_time = time.time()
        # Pass context_chunks (structured list) to service
        response_stream = vertex_ai_service.generate_chat_response(query, context_chunks, chat_history, stream=True, user_email=current_user_email)
        
        stop_streaming_visible = False

        for chunk in response_stream:
            if hasattr(chunk, 'parts'):
                text_chunk = ''.join(part.text for part in chunk.parts if hasattr(part, 'text'))
            else:
                text_chunk = chunk
            
            if text_chunk:
                full_answer += text_chunk
                
                # Check for Analysis Block Start
                if "[[BEST_MATCH_ANALYSIS:" in full_answer and not stop_streaming_visible:
                    # Found the start of the analysis block
                    # We need to split and only yield the part BEFORE the block
                    
                    # Case 1: The tag is inside the current text_chunk
                    if "[[BEST_MATCH_ANALYSIS:" in text_chunk:
                         pre_tag, post_tag = text_chunk.split("[[BEST_MATCH_ANALYSIS:", 1)
                         if pre_tag:
                             yield f"data: {json.dumps({'chunk': pre_tag})}\n\n"
                    
                    # Case 2: The tag was split across chunks, and this chunk completes it.
                    # We don't yield this chunk because it's part of the tag or after it.
                    # But we must ensure we didn't miss any pre-tag text from previous chunks?
                    # No, previous chunks were yielded. 
                    # The only risk is if text_chunk contains "MATCH_ANALYSIS:" and we drop it. Correct.
                    
                    stop_streaming_visible = True
                    # Continue consuming stream to get the full JSON, but don't yield data
                
                elif not stop_streaming_visible:
                    # Normal text
                    yield f"data: {json.dumps({'chunk': text_chunk})}\n\n"

        text_response_time = time.time() - text_start_time
        
        # Parse the Analysis Block from full_answer
        try:
            # Improved Regex: Capture everything between the tags
            match = re.search(r'\[\[BEST_MATCH_ANALYSIS:(.*?)\]\]', full_answer, re.DOTALL)
            if match:
                json_str = match.group(1).strip()
                # Clean up any potential markdown code blocks inside the tag
                json_str = json_str.replace("```json", "").replace("```", "").strip()
                
                analysis_json = json.loads(json_str)
                debug_log(f"LLM ANALYSIS: {json.dumps(analysis_json, indent=2)}")
                
                if not analysis_json.get("justification"):
                    logger.warning(f"Accuracy Alert: LLM returned analysis without justification for message_id {message_id}")
                    debug_warn("Accuracy Alert: Missing justification in LLM response.")
                
                # Support both new plural and legacy singular keys
                citation_ids_raw = analysis_json.get("selected_chunk_ids")
                if citation_ids_raw is None:
                    citation_ids_raw = analysis_json.get("selected_chunk_id")
                
                confidence = analysis_json.get("confidence_level", "Low")
                
                # Validation: If confidence is Low, treat as no reference
                if confidence == "Low":
                    debug_log("Validation: Confidence is Low. Discarding citations.")
                    citation_ids_list = []
                else:
                    # Normalize to list
                    if isinstance(citation_ids_raw, list):
                        citation_ids_list = citation_ids_raw
                    elif citation_ids_raw and str(citation_ids_raw).lower() not in ["null", "none"]:
                        citation_ids_list = [citation_ids_raw]
                    else:
                        citation_ids_list = []

            else:
                # Fallback for legacy citation or missed tag
                 match_legacy = re.search(r'\[\[CITATION:\s*([a-zA-Z0-9_-]+)\s*\]\]', full_answer)
                 if match_legacy:
                     citation_ids_list = [match_legacy.group(1)]
                     debug_log(f"CITATION RECOVERED (Legacy): {citation_ids_list[0]}")
                 else:
                     citation_ids_list = []
                     debug_log("DEBUG: No Analysis Block or Citation found in LLM response.")
                     debug_log(f"DEBUG RAW TAIL: {full_answer[-500:]}") # Last 500 chars

        except Exception as e:
            logger.error(f"Failed to parse analysis block: {e}")
            debug_log(f"DEBUG: Parsing error: {e}")
            # print(f"DEBUG RAW BLOCK: {match.group(1) if match else 'No match'}")
            citation_ids_list = []

        # Determine References from Citations
        t_ref_start = time.time()
        actual_references_to_send = []
        
        # Helper: Get ordered IDs to resolve Ranks if needed (chunk_ids ordered by relevance)
        ordered_chunk_ids = [c.get('id') for c in context_chunks if isinstance(c, dict) and c.get('id')]

        debug_info = {
            "parsed_citation_ids": str(citation_ids_list),
            "mapping_attempted": False,
            "mapping_success": False,
            "chunks_available": len(ordered_chunk_ids),
            "lookup_success": False,
            "error_msg": None
        }

        for citation_id in citation_ids_list:
             if not citation_id: continue

             # Find metadata for this chunk
             # Remove brackets if they somehow stayed (clean up)
             citation_id_str = str(citation_id).replace("[", "").replace("]", "").strip()
             
             # Robustness: Handle if Model returns Rank (digit) instead of UUID
             # The prompt asks for ID, but models sometimes return the Rank number (e.g., "1", "4")
             # Also handle "Chunk 4", "Source 4", "#4"
             
             # Try to extract integer rank if it looks like a rank
             rank_match = re.search(r'^[\D]*(\d+)[\D]*$', citation_id_str)
             is_digit_like = citation_id_str.isdigit() or rank_match
             
             # We assume it's a rank if it's a small integer (e.g. < 20) or if it's definitely NOT a UUID (UUIDs are long)
             # UUID length is usually 32-36 chars. Ranks are 1-2 chars.
             is_likely_rank = is_digit_like and len(citation_id_str) < 5
             
             resolved_id = citation_id_str

             if is_likely_rank:
                 try:
                     debug_info["mapping_attempted"] = True
                     if rank_match:
                         rank_val = int(rank_match.group(1))
                     else:
                         rank_val = int(citation_id_str)
                         
                     rank_idx = rank_val - 1 # Rank is 1-based in prompt
                     debug_log(f"DEBUG: Attempting to map Rank '{rank_val}' (index {rank_idx}) to UUID. Available chunks: {len(ordered_chunk_ids)}")
                     
                     if 0 <= rank_idx < len(ordered_chunk_ids):
                         mapped_id = ordered_chunk_ids[rank_idx]
                         debug_log(f"DEBUG: Mapping returned Rank '{rank_val}' to UUID '{mapped_id}'")
                         resolved_id = mapped_id
                         debug_info["mapping_success"] = True
                     else:
                         msg = f"Rank '{rank_val}' is out of bounds for {len(ordered_chunk_ids)} chunks."
                         debug_log(f"DEBUG: {msg}")
                         debug_info["error_msg"] = msg
                         continue # Skip this invalid rank
                 except Exception as e:
                     logger.warning(f"Failed to map rank to ID: {e}")
                     debug_info["error_msg"] = str(e)
                     continue

             chunk_data = context_chunk_map.get(resolved_id)
             if chunk_data:
                 original_doc_id = chunk_data.get("original_doc_firestore_id")
                 if original_doc_id:
                     parent_metadata = parent_metadata_map.get(original_doc_id, {})
                     ref_obj = {
                        "filename": parent_metadata.get("original_filename", "Filename Unavailable"),
                        "doc_id": original_doc_id,
                        "chunk_id": resolved_id, 
                        "start_page": chunk_data.get("start_page"), 
                        "end_page": chunk_data.get("end_page")
                     }
                     # Avoid duplicates
                     if not any(r['chunk_id'] == resolved_id for r in actual_references_to_send):
                        actual_references_to_send.append(ref_obj)
                        debug_info["lookup_success"] = True
                        debug_log(f"CITATION SUCCESS: Resolved to {ref_obj['filename']} Page {ref_obj['start_page']} (Chunk {resolved_id})")
                 else:
                     debug_log(f"CITATION ERROR: Could not find parent doc ID for chunk {resolved_id}")
             else:
                 msg = f"Chunk ID {resolved_id} not found in context map."
                 debug_log(f"CITATION ERROR: {msg}")
        
        # Fallback: If no citation found (e.g. model forgot), use Top Vector Result (Index 0)
        # This acts as the Safety Net requested.
        # ONLY use fallback if we didn't get a valid analysis block (i.e. model failed to follow instructions).
        # If model explicitly returned null or Low confidence (analysis_json exists), we respect that and send NO reference.
        
        # MODIFIED: If analysis_json is present but citation_id is None (Low confidence),
        # we still check if the top vector match is VERY strong (e.g. implicit high confidence from vector DB).
        # But for now, we respect the LLM's "Low" decision.
        # The issue "reference not coming" might be due to parsing failure.
        
        if not analysis_json and not actual_references_to_send and ordered_chunk_ids and context_chunk_map:
             # Use the first ID from the ordered list that exists in context_chunk_map
             fallback_id = None
             for cid in ordered_chunk_ids:
                 if cid in context_chunk_map:
                     fallback_id = cid
                     break
             
             if fallback_id:
                 chunk_data = context_chunk_map.get(fallback_id)
                 if chunk_data:
                     original_doc_id = chunk_data.get("original_doc_firestore_id")
                     parent_metadata = parent_metadata_map.get(original_doc_id, {})
                     ref_obj = {
                        "filename": parent_metadata.get("original_filename", "Filename Unavailable"),
                        "doc_id": original_doc_id,
                        "chunk_id": fallback_id,
                        "start_page": chunk_data.get("start_page"), 
                        "end_page": chunk_data.get("end_page")
                     }
                     actual_references_to_send = [ref_obj]
                     debug_log(f"CITATION FALLBACK: Used top vector result {fallback_id}")
                     debug_info["fallback_used"] = True

        t_ref_end = time.time()
        reference_time_val = t_ref_end - t_ref_start
        
        # Clean full answer for saving (remove citation tag if it made it in)
        clean_answer = full_answer.split("[[BEST_MATCH_ANALYSIS:")[0].split("[[CITATION:")[0].strip()

        # Fallback for empty text answer
        if not clean_answer:
            if actual_references_to_send:
                clean_answer = "I have found relevant documents but could not generate a textual summary. Please review the references below."
            else:
                clean_answer = "I apologize, but I could not generate a response based on the provided documents."

        # ═══════════════════════════════════════════════════════════════════
        # RESPONSE VALIDATION - Verify answer relevance and citation grounding
        # ═══════════════════════════════════════════════════════════════════
        validation_result = None
        if config.ENABLE_RESPONSE_VALIDATION and clean_answer and citation_ids_list:
            yield f"data: {json.dumps({'status': 'Validating response accuracy...'})}\n\n"
            try:
                validation_result = validate_response(
                    query=query,
                    answer=clean_answer,
                    cited_chunks=citation_ids_list,
                    context_chunk_map=context_chunk_map
                )
                
                # Handle validation verdict
                # NOTE: We NEVER clear references - always show them
                # Validation only adds disclaimers/notes when needed
                verdict = validation_result.get("overall_verdict", "VALID")
                relevance_score = validation_result.get("answer_relevance_score", 1.0)
                grounding_score = validation_result.get("grounding_score", 1.0)
                user_note = validation_result.get("user_note", "")
                
                if verdict in ["INVALID", "PARTIAL"]:
                    logger.info(f"Response validation {verdict} for message_id {message_id}: {validation_result.get('issues', [])}")
                    clean_answer += get_validation_disclaimer(verdict, validation_result.get("issues", []), relevance_score, grounding_score, user_note)
                
                # Log validation metrics
                debug_log(f"VALIDATION: verdict={verdict}, relevance={validation_result.get('answer_relevance_score')}, grounding={validation_result.get('grounding_score')}")
                
            except Exception as e:
                logger.error(f"Response validation failed for message_id {message_id}: {e}")
                # Continue without validation on error - don't block the response

        # Build metadata for saving
        save_metadata = {}
        if analysis_json:
            save_metadata["analysis"] = analysis_json
        if validation_result:
            save_metadata["validation"] = validation_result
        
        new_session_id, _ = save_chat_message(
            session_id, current_user_email, query, clean_answer, 
            references=actual_references_to_send, message_id=message_id, 
            response_time=f"{text_response_time:.2f}", 
            reference_time=f"{reference_time_val:.2f}", 
            text_response_time=f"{text_response_time:.2f}",
            metadata=save_metadata if save_metadata else None
        )
        
        # Flush buffered logs from Redis to Firestore with the correct session_id
        # This is needed for the first message when session_id was None during processing
        if not session_id and new_session_id:
            SystemLogModel.flush_buffered_logs(message_id, new_session_id)
        
        # Log LLM response completion with detailed metrics
        SystemLogModel.add_log_entry(SystemLogModel(
            session_id=new_session_id or session_id, 
            message_id=message_id, 
            user_query_text=query,
            step_name="LLM_RESPONSE_COMPLETE", 
            status="SUCCESS",
            step_details={
                "text_response_time_sec": round(text_response_time, 2),
                "reference_resolution_time_sec": round(reference_time_val, 2),
                "total_answer_length": len(clean_answer),
                "references_found": len(actual_references_to_send),
                "citation_ids_from_llm": citation_ids_list if citation_ids_list else [],
                "analysis_block_found": bool(analysis_json),
                "confidence_level": analysis_json.get("confidence_level") if analysis_json else None,
                "fallback_reference_used": debug_info.get("fallback_used", False),
                "validation_verdict": validation_result.get("overall_verdict") if validation_result else None,
                "validation_relevance_score": validation_result.get("answer_relevance_score") if validation_result else None,
                "validation_grounding_score": validation_result.get("grounding_score") if validation_result else None
            }
        ).to_dict())
        
        # Log search query activity
        log_search_activity(
            user_email=current_user_email,
            query=query,
            session_id=new_session_id,
            results_count=len(actual_references_to_send) if actual_references_to_send else 0,
            request_obj=request
        )

        final_payload = {
            "event": "done", 
            "session_id": new_session_id, 
            "message_id": message_id, 
            "references": actual_references_to_send, 
            "response_time": f"{text_response_time:.2f}",
            "reference_time": f"{reference_time_val:.2f}",
            "text_response_time": f"{text_response_time:.2f}",
            "debug_info": debug_info,
            "text_answer": clean_answer # Pass the final text to frontend for fallback
        }
        
        if analysis_json:
            final_payload["metadata"] = {"analysis": analysis_json}
        
        if validation_result:
            final_payload["validation"] = {
                "verdict": validation_result.get("overall_verdict"),
                "relevance_score": validation_result.get("answer_relevance_score"),
                "grounding_score": validation_result.get("grounding_score")
            }
            
        yield f"data: {json.dumps(final_payload)}\n\n"

    except Exception as e:
        logger.error(f"Exception in stream_response generator for message_id {message_id}: {e}", exc_info=True)
        
        # Log the LLM error
        SystemLogModel.add_log_entry(SystemLogModel(
            session_id=session_id, 
            message_id=message_id, 
            user_query_text=query,
            step_name="LLM_RESPONSE_ERROR", 
            status="ERROR",
            error_message=str(e),
            step_details={"exception_type": type(e).__name__}
        ).to_dict())
        
        error_message = "An error occurred during streaming."
        if isinstance(e, GeminiConfigurationError):
             error_message = "The AI service is temporarily unavailable due to a configuration issue. Please contact support."
        
        yield f"data: {json.dumps({'error': error_message})}\n\n"
        final_error_payload = {"event": "done", "session_id": session_id, "message_id": message_id, "error_occurred": True, "references": []}
        yield f"data: {json.dumps(final_error_payload)}\n\n"

def stream_no_grounded_answer_response(answer_text, refs_to_send, reason_code, session_id, message_id, query, current_user_email):
    """Generator function for non-LLM responses (e.g., no neighbors found)."""
    yield f"data: {json.dumps({'chunk': answer_text})}\n\n"
    new_session_id, _ = save_chat_message(session_id, current_user_email, query, answer_text, references=refs_to_send, message_id=message_id)
    
    # Log the no-grounded-answer response
    SystemLogModel.add_log_entry(SystemLogModel(
        session_id=new_session_id or session_id, 
        message_id=message_id, 
        user_query_text=query,
        step_name="NO_GROUNDED_ANSWER_RESPONSE", 
        status="WARNING",
        step_details={
            "reason_code": reason_code,
            "answer_length": len(answer_text),
            "references_count": len(refs_to_send) if refs_to_send else 0
        }
    ).to_dict())
    
    # Flush buffered logs from Redis to Firestore with the correct session_id
    # This is needed for the first message when session_id was None during processing
    if not session_id and new_session_id:
        SystemLogModel.flush_buffered_logs(message_id, new_session_id)
    
    final_payload = {"event": "done", "session_id": new_session_id, "message_id": message_id, "references": refs_to_send}
    yield f"data: {json.dumps(final_payload)}\n\n"

# Other routes... (history, session management, etc.) remain below
@doc_bp.route('/history', methods=['GET'])
@jwt_required() 
def get_history():
    """Retrieves paginated global upload history, with optional search and status filters."""
    limit = request.args.get('limit', 10, type=int) 
    start_after_doc_id = request.args.get('start_after', None, type=str)
    search_term = request.args.get('search', None, type=str)
    frontend_status_filter = request.args.get('status', None, type=str) 

    limit = max(1, min(limit, 100)) 

    filter_to_granular_map = {
        "Pending": ["pending", "queued_for_splitting"],
        "Processing": [
            "processing", "splitting", "splitting_in_progress", 
            "pending_chunk_processing", "ocr_pending", "ocr_in_progress", 
            "pending_vectorization", "vectorizing"
        ],
        "Failed": [
            "error", "error_splitting", "error_creating_chunks", "incomplete", 
            "error_worker_failure", "ocr_failed", "embedding_failed", 
            "vectorization_failed", "upload_failed", "processing_error", "unknown"
        ],
        "Completed": ["completed"]
    }

    backend_status_query_value = None 
    if frontend_status_filter and frontend_status_filter != "All Statuses":
        standardized_filter_key = frontend_status_filter.capitalize() if frontend_status_filter else None
        
        granular_statuses_for_filter = filter_to_granular_map.get(standardized_filter_key)
        
        if granular_statuses_for_filter:
            backend_status_query_value = granular_statuses_for_filter
        else:
            logger.warning(f"Unknown status filter category received: {frontend_status_filter}. No status filter will be applied.")
            backend_status_query_value = None


    history, last_doc_id, total_items, error = get_upload_history( 
        limit=limit,
        start_after_doc_id=start_after_doc_id,
        search_term=search_term,
        status_filter=backend_status_query_value 
    )

    if error:
        if "Query requires a Firestore index" in error:
             return jsonify({"msg": error}), 400 
        else:
             return jsonify({"msg": error}), 500 

    status_map = {
        "queued_for_splitting": "Pending",
        "pending": "Pending", 
        "splitting_in_progress": "Processing",
        "pending_chunk_processing": "Processing",
        "processing": "Processing", 
        "completed": "Completed",
        "error": "Failed", 
        "error_splitting": "Failed",
        "error_creating_chunks": "Failed",
        "incomplete": "Failed", 
        "error_worker_failure": "Failed",
    }

    processed_history = []
    for doc_data_item in history: 
        mapped_doc_data = doc_data_item.copy() 
        raw_granular_status = mapped_doc_data.get('status')
        granular_status = raw_granular_status.strip() if raw_granular_status else None

        if granular_status in status_map:
            mapped_doc_data['status'] = status_map[granular_status]
        elif granular_status == 'completed': 
            mapped_doc_data['status'] = "Completed"
        elif granular_status and ('error' in granular_status.lower() or 'fail' in granular_status.lower() or 'incomplete' == granular_status):
            mapped_doc_data['status'] = "Failed"
        elif granular_status and ('pending' in granular_status.lower() or 'queued' in granular_status.lower()):
            if granular_status == "pending_chunk_processing": 
                 mapped_doc_data['status'] = status_map[granular_status] 
            else:
                 mapped_doc_data['status'] = "Pending"
        elif granular_status: 
            mapped_doc_data['status'] = "Processing" 
        else: 
            mapped_doc_data['status'] = "Pending" 
        
        processed_history.append(mapped_doc_data)

    # Log history page access
    current_user_email = get_jwt_identity()
    log_navigation_activity(
        user_email=current_user_email,
        page_name='Document History',
        page_path='/history',
        request_obj=request
    )

    return jsonify({
        "history": processed_history,
        "next_cursor": last_doc_id,
        "limit": limit,
        "search_term": search_term,
        "status": frontend_status_filter, 
        "total_items": total_items 
        }), 200


@doc_bp.route('/chat/sessions', methods=['GET'])
@jwt_required()
def list_chat_sessions():
    """Retrieves a list of chat session summaries for the logged-in user."""
    current_user_email = get_jwt_identity()
    limit = request.args.get('limit', 10, type=int) # Optional limit

    sessions, error = get_user_chat_sessions(current_user_email, limit)

    if error:
        if "Query requires a Firestore index" in error:
             return jsonify({"msg": error}), 400 
        else:
             return jsonify({"msg": error}), 500 

    return jsonify(sessions), 200


@doc_bp.route('/chat/sessions/<string:session_id>/messages', methods=['GET'])
@jwt_required()
def list_session_messages(session_id):
    """Retrieves all messages for a specific chat session."""
    current_user_email = get_jwt_identity()

    messages, error = get_session_messages(session_id, current_user_email)

    if error:
        if "not found" in error.lower():
            return jsonify({"msg": error}), 404 
        elif "access denied" in error.lower():
             return jsonify({"msg": error}), 403 
        else:
             return jsonify({"msg": error}), 500 

    return jsonify(messages), 200




# --- New Download Endpoint ---
@doc_bp.route('/download/<string:doc_id>', methods=['GET'])
@jwt_required()
def download_original_document(doc_id):
    debug_log("hitting")
    """Generates a signed URL for downloading the original document."""
    from app import db 
    current_user_email = get_jwt_identity() 

    try:
        doc_ref = db.collection("document_metadata").document(doc_id)
        debug_log("doc_ref value: ",doc_ref)
        doc_snap = doc_ref.get()

        if not doc_snap.exists:
            return jsonify({"msg": "Document metadata not found."}), 404

        doc_data = doc_snap.to_dict()
        debug_log("doc data: ",doc_data)

        blob_name = doc_data.get('gcs_blob_name')
        original_filename = doc_data.get('original_filename', 'downloaded_document.pdf') 

        if not blob_name:
            logger.error(f"GCS blob name not found in metadata for doc_id: {doc_id}")
            return jsonify({"msg": "GCS blob name not found in metadata."}), 500

        signed_url = gcs_service.generate_download_signed_url(blob_name, expiration_minutes=5)

        if not signed_url:
            logger.error(f"Failed to generate download URL for blob_name: {blob_name} (doc_id: {doc_id})")
            return jsonify({"msg": "Failed to generate download URL."}), 500

        return jsonify({
            "signed_url": signed_url,
            "filename": original_filename
        }), 200

    except Exception as e:
        logger.error(f"Failed to generate download URL for doc {doc_id}: {e}", exc_info=True)
        return jsonify({"msg": f"An internal error occurred: {e}"}), 500


# --- NEW SEPARATE ENDPOINTS ---

# --- Document Metadata Endpoint ---
@doc_bp.route('/<string:doc_id>/metadata', methods=['GET'])
@jwt_required()
def get_document_metadata(doc_id):
    """Retrieves basic metadata for a specific document."""
    current_user_email = get_jwt_identity() 
    try:
        doc_ref = db.collection("document_metadata").document(doc_id) 
        doc_snap = doc_ref.get() 

        if not doc_snap.exists:
            return jsonify({"msg": "Document not found."}), 404

        doc_data = doc_snap.to_dict()
        doc_data['id'] = doc_snap.id 

        if 'upload_timestamp' in doc_data and isinstance(doc_data['upload_timestamp'], datetime.datetime):
            doc_data['upload_timestamp'] = doc_data['upload_timestamp'].isoformat()
        if 'last_status_update' in doc_data and isinstance(doc_data['last_status_update'], datetime.datetime):
            doc_data['last_status_update'] = doc_data['last_status_update'].isoformat()

        blob_name = doc_data.get('gcs_blob_name')
        download_url = None
        if blob_name:
            download_url = gcs_service.generate_download_signed_url(blob_name, expiration_minutes=5)
            if not download_url:
                logger.warning(f"Failed to generate download URL for blob {blob_name} (doc {doc_id}).")
        doc_data['download_url'] = download_url 

        # Add actual user email address from admin API
        user_id = doc_data.get('user_email')
        user_email_address = None
        if user_id:
            try:
                admin_url = config.ADMIN_SERVER_URL
                user_api_url = f"{admin_url}/users/{user_id}/minimal"
                # Pass JWT secret in header for authentication
                headers = {
                    'X-JWT-Secret': config.JWT_SECRET_KEY
                }
                response = requests.get(user_api_url, headers=headers, timeout=5)
                if response.status_code == 200:
                    user_data = response.json()
                    user_email_address = user_data.get('email')
                else:
                    logger.warning(f"Failed to fetch user data from admin API: {response.status_code}")
            except Exception as e:
                logger.warning(f"Failed to fetch user email for user_id {user_id} from admin API: {e}")
        doc_data['user_email_address'] = user_email_address

        return jsonify(doc_data), 200

    except Exception as e:
        logger.error(f"Failed to retrieve metadata for doc {doc_id}: {e}", exc_info=True)
        return jsonify({"msg": f"An internal error occurred: {e}"}), 500

# --- Document Chunks Endpoint (Paginated) ---
@doc_bp.route('/<string:doc_id>/chunks', methods=['GET'])
@jwt_required()
def get_document_chunks_paginated(doc_id):
    """Retrieves paginated chunks for a specific document."""
    current_user_email = get_jwt_identity() 
    limit = request.args.get('limit', 20, type=int)
    start_after_chunk_id = request.args.get('start_after', None, type=str)
    limit = max(1, min(limit, 100))

    try:
        chunks, next_chunk_cursor, chunks_error = metadata_model.get_chunks_for_document(
            original_doc_id=doc_id,
            limit=limit,
            start_after_chunk_id=start_after_chunk_id
        )
        if chunks_error:
            logger.error(f"Error fetching chunks for doc {doc_id}: {chunks_error}")
            if "Query requires a Firestore index" in chunks_error:
                 return jsonify({"msg": chunks_error}), 400
            else:
                 return jsonify({"msg": "Failed to retrieve chunks."}), 500

        processed_chunks = []
        for chunk_data in chunks:
            mapped_chunk = chunk_data.copy()
            if 'classified_document_type' in mapped_chunk:
                mapped_chunk['classified_chunk_document_type_label'] = mapped_chunk.pop('classified_document_type')
            if 'selected_parser_processor_id' in mapped_chunk:
                mapped_chunk['used_parser_processor_id'] = mapped_chunk.pop('selected_parser_processor_id')
            processed_chunks.append(mapped_chunk)

        response_data = {
            "items": processed_chunks, 
            "next_cursor": next_chunk_cursor,
            "limit": limit
        }
        return jsonify(response_data), 200

    except Exception as e:
        logger.error(f"Failed to retrieve chunks for doc {doc_id}: {e}", exc_info=True)
        return jsonify({"msg": f"An internal error occurred: {e}"}), 500

# --- Chunk Details Endpoint ---
@doc_bp.route('/chunks/<string:chunk_id>/details', methods=['GET'])
@jwt_required()
def get_chunk_details_endpoint(chunk_id):
    """Retrieves detailed processing info for a specific chunk."""
    try:
        details = metadata_model.get_chunk_details(chunk_id)
        
        # Also fetch properties from the main chunk document (specifically 'entities')
        chunk_ref = db.collection("document_chunks").document(chunk_id)
        chunk_snap = chunk_ref.get()
        chunk_data = chunk_snap.to_dict() if chunk_snap.exists else {}
        
        if not details:
            if chunk_snap.exists:
                # If details don't exist but chunk does, create a partial details object
                details = {
                    "chunk_id": chunk_id, 
                    "extracted_entities": chunk_data.get("extracted_entities", {}),
                    "full_text": chunk_data.get("ocr_text_preview", "Full text not available in details view."),
                }
            else:
                return jsonify({"msg": "Chunk details not found."}), 404

        # Merge 'entities' from the main chunk document into the response
        if chunk_data and "entities" in chunk_data:
             details["entities"] = chunk_data["entities"]
        
        # Helper to strict types
        def serialize_datetime(obj):
            if isinstance(obj, datetime.datetime):
                return obj.isoformat()
            raise TypeError("Type not serializable")

        return json.dumps(details, default=serialize_datetime), 200, {'Content-Type': 'application/json'}

    except Exception as e:
        logger.error(f"Failed to retrieve details for chunk {chunk_id}: {e}", exc_info=True)
        return jsonify({"msg": f"An internal error occurred: {e}"}), 500


# --- Document Logs Endpoint (Paginated) ---
@doc_bp.route('/<string:doc_id>/logs', methods=['GET'])
@jwt_required()
def get_document_logs_paginated(doc_id):
    """Retrieves paginated processing logs for a specific document."""
    current_user_email = get_jwt_identity() 
    limit = request.args.get('limit', 50, type=int) 
    start_after_log_id = request.args.get('start_after', None, type=str)
    level_filter = request.args.get('level', None, type=str)
    chunk_id_filter = request.args.get('chunk_id', None, type=str) # Added chunk_id_filter
    limit = max(1, min(limit, 200)) 

    try:
        logs, next_log_cursor, logs_error = log_model.get_logs(
            document_id=doc_id, 
            limit=limit,
            start_after_doc_id=start_after_log_id,
            level=level_filter if level_filter and level_filter.upper() != 'ALL LEVELS' else None,
            chunk_id=chunk_id_filter # Pass chunk_id to model
        )

        if logs_error:
            logger.error(f"Error fetching logs for doc {doc_id}: {logs_error}")
            if "Query requires a Firestore index" in logs_error:
                 return jsonify({"msg": logs_error}), 400 
            else:
                 return jsonify({"msg": f"Failed to retrieve logs: {logs_error}"}), 500

        response_data = {
            "items": logs,
            "next_cursor": next_log_cursor,
            "limit": limit
        }
        return jsonify(response_data), 200

    except Exception as e:
        logger.error(f"Failed to retrieve logs for doc {doc_id}: {e}", exc_info=True)
        return jsonify({"msg": f"An internal error occurred: {e}"}), 500

# --- Processing Statistics Endpoint ---
@doc_bp.route('/<string:doc_id>/processing-stats', methods=['GET'])
@jwt_required()
def get_document_processing_stats(doc_id):
    """Retrieves processing statistics for a specific document."""
    try:
        stats, error = metadata_model.get_processing_stats(doc_id)
        if error:
            logger.error(f"Error fetching processing stats for doc {doc_id}: {error}")
            return jsonify({"msg": "Failed to retrieve processing statistics."}), 500
        
        if stats is None:
            return jsonify({"msg": "Processing statistics not found."}), 404

        return jsonify(stats), 200
    except Exception as e:
        logger.error(f"Failed to retrieve processing stats for doc {doc_id}: {e}", exc_info=True)
        return jsonify({"msg": f"An internal error occurred: {e}"}), 500

# --- Overall Processing Statistics Endpoint ---
@doc_bp.route('/overall-processing-stats', methods=['GET'])
@jwt_required()
def get_overall_stats():
    """Retrieves aggregate processing statistics for all documents."""
    from flask_jwt_extended import get_jwt
    claims = get_jwt()
    if claims.get("role") != "superadmin":
        return jsonify({"msg": "Permission denied. Super Admin access required."}), 403
    try:
        stats, error = metadata_model.get_overall_processing_stats()
        if error:
            logger.error(f"Error fetching overall stats: {error}")
            return jsonify({"msg": "Failed to retrieve overall processing statistics."}), 500
        
        return jsonify(stats), 200

    except Exception as e:
        logger.error(f"Failed to retrieve overall stats: {e}", exc_info=True)
        return jsonify({"msg": f"An internal error occurred: {e}"}), 500

# --- BULK Reprocess Endpoint ---
@doc_bp.route('/bulk-reprocess-legacy', methods=['POST'])
@jwt_required()
def bulk_reprocess_legacy_docs():
    """Triggers reprocessing for all documents that have legacy OCR chunks."""
    from flask_jwt_extended import get_jwt
    claims = get_jwt()
    if claims.get("role") != "superadmin":
        return jsonify({"msg": "Permission denied. Super Admin access required."}), 403
    current_user_email = get_jwt_identity()
    logger.info(f"BULK REPROCESS request for legacy docs by user {current_user_email}")
    
    try:
        redis_conn = get_redis_client()
        if not redis_conn:
            logger.error(f"Manual legacy reprocess by {current_user_email}: Redis connection failed")
            # Log failure to system_logs
            try:
                db.collection("system_logs").add({
                    "timestamp": firestore.SERVER_TIMESTAMP,
                    "document_id": "LEGACY_REPROCESS_MANUAL",
                    "step_name": "LEGACY_REPROCESS_MANUAL_FAILED",
                    "details": {
                        "error": "Redis connection failed",
                        "triggered_by": current_user_email,
                        "source": "MANUAL_API"
                    },
                    "status": "ERROR"
                })
            except Exception as log_err:
                logger.warning(f"Failed to log Redis connection error: {log_err}")
            return jsonify({"msg": "Failed to connect to Redis."}), 500
            
        # ASYNC ORCHESTRATION: Queue a single job task with trigger source
        task_str = f"initiate_bulk_reprocess_legacy::{current_user_email}::MANUAL"
        redis_conn.rpush(QUEUE_NAME, task_str)
        
        logger.info(f"Queued async bulk legacy reprocess job: {task_str}")
        
        # Log successful trigger to activity_logs
        log_admin_activity(
            user_email=current_user_email,
            admin_action='manual_legacy_reprocess_triggered',
            target_info={"task": task_str},
            request_obj=request
        )
        
        # Log successful trigger to system_logs
        try:
            db.collection("system_logs").add({
                "timestamp": firestore.SERVER_TIMESTAMP,
                "document_id": "LEGACY_REPROCESS_MANUAL",
                "step_name": "LEGACY_REPROCESS_MANUAL_TRIGGERED",
                "details": {
                    "triggered_by": current_user_email,
                    "source": "MANUAL_API",
                    "task": task_str
                },
                "status": "INFO"
            })
        except Exception as log_err:
            logger.warning(f"Failed to log manual legacy reprocess trigger: {log_err}")

        return jsonify({
            "msg": f"Bulk reprocessing job initiated successfully. Progress will be tracked in background.", 
            "status": "queued"
        }), 200
            

        
    except Exception as e:
        logger.error(f"Unexpected error during bulk reprocess: {e}", exc_info=True)
        # Log exception to system_logs
        try:
            db.collection("system_logs").add({
                "timestamp": firestore.SERVER_TIMESTAMP,
                "document_id": "LEGACY_REPROCESS_MANUAL",
                "step_name": "LEGACY_REPROCESS_MANUAL_FAILED",
                "details": {
                    "error": str(e),
                    "triggered_by": current_user_email,
                    "source": "MANUAL_API"
                },
                "status": "ERROR"
            })
        except Exception as log_err:
            logger.warning(f"Failed to log manual legacy reprocess error: {log_err}")
        return jsonify({"msg": f"An internal error occurred: {e}"}), 500


@doc_bp.route('/legacy-reprocess-history', methods=['GET'])
@jwt_required()
def get_legacy_reprocess_history():
    """
    Fetches paginated history of legacy reprocess runs.
    Query params:
    - limit: Number of runs to return (default: 10, max: 50)
    - start_after: Document ID for pagination
    - status: Filter by status (optional)
    """
    from flask_jwt_extended import get_jwt
    claims = get_jwt()
    if claims.get("role") != "superadmin":
        return jsonify({"msg": "Permission denied. Super Admin access required."}), 403
    
    try:
        limit = min(int(request.args.get('limit', 10)), 50)  # Cap at 50
        start_after = request.args.get('start_after')
        status_filter = request.args.get('status')
        
        # Query bulk_process_runs collection
        query = db.collection("bulk_process_runs").where(
            "type", "==", "legacy_reprocess_selective"
        ).order_by("start_time", direction=firestore.Query.DESCENDING).limit(limit)
        
        if start_after:
            start_doc = db.collection("bulk_process_runs").document(start_after).get()
            if start_doc.exists:
                query = query.start_after(start_doc)
        
        docs = query.stream()
        runs = []
        last_doc_id = None
        
        for doc in docs:
            data = doc.to_dict()
            last_doc_id = doc.id
            
            # Apply status filter in Python if needed
            if status_filter and data.get("status") != status_filter:
                continue
            
            runs.append({
                "run_id": data.get("run_id"),
                "start_time": data.get("start_time").isoformat() if data.get("start_time") else None,
                "end_time": data.get("end_time").isoformat() if data.get("end_time") else None,
                "trigger_source": data.get("trigger_source", "UNKNOWN"),
                "initiated_by": data.get("initiated_by"),
                "status": data.get("status"),
                "docs_found": data.get("docs_found", 0),
                "docs_queued": data.get("docs_queued", 0),
                "docs_skipped": data.get("docs_skipped", 0),
                "skip_aged_chunks": data.get("skip_aged_chunks", False),
                "max_age_days": data.get("max_age_days"),
                "errors": data.get("errors", [])
            })
        
        return jsonify({
            "runs": runs,
            "next_cursor": last_doc_id,
            "limit": limit
        }), 200
        
    except Exception as e:
        logger.error(f"Error fetching legacy reprocess history: {e}", exc_info=True)
        return jsonify({"msg": "Failed to fetch history."}), 500


# --- NEW Reprocess Endpoint ---
@doc_bp.route('/<string:doc_id>/reprocess', methods=['POST'])
@jwt_required()
def reprocess_document(doc_id):
    """Triggers reprocessing for a document that previously encountered an error."""
    current_user_email = get_jwt_identity()
    logger.info(f"Reprocess request received for doc {doc_id} by user {current_user_email}")

    try:
        doc_ref = db.collection("document_metadata").document(doc_id)
        doc_snap = doc_ref.get()

        if not doc_snap.exists:
            logger.warning(f"Reprocess failed: Document {doc_id} not found.")
            return jsonify({"msg": "Document not found."}), 404

        doc_data = doc_snap.to_dict()
        current_status = doc_data.get('status')
        if current_status != 'error': # This should be 'Failed' if we simplify statuses
            logger.warning(f"Reprocess skipped: Document {doc_id} status is '{current_status}', not 'error' or 'Failed'.")
            return jsonify({"msg": f"Document status is '{current_status}'. Reprocessing only allowed for 'error' or 'Failed' status."}), 400

        gcs_uri = doc_data.get('gcs_uri')
        if not gcs_uri:
            logger.error(f"Reprocess failed: GCS URI missing for document {doc_id}.")
            return jsonify({"msg": "Cannot reprocess: GCS URI missing from metadata."}), 500

        logger.info(f"Updating status for doc {doc_id} to 'pending' for reprocessing.")
        update_document_status(doc_id, "pending", "Reprocessing triggered by user.") # This should become "Pending"

        redis_conn = get_redis_client()
        if not redis_conn:
            logger.error(f"Reprocess failed: Could not connect to Redis for doc {doc_id}.")
            return jsonify({"msg": "Failed to connect to Redis to enqueue reprocessing task."}), 500

        try:
            # Ensure this task format matches what the worker expects for the 17-step flow
            task_string = f"split_and_process_pdf::{doc_id}::{gcs_uri}" 
            redis_conn.rpush(QUEUE_NAME, task_string)
            logger.info(f"Successfully enqueued reprocessing task for doc {doc_id}: {task_string}")
        except Exception as redis_e:
            logger.error(f"Reprocess failed: Error enqueuing task for doc {doc_id}: {redis_e}", exc_info=True)
            return jsonify({"msg": f"Failed to enqueue reprocessing task: {redis_e}"}), 500

        return jsonify({"msg": "Document reprocessing successfully initiated."}), 200

    except Exception as e:
        logger.error(f"Unexpected error during reprocessing request for doc {doc_id}: {e}", exc_info=True)
        return jsonify({"msg": f"An internal error occurred: {e}"}), 500

@doc_bp.route('/<string:doc_id>/force-reprocess', methods=['POST'])
@jwt_required()
def force_reprocess_document(doc_id):
    """
    Forces a document to be reprocessed, regardless of its current state.
    This involves deleting existing chunks and re-queueing the document.
    """
    current_user_email = get_jwt_identity()
    logger.info(f"FORCE REPROCESS request for doc {doc_id} by user {current_user_email}")

    try:
        doc_ref = db.collection("document_metadata").document(doc_id)
        doc_snap = doc_ref.get()

        if not doc_snap.exists:
            return jsonify({"msg": "Document not found."}), 404

        doc_data = doc_snap.to_dict()
        gcs_uri = doc_data.get('gcs_uri')

        if not gcs_uri:
            return jsonify({"msg": "Cannot reprocess: GCS URI missing."}), 500

        # 1. Delete existing chunks from Firestore and GCS
        logger.info(f"Deleting existing chunks for parent doc {doc_id}")
        delete_result = bulk_processing_utils.delete_all_chunks_for_parent(doc_id)
        if "error" in delete_result:
            logger.error(f"Error during chunk deletion for doc {doc_id}: {delete_result['error']}")
            # Continue anyway, but log the error
        
        # 2. Reset parent document status
        logger.info(f"Resetting status for doc {doc_id} to 'queued_for_splitting'")
        doc_ref.update({
            "status": "queued_for_splitting",
            "error_message": firestore.DELETE_FIELD,
            "total_chunks": 0,
            "completed_chunks": 0,
            "last_status_update": datetime.datetime.now(tz=datetime.timezone.utc)
        })

        # 3. Re-enqueue the task
        redis_conn = get_redis_client()
        if not redis_conn:
            return jsonify({"msg": "Failed to connect to Redis."}), 500

        task_string = f"split_and_process_pdf::{doc_id}::{gcs_uri}"
        redis_conn.rpush(QUEUE_NAME, task_string)
        logger.info(f"Successfully enqueued force reprocess task for doc {doc_id}")

        return jsonify({"msg": "Document force reprocessing successfully initiated."}), 200

    except Exception as e:
        logger.error(f"Unexpected error during force reprocess for doc {doc_id}: {e}", exc_info=True)
        return jsonify({"msg": f"An internal error occurred: {e}"}), 500

@doc_bp.route('/chunks/<string:chunk_id>/reprocess', methods=['POST'])
@jwt_required()
def reprocess_chunk(chunk_id):
    """Triggers reprocessing for a single chunk."""
    current_user_email = get_jwt_identity()
    logger.info(f"Reprocess request received for chunk {chunk_id} by user {current_user_email}")

    try:
        chunk_ref = db.collection("document_chunks").document(chunk_id)
        chunk_snap = chunk_ref.get()

        if not chunk_snap.exists:
            return jsonify({"msg": "Chunk not found."}), 404

        chunk_data = chunk_snap.to_dict()

        logger.info(f"Chunk data: {chunk_data}")

        parent_doc_id = chunk_data.get('parent_doc_id') or chunk_data.get('original_doc_firestore_id')
        
        if not parent_doc_id:
             return jsonify({"msg": "Parent document ID not found for chunk."}), 500

        # Update chunk status to pending_reprocess
        logger.info(f"Updating status for chunk {chunk_id} to 'pending_reprocess'.")
        chunk_ref.update({
            "status": "pending_reprocess",
            "error_message": firestore.DELETE_FIELD,
            "last_updated": datetime.datetime.now(tz=datetime.timezone.utc)
        })

        # Enqueue task for the parent document which will pick up any pending chunks
        redis_conn = get_redis_client()
        if not redis_conn:
            logger.error(f"Reprocess failed: Could not connect to Redis for chunk {chunk_id}.")
            return jsonify({"msg": "Failed to connect to Redis to enqueue reprocessing task."}), 500

        # Reuse existing legacy reprocess task which processes all 'pending_reprocess' chunks for a doc
        task_string = f"reprocess_legacy_doc_chunks::{parent_doc_id}"
        redis_conn.rpush(QUEUE_NAME, task_string)
        
        logger.info(f"Successfully enqueued reprocessing task for chunk {chunk_id} (via parent {parent_doc_id})")

        return jsonify({"msg": "Chunk reprocessing successfully initiated."}), 200

    except Exception as e:
        logger.error(f"Unexpected error during chunk reprocessing request for {chunk_id}: {e}", exc_info=True)
        return jsonify({"msg": f"An internal error occurred: {e}"}), 500

def _run_migration_background(user_email):
    """
    Background worker for the migration task.
    Scans all chunks and backfills 'extraction_source'.
    Uses pagination with batches of 500 to prevent OOM.
    """
    import time
    from app import db
    
    logger.info(f"Background Migration Started by {user_email}")
    
    try:
        chunks_ref = db.collection("document_chunks")
        
        updated_count = 0
        legacy_count = 0
        llm_count = 0
        skipped_count = 0
        
        batch = db.batch()
        batch_size = 0
        total_examined = 0
        
        start_time = time.time()
        last_doc = None
        batch_size_limit = 500
        
        while True:
            # 1. Fetch chunks using pagination logic
            query_ref = chunks_ref.order_by("__name__").limit(batch_size_limit)
            if last_doc:
                query_ref = query_ref.start_after(last_doc)
            
            docs = list(query_ref.stream())
            if not docs:
                break
                
            last_doc = docs[-1]
            
            for doc in docs:
                total_examined += 1
                data = doc.to_dict()
                
                # Skip if already tagged
                if data.get('extraction_source'):
                    skipped_count += 1
                    continue
                    
                # Determine source
                is_failed = data.get('extracted_entities', {}).get('_extraction_failed') == True
                new_source = "legacy" if is_failed else "llm"
                
                batch.update(doc.reference, {"extraction_source": new_source})
                batch_size += 1
                updated_count += 1
                
                if new_source == "legacy":
                    legacy_count += 1
                else:
                    llm_count += 1
                
                # Commit batch every 400
                if batch_size >= 400:
                    batch.commit()
                    batch = db.batch()
                    batch_size = 0
                    
                # Log progress every 1000 examined
                if total_examined % 1000 == 0:
                    elapsed = time.time() - start_time
                    rate = total_examined / elapsed if elapsed > 0 else 0
                    logger.info(f"[Migration Progress] Examined: {total_examined}, Updated: {updated_count} (Legacy: {legacy_count}, LLM: {llm_count}), Rate: {rate:.1f} docs/sec")
                    
        if batch_size > 0:
            batch.commit()
            
        logger.info(f"[Migration Complete] Finished in {time.time() - start_time:.2f}s. Total Examined: {total_examined}, Updated: {updated_count}. Legacy: {legacy_count}, LLM: {llm_count}, Skipped: {skipped_count}")

    except Exception as e:
        logger.error(f"[Migration Failed] Detailed error: {e}", exc_info=True)

@doc_bp.route('/maintenance/tag-legacy-chunks', methods=['POST'])
@jwt_required()
def tag_legacy_chunks():
    """
    Superadmin only. Triggers ASYNC migration to normalize 'extraction_source'.
    Returns immediately so large datasets don't timeout.
    """
    current_user_email = get_jwt_identity()
    if not bulk_processing_utils.is_superadmin(current_user_email):
        return jsonify({"msg": "Superadmin access required"}), 403

    # Spawn background thread
    thread = threading.Thread(target=_run_migration_background, args=(current_user_email,))
    thread.daemon = True # Ensure it doesn't block server shutdown if hung
    thread.start()
            
    return jsonify({
        "msg": "Migration started in background.", 
        "details": "Check console logs for '[Migration Progress]' updates."
    }), 202

@doc_bp.route('/chunks/<string:chunk_id>/download', methods=['GET'])
@jwt_required()
def download_chunk(chunk_id):
    
    """Generates a signed URL for downloading a specific document chunk."""
    current_user_email = get_jwt_identity() 
    try:
        chunk_ref = db.collection("document_chunks").document(chunk_id)
        chunk_snap = chunk_ref.get()

        if not chunk_snap.exists:
            return jsonify({"msg": "Chunk not found."}), 404

        chunk_data = chunk_snap.to_dict()
        chunk_gcs_uri = chunk_data.get('gcs_path_chunk') 
        if not chunk_gcs_uri:
            logger.error(f"GCS URI not found in metadata for chunk_id: {chunk_id} (looked for 'gcs_path_chunk')") 
            return jsonify({"msg": "GCS URI not found in chunk metadata."}), 500
        
        if not chunk_gcs_uri.startswith("gs://"):
            logger.error(f"Invalid GCS URI format for chunk {chunk_id}: {chunk_gcs_uri}")
            return jsonify({"msg": "Invalid GCS URI format in chunk metadata."}), 500
        
        blob_name = chunk_gcs_uri[5:].split("/", 1)[1] 

        original_filename = chunk_data.get('original_filename', 'chunk_download')
        start_page = chunk_data.get('start_page', 'X')
        end_page = chunk_data.get('end_page', 'Y')
        suggested_filename = f"{os.path.splitext(original_filename)[0]}_chunk_p{start_page}-{end_page}.pdf"

        signed_url = gcs_service.generate_download_signed_url(blob_name, expiration_minutes=5)

        if not signed_url:
            logger.error(f"Failed to generate download URL for blob_name: {blob_name} (chunk_id: {chunk_id})")
            return jsonify({"msg": "Failed to generate download URL for chunk."}), 500

        return jsonify({
            "signed_url": signed_url,
            "filename": suggested_filename
        }), 200

    except Exception as e:
        logger.error(f"Failed to generate download URL for chunk {chunk_id}: {e}", exc_info=True)
        return jsonify({"msg": f"An internal error occurred: {e}"}), 500


# --- Status Endpoint ---
@doc_bp.route('/status/vector-search', methods=['GET'])
# @jwt_required() # Optional: Protect status endpoint
def get_vector_search_status():
    """Checks the status of the connection to the Vector Search endpoint."""
    status = vertex_ai_service.check_vector_search_availability()
    status_code = 200 if status.get("endpoint_connected") and status.get("index_deployed") else 503 # Service Unavailable
    return jsonify(status), status_code




# --- New Session Management Endpoints ---

@doc_bp.route('/chat/sessions/<string:session_id>', methods=['PATCH']) 
@jwt_required()
def rename_session(session_id):
    """Renames a specific chat session."""
    current_user_email = get_jwt_identity()
    data = request.get_json()
    new_title = data.get('title')

    if not new_title:
        return jsonify({"msg": "New title is required."}), 400

    success, message = metadata_model.rename_chat_session(session_id, current_user_email, new_title)

    if success:
        return jsonify({"msg": message}), 200
    else:
        status_code = 403 if "Access denied" in message else 404 if "not found" in message else 500
        return jsonify({"msg": message}), status_code

@doc_bp.route('/chat/sessions/<string:session_id>', methods=['DELETE'])
@jwt_required()
def delete_session(session_id):
    """Deletes a specific chat session."""
    current_user_email = get_jwt_identity()

    success, message = metadata_model.delete_chat_session(session_id, current_user_email)

    if success:
        return jsonify({"msg": message}), 200 
    else:
        status_code = 403 if "Access denied" in message else 500 
        return jsonify({"msg": message}), status_code
