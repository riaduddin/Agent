from flask import Blueprint, request, jsonify
from flask_jwt_extended import get_jwt_identity
import logging
from flask import Blueprint, request, jsonify
from flask_jwt_extended import get_jwt_identity
from threading import Thread # For background task in categorization
from collections import Counter
from app import db # For categorization
from app.utils.redis_client import get_redis_client, QUEUE_NAME # For categorization
from app.services import batch_processing_service
from app.services import cloudscheduler_batch_process_trigger
from app.services import category_api_service
from app.utils.utils import admin_or_superadmin_required, superadmin_required
from redis.exceptions import ConnectionError as RedisConnectionError # For categorization
from app.services.activity_log_service import log_admin_activity
from app.models.activity_log_model import ActivityTypes
from google.cloud import firestore
import uuid
from datetime import datetime
from app.utils.debug_logger import debug_log, debug_warn, debug_error

logger = logging.getLogger(__name__) # For categorization
batch_processing_bp = Blueprint('batch_processing_bp', __name__, url_prefix='/api/v1/batch')

def log_system_event(step_name, details, doc_id="BATCH_SYSTEM", status="INFO"):
    """Helper to log events to system_logs collection."""
    try:
        db.collection("system_logs").add({
            "timestamp": firestore.SERVER_TIMESTAMP,
            "document_id": doc_id,
            "step_name": step_name,
            "details": details,
            "status": status
        })
    except Exception as e:
        debug_error(f"Failed to write system log: {e}")

from app.services.categorization_service import categorize_document_strict

# Categorization task function (moved from batch_routes.py)
def queue_categorization_tasks():
    """
    Queries for ALL documents and processes them IN-LINE (no worker queue).
    Uses ONLY admin-defined categories (no Gemini fallback).
    This runs in a background thread.
    """
    debug_log("Starting inline_categorization_tasks thread.")
    logger.info("Starting background thread for inline categorization backfill.")
    
    log_system_event("BATCH_QUEUING_STARTED", {"message": "Background thread started for inline categorization backfill."})

    try:
        redis_client = get_redis_client()
        debug_log("Redis client obtained in thread.")
    except Exception as e:
        debug_error(f"Failed to get Redis client in thread: {e}")
        logger.error("Could not connect to Redis in thread. Aborting batch categorization.")
        log_system_event("BATCH_QUEUING_FAILED", {"error": str(e)}, status="ERROR")
        return

    try:
        debug_log("Setting categorization_batch_status to 'running'.")
        redis_client.set("categorization_batch_status", "running", ex=86400) # Lock for 24h
        
        # Fetch categories ONCE at the start
        debug_log("Fetching categories from API for batch run...")
        _, _, full_categories = category_api_service.get_categories_with_fallback()
        debug_log(f"Fetched {len(full_categories)} categories.")
        
        debug_log("Streaming all documents from Firestore to process inline.")
        docs_stream = db.collection("document_metadata").stream()
        count = 0
        total_checked = 0
        success_count = 0
        debug_log("Starting to loop through documents and process inline.")
        
        for doc in docs_stream:
            # Check for stop signal (if status is no longer 'running')
            if count % 10 == 0:
                current_status = redis_client.get("categorization_batch_status")
                if current_status != "running":
                    debug_log("Stop signal received. Halting batch processing.")
                    logger.info("Batch processing halted by user request (status changed from running).")
                    log_system_event("BATCH_STOPPED", {"message": "Batch processing stopped by user."})
                    break

            total_checked += 1
            doc_data = doc.to_dict()
            
            # Process ALL documents (User requested re-categorization of everything)
            # Remove previous check: if "categoriesd" not in doc_data:
            
            try:
                # Pass cached categories to avoid DB hits
                is_categorized = categorize_document_strict(doc.id, full_categories)
                if is_categorized:
                    success_count += 1
                    # Mark as processed (updating timestamp/flag)
                    db.collection("document_metadata").document(doc.id).update({
                        "categoriesd": "processed_inline_v2",
                        "last_categorized_at": firestore.SERVER_TIMESTAMP
                    })
            except Exception as inner_e:
                debug_error(f"Error processing doc {doc.id}: {inner_e}")
            
            count += 1
            # Log progress every 5 documents for better UI responsiveness (was 100)
            if count % 5 == 0:
                debug_log(f"Processed {count} documents (Success: {success_count})...")
                # logger.info(f"Processed {count} documents for categorization...") # Reduce log spam
                log_system_event("BATCH_QUEUING_PROGRESS", {
                    "processed_count": count, 
                    "success_count": success_count,
                    "total_scanned": total_checked
                })
            
            if total_checked % 1000 == 0:
                 debug_log(f"Scanned {total_checked} total documents...")

        debug_log(f"Finished processing loop. Scanned {total_checked} documents. Total processed: {count}. Success: {success_count}.")
        logger.info(f"Finished processing. Total documents processed: {count}.")
        log_system_event("BATCH_QUEUING_FINISHED", {
            "total_processed": count, 
            "success_count": success_count,
            "total_scanned": total_checked
        })

    except Exception as e:
        debug_error(f"An error occurred during batch processing: {e}")
        logger.error(f"An error occurred during batch processing: {e}", exc_info=True)
        log_system_event("BATCH_QUEUING_ERROR", {"error": str(e)}, status="ERROR")
    finally:
        debug_log("Setting categorization_batch_status to 'idle'.")
        redis_client.set("categorization_batch_status", "idle", ex=3600) # Reset status
        debug_log("Finished queue_categorization_tasks thread.")

@batch_processing_bp.route('/buckets', methods=['GET'])
@admin_or_superadmin_required
def get_buckets():
    """
    Get a list of all GCS buckets.
    """
    try:
        buckets = batch_processing_service.list_gcs_buckets()
        return jsonify(buckets), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@batch_processing_bp.route('/start', methods=['POST'])
@admin_or_superadmin_required
def start_batch_processing():
    """
    Start a new batch processing run.
    """
    data = request.get_json()
    bucket_name = data.get('bucket_name')
    if not bucket_name:
        return jsonify({"error": "Bucket name is required"}), 400
    
    try:
        run_id = batch_processing_service.start_run(bucket_name)
        
        # Log batch processing start
        current_user_email = get_jwt_identity()
        log_admin_activity(
            user_email=current_user_email,
            admin_action='batch_processing_start',
            target_info={'bucket_name': bucket_name, 'run_id': run_id},
            request_obj=request
        )
        
        return jsonify({"message": f"Batch processing started successfully with run ID: {run_id}"}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@batch_processing_bp.route('/reports', methods=['GET'])
@admin_or_superadmin_required
def get_reports():
    """
    Get reports of all batch processing runs.
    """
    try:
        runs = batch_processing_service.get_runs()
        return jsonify(runs), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@batch_processing_bp.route('/reports/<string:run_id>/files', methods=['GET'])
@admin_or_superadmin_required
def get_run_files(run_id):
    """
    Get a paginated list of files for a specific batch run.
    """
    limit = request.args.get('limit', 100, type=int)
    start_after = request.args.get('start_after', None, type=str)
    
    try:
        files, next_cursor, error = batch_processing_service.get_files_for_run(run_id, limit, start_after)
        if error:
            return jsonify({"error": error}), 500
        return jsonify({"files": files, "next_cursor": next_cursor}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@batch_processing_bp.route('/trigger', methods=['POST'])
@admin_or_superadmin_required
def trigger_batch_processing():
    """
    Manually trigger a batch processing run for testing.
    """
    # TODO: Implement trigger
    return jsonify({"message": "Manual trigger successful"}), 200

@batch_processing_bp.route('/config', methods=['POST'])
@admin_or_superadmin_required
def set_config():
    """
    Set the default bucket for batch processing.
    """
    data = request.get_json()
    bucket_name = data.get('bucket_name')
    if not bucket_name:
        return jsonify({"error": "Bucket name is required"}), 400
    
    current_user_id = get_jwt_identity()
    
    try:
        batch_processing_service.save_default_bucket(bucket_name, current_user_id)
        return jsonify({"message": "Configuration saved successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@batch_processing_bp.route('/config', methods=['GET'])
@admin_or_superadmin_required
def get_config():
    """
    Get the default bucket for batch processing.
    """
    try:
        bucket_name = batch_processing_service.get_default_bucket()
        return jsonify({"default_bucket_name": bucket_name}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@batch_processing_bp.route('/legacy-reprocess-config', methods=['GET'])
@admin_or_superadmin_required
def get_legacy_reprocess_config():
    """Get the nightly legacy reprocess toggle configuration."""
    try:
        config_data = batch_processing_service.get_legacy_reprocess_config()
        return jsonify(config_data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@batch_processing_bp.route('/legacy-reprocess-config', methods=['POST'])
@superadmin_required
def set_legacy_reprocess_config():
    """Enable/disable nightly legacy reprocess runs (superadmin only)."""
    data = request.get_json() or {}
    enabled = data.get('enabled')
    if enabled is None:
        return jsonify({"error": "'enabled' (true/false) is required"}), 400

    if isinstance(enabled, str):
        enabled = enabled.strip().lower() in {"true", "1", "yes", "y"}

    current_user_id = get_jwt_identity()
    try:
        batch_processing_service.set_legacy_reprocess_enabled(bool(enabled), current_user_id)
        log_admin_activity(
            user_email=current_user_id,
            admin_action='legacy_reprocess_toggle',
            target_info={"enabled": bool(enabled)},
            request_obj=request
        )
        log_system_event(
            "LEGACY_REPROCESS_SETTING_CHANGED",
            {
                "enabled": bool(enabled),
                "changed_by": current_user_id,
                "previous_state": "unknown"
            },
            doc_id="LEGACY_REPROCESS_CONFIG",
            status="INFO"
        )
        return jsonify({"message": "Legacy reprocess setting updated", "enabled": bool(enabled)}), 200
    except Exception as e:
        log_system_event(
            "LEGACY_REPROCESS_SETTING_CHANGE_FAILED",
            {
                "enabled": bool(enabled),
                "changed_by": current_user_id,
                "error": str(e)
            },
            doc_id="LEGACY_REPROCESS_CONFIG",
            status="ERROR"
        )
        return jsonify({"error": str(e)}), 500

@batch_processing_bp.route('/system/start-categorization-batch', methods=['POST'])
@admin_or_superadmin_required
def start_categorization_batch():
    """
    API endpoint for admins to trigger the batch categorization process.
    """
    debug_log("Hit start_categorization_batch endpoint.")
    try:
        redis_client = get_redis_client()
        debug_log("Successfully got Redis client.")
    except RedisConnectionError as e:
        debug_error(f"Failed to get Redis client: {e}")
        logger.error(f"API Error: Could not connect to Redis to start batch. {e}")
        return jsonify({"msg": "The queuing service (Redis) is currently unavailable. Please try again later."}), 503

    debug_log("Checking categorization_batch_status from Redis.")
    status = redis_client.get("categorization_batch_status")
    debug_log(f"Current status is: '{status}'")
    if status and status == 'running':
        debug_log("Batch is already running. Returning 409 conflict.")
        return jsonify({"msg": "A categorization batch is already in progress."}), 409

    debug_log("Admin triggered batch categorization. Starting background thread.")
    logger.info("Admin triggered batch categorization.")
    # Start the queuing process in a background thread
    thread = Thread(target=queue_categorization_tasks)
    thread.daemon = True
    thread.start()

    debug_log("Returning 202 Accepted response.")
    return jsonify({"msg": "Categorization batch process has been initiated."}), 202

@batch_processing_bp.route('/system/reset-categorization-batch-status', methods=['POST'])
@admin_or_superadmin_required
def reset_categorization_batch_status():
    """
    Resets the categorization batch status to 'idle'.
    """
    try:
        redis_client = get_redis_client()
        redis_client.set("categorization_batch_status", "idle")
        
        # Log the reset event so we can filter stats
        log_system_event("BATCH_RESET", {"message": "Batch status and stats reset by admin."})
        
        return jsonify({"msg": "Categorization batch status has been reset to idle."}), 200
    except Exception as e:
        logger.error(f"Error resetting batch status: {e}")
        return jsonify({"msg": "Failed to reset status", "error": str(e)}), 500

@batch_processing_bp.route('/system/categorization-batch-status', methods=['GET'])
@admin_or_superadmin_required
def get_categorization_batch_status():
    """
    Returns the status of the categorization batch and simple statistics.
    """
    try:
        redis_client = get_redis_client()
        status = redis_client.get("categorization_batch_status")
        status_str = status if status else "idle"

        # Get counts using aggregation queries for performance
        total_docs_query = db.collection("document_metadata").count()
        total_docs_snapshot = total_docs_query.get()
        total_docs = total_docs_snapshot[0][0].value

        logs_ref = db.collection("system_logs")
        
        # Check for last reset
        reset_query = logs_ref.where("step_name", "==", "BATCH_RESET")\
            .order_by("timestamp", direction=firestore.Query.DESCENDING).limit(1)
        reset_docs = list(reset_query.stream())
        last_reset_time = reset_docs[0].to_dict().get("timestamp") if reset_docs else None

        # Fetch latest progress log to estimate processed/pending
        progress_query = logs_ref.where("step_name", "in", [
            "BATCH_QUEUING_PROGRESS", 
            "BATCH_QUEUING_FINISHED"
        ]).order_by("timestamp", direction=firestore.Query.DESCENDING).limit(1)
        progress_docs = list(progress_query.stream())
        
        processed_count = 0 # Actually updated (actioned)
        total_scanned = 0   # Checked
        
        if progress_docs:
            data = progress_docs[0].to_dict()
            progress_time = data.get("timestamp")
            
            # Only use progress stats if they are newer than the last reset
            if not last_reset_time or (progress_time and progress_time > last_reset_time):
                details = data.get("details", {})
                processed_count = details.get("success_count", 0)
                total_scanned = details.get("total_scanned", 0)

        pending_count = max(0, total_docs - total_scanned)
        
        # Get recent activity logs
        logs_query = logs_ref.where("step_name", "in", [
            "categorization_finished", 
            "categorization_started",
            "BATCH_QUEUING_STARTED",
            "BATCH_QUEUING_PROGRESS",
            "BATCH_QUEUING_FINISHED",
            "BATCH_QUEUING_FAILED",
            "BATCH_QUEUING_ERROR",
            "BATCH_RESET",
            "BATCH_STOPPED"
        ]).order_by("timestamp", direction=firestore.Query.DESCENDING).limit(10)
        logs_docs = logs_query.stream()
        
        recent_activity = []
        
        for log in logs_docs:
            data = log.to_dict()
            recent_activity.append({
                "timestamp": data.get("timestamp"),
                "doc_id": data.get("document_id"),
                "action": data.get("step_name"),
                "details": data.get("details")
            })
            
        return jsonify({
            "status": status_str,
            "total_documents": total_docs,
            "processed_count": total_scanned, # Total scanned/checked
            "actioned_count": processed_count, # Successfully updated
            "pending_count": pending_count,
            "recent_activity": recent_activity
        }), 200

    except Exception as e:
        logger.error(f"Error getting categorization batch status: {e}", exc_info=True)
        return jsonify({"msg": "Failed to get status", "error": str(e)}), 500

@batch_processing_bp.route('/ignored-folders', methods=['POST'])
@admin_or_superadmin_required
def add_ignored_folder_route():
    """
    API endpoint for admins to add a folder to the ignored list.
    """
    data = request.get_json()
    folder_name = data.get('folder_name')
    if not folder_name:
        return jsonify({"msg": "Folder name is required."}), 400
    
    try:
        batch_processing_service.add_ignored_folder(folder_name)
        return jsonify({"msg": f"Folder '{folder_name}' added to ignored list."}), 200
    except Exception as e:
        logger.error(f"Error adding ignored folder via API: {e}", exc_info=True)
        return jsonify({"msg": "Failed to add ignored folder.", "error": str(e)}), 500

@batch_processing_bp.route('/ignored-folders', methods=['GET'])
@admin_or_superadmin_required
def get_ignored_folders_route():
    """
    API endpoint for admins to retrieve the list of ignored folders.
    """
    try:
        folders = batch_processing_service.get_ignored_folders()
        return jsonify({"ignored_folders": folders}), 200
    except Exception as e:
        logger.error(f"Error retrieving ignored folders via API: {e}", exc_info=True)
        return jsonify({"msg": "Failed to retrieve ignored folders.", "error": str(e)}), 500

@batch_processing_bp.route('/ignored-folders', methods=['DELETE'])
@admin_or_superadmin_required
def remove_ignored_folder_route():
    """
    API endpoint for admins to remove a folder from the ignored list.
    """
    data = request.get_json()
    folder_path = data.get('folder_path')
    if not folder_path:
        return jsonify({"msg": "Folder path is required."}), 400
    
    try:
        success = batch_processing_service.remove_ignored_folder(folder_path)
        if success:
            return jsonify({"msg": f"Folder '{folder_path}' removed from ignored list."}), 200
        else:
            return jsonify({"msg": f"Folder '{folder_path}' not found in ignored list."}), 404
    except Exception as e:
        logger.error(f"Error removing ignored folder via API: {e}", exc_info=True)
        return jsonify({"msg": "Failed to remove ignored folder.", "error": str(e)}), 500
@batch_processing_bp.route('/system/category-distribution', methods=['GET'])
@admin_or_superadmin_required
def get_category_distribution():
    """
    Returns the distribution of documents across categories.
    """
    try:
        # Fetch all documents, projecting only the 'categories' field
        docs = db.collection("document_metadata").select(["categories"]).stream()
        
        category_counts = Counter()
        uncategorized_count = 0
        
        for doc in docs:
            data = doc.to_dict()
            categories = data.get("categories", [])
            
            # Normalize to list if it's a string
            if isinstance(categories, str):
                categories = [categories]
            elif categories is None:
                categories = []
                
            if not categories:
                uncategorized_count += 1
                continue
                
            for cat in categories:
                if cat:
                    category_counts[cat] += 1
                else:
                    uncategorized_count += 1
        
        # Format for response
        distribution = []
        for cat, count in category_counts.items():
            distribution.append({"category": cat, "count": count})
            
        # Add uncategorized if any
        if uncategorized_count > 0:
            found = False
            for item in distribution:
                # Check for various casing of Uncategorized to merge
                if item["category"].upper() == "UNCATEGORIZED":
                    item["count"] += uncategorized_count
                    found = True
                    break
            if not found:
                distribution.append({"category": "UNCATEGORIZED", "count": uncategorized_count})
        
        # Sort by count desc
        distribution.sort(key=lambda x: x['count'], reverse=True)
        
        return jsonify(distribution), 200

    except Exception as e:
        logger.error(f"Error getting category distribution: {e}", exc_info=True)
        return jsonify({"msg": "Failed to get category distribution", "error": str(e)}), 500
