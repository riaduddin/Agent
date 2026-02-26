# backend/app/routes/batch_routes.py
import logging
from flask import Blueprint, jsonify
from flask_jwt_extended import jwt_required
from threading import Thread
from app import db
from app.utils.redis_client import get_redis_client, QUEUE_NAME
from app.utils.utils import admin_or_superadmin_required
from app.utils.debug_logger import debug_log, debug_warn, debug_error

logger = logging.getLogger(__name__)
batch_bp = Blueprint('batch_bp', __name__)

def queue_categorization_tasks():
    """
    Queries for uncategorized documents and queues them for processing.
    This runs in a background thread.
    """
    debug_log("Starting queue_categorization_tasks thread.")
    logger.info("Starting background thread for categorization batch queuing.")
    
    try:
        redis_client = get_redis_client()
        debug_log("Redis client obtained in thread.")
    except Exception as e:
        debug_error(f"Failed to get Redis client in thread: {e}")
        logger.error("Could not connect to Redis in thread. Aborting batch categorization.")
        return

    try:
        debug_log("Setting categorization_batch_status to 'running'.")
        redis_client.set("categorization_batch_status", "running", ex=86400) # Lock for 24h
        
        debug_log("Streaming all documents from Firestore to find uncategorized ones.")
        docs_stream = db.collection("document_metadata").stream()
        count = 0
        total_checked = 0
        debug_log("Starting to loop through documents and queue tasks.")
        for doc in docs_stream:
            total_checked += 1
            doc_data = doc.to_dict()
            if "categoriesd" not in doc_data:
                task_string = f"backfill_category::{doc.id}"
                redis_client.rpush(QUEUE_NAME, task_string)
                count += 1
                if count % 100 == 0:
                    debug_log(f"Queued {count} documents...")
                    logger.info(f"Queued {count} documents for categorization...")
            if total_checked % 1000 == 0:
                 debug_log(f"Scanned {total_checked} total documents...")


        debug_log(f"Finished queuing loop. Scanned {total_checked} documents. Total queued: {count}.")
        logger.info(f"Finished queuing. Total documents queued: {count}.")

    except Exception as e:
        debug_error(f"An error occurred during batch queuing: {e}")
        logger.error(f"An error occurred during batch queuing: {e}", exc_info=True)
    finally:
        debug_log("Setting categorization_batch_status to 'idle'.")
        redis_client.set("categorization_batch_status", "idle", ex=3600) # Reset status
        debug_log("Finished queue_categorization_tasks thread.")


from redis.exceptions import ConnectionError as RedisConnectionError

@batch_bp.route('/system/start-categorization-batch', methods=['POST'])
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
