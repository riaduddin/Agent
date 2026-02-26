
import logging
import json
import concurrent.futures
from google.cloud import pubsub_v1
from app import config, db
from app.utils.redis_client import get_redis_client, QUEUE_NAME
from app.services import batch_processing_service
from google.cloud import firestore

logger = logging.getLogger(__name__)

# Topic ID for reprocessing triggers (can be shared or separate)
# For now, we will assume a generic "scheduler_trigger_topic" is used 
# where the payload differentiates the action.
# Or if user creates a specific topic, we can use that.
# Let's support a GENERIC scheduler topic that dispatches based on payload "action".

def process_scheduler_message(message_data):
    """
    Parses the message data from Cloud Scheduler (via Pub/Sub) and executes the task.
    Expected format: JSON '{"action": "reprocess_legacy", ...}'
    """
    try:
        if isinstance(message_data, bytes):
            message_data = message_data.decode('utf-8')
        
        logger.info(f"Scheduler Trigger Received: {message_data}")
        
        try:
            payload = json.loads(message_data)
        except json.JSONDecodeError:
            # Fallback: if plain text matches a known keyword
            payload = {"action": message_data.strip()}

        action = payload.get("action")

        if action == "reprocess_legacy":
            trigger_legacy_reprocessing()
        else:
            logger.warning(f"Unknown scheduler action received: {action}")

    except Exception as e:
        logger.error(f"Error processing scheduler message: {e}", exc_info=True)

def trigger_legacy_reprocessing():
    """Logic to queue the bulk legacy reprocess task."""
    try:
        if not batch_processing_service.get_legacy_reprocess_enabled():
            logger.info("Scheduler: Legacy reprocess is disabled by settings. Skipping trigger.")
            try:
                db.collection("system_logs").add({
                    "timestamp": firestore.SERVER_TIMESTAMP,
                    "document_id": "LEGACY_REPROCESS_SYSTEM",
                    "step_name": "LEGACY_REPROCESS_SKIPPED",
                    "details": {"reason": "disabled_by_settings", "source": "SCHEDULER_TRIGGER_SERVICE"},
                    "status": "INFO"
                })
            except Exception as log_err:
                logger.warning(f"Failed to log legacy reprocess skip: {log_err}")
            return

        redis_conn = get_redis_client()
        if not redis_conn:
            logger.error("Scheduler: Failed to connect to Redis for legacy reprocess task.")
            try:
                db.collection("system_logs").add({
                    "timestamp": firestore.SERVER_TIMESTAMP,
                    "document_id": "LEGACY_REPROCESS_SYSTEM",
                    "step_name": "LEGACY_REPROCESS_FAILED",
                    "details": {"error": "Redis connection failed", "source": "SCHEDULER_TRIGGER_SERVICE"},
                    "status": "ERROR"
                })
            except Exception as log_err:
                logger.warning(f"Failed to log redis connection error: {log_err}")
            return

        # Use a system identifier and pass trigger source
        system_user = "scheduler_system_job"
        task_str = f"initiate_bulk_reprocess_legacy::{system_user}::CLOUD_SCHEDULER"
        
        redis_conn.rpush(QUEUE_NAME, task_str)
        logger.info(f"Scheduler: Successfully queued bulk legacy reprocess task: {task_str}")
        
    except Exception as e:
        logger.error(f"Scheduler: Failed to queue legacy reprocess task: {e}", exc_info=True)
        try:
            db.collection("system_logs").add({
                "timestamp": firestore.SERVER_TIMESTAMP,
                "document_id": "LEGACY_REPROCESS_SYSTEM",
                "step_name": "LEGACY_REPROCESS_FAILED",
                "details": {"error": str(e), "source": "SCHEDULER_TRIGGER_SERVICE"},
                "status": "ERROR"
            })
        except Exception as log_err:
            logger.warning(f"Failed to log legacy reprocess error: {log_err}")


def message_callback(message):
    process_scheduler_message(message.data)
    message.ack()

def subscribe_to_scheduler_topic():
    """
    Subscribes to the generic scheduler topic defined in config.
    This can run alongside the batch processing trigger.
    """
    project_id = config.PROJECT_ID
    # We reuse the existing subscription config or look for a new one.
    # To avoid breaking existing batch flow, let's look for a SPECIFIC env var first,
    # or fallback to a convention.
    
    # Ideally, we should have a separate subscription for this new schedule.
    subscription_id = "legacy-reprocess-sub" # Default assumption
    # In a real scheduling setup, the user would create:
    # Topic: legacy-reprocess-topic
    # Subscription: legacy-reprocess-sub (push or pull)
    
    # Since we can't easily change env vars in a running Cloud Run service without re-deploy,
    # we might hardcode the subscription name OR reuse the existing pattern if flexible.
    
    # Let's assume the user will configure a NEW subscription for this specific task
    # OR we use a single "scheduler-dispatch" subscription for all scheduler jobs.
    
    # For this implementation, let's assume we want to listen to 'legacy-reprocess-sub'
    # but we need to know the TOPIC or SUBSCRIPTION name.
    
    # Only try to subscribe if explicit config exists or we default safely.
    target_subscription = "legacy-reprocess-sub" 
    
    try:
        subscriber = pubsub_v1.SubscriberClient()
        subscription_path = subscriber.subscription_path(project_id, target_subscription)
        
        logger.info(f"Scheduler Service: Listening on {subscription_path} for nightly tasks...")
        
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = subscriber.subscribe(subscription_path, callback=message_callback)
        
        try:
            future.result()
        except Exception as e:
            logger.error(f"Scheduler Service Subscription ended: {e}")
            future.cancel()
            
    except Exception as e:
        # It's possible the subscription doesn't exist yet. Log warning only.
        logger.warning(f"Scheduler Service: Could not subscribe to {target_subscription}. Ensure it exists in GCP. Error: {e}")

