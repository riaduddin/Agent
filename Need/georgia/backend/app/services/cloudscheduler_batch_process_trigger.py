import logging
import json
from app.utils.redis_client import get_redis_client
from app.services import batch_processing_service
from app.services.doc_processing_helpers import bulk_processing_utils
from app import config, db
from google.cloud import pubsub_v1
from google.cloud import scheduler_v1
import google.api_core.exceptions
import concurrent.futures
from app.utils.debug_logger import debug_trigger, debug_connection, debug_error, debug_success, debug_banner, debug_warning
from google.cloud import firestore

logger = logging.getLogger(__name__)

# Redis Lock Configuration
BATCH_PROCESS_LOCK_KEY = "batch_process_lock"
LEGACY_PROCESS_LOCK_KEY = "legacy_process_lock"
LOCK_TIMEOUT_SECONDS = 3600  # 1 hour for larger jobs

def trigger_scheduled_job(job_type: str):
    """
    Orchestrates the execution of a scheduled job with Redis locking.
    Supported job_types: 'batch_run', 'reprocess_legacy'
    
    For 'reprocess_legacy': Skips legacy chunks aged >= config.LEGACY_CHUNK_MAX_AGE_DAYS days
    to optimize reprocessing of recent changes and reduce redundant processing.
    """
    source = "CLOUD SCHEDULER"
    debug_trigger("JOB_START", details=f"Initiating {job_type}", source=source)
    
    if job_type == "reprocess_legacy":
        enabled = batch_processing_service.get_legacy_reprocess_enabled()
        if not enabled:
            debug_warning("LEGACY_DISABLED", details="Nightly legacy reprocess is disabled by settings.", source=source)
            logger.info("Legacy reprocess skipped: disabled by superadmin settings.")
            try:
                db.collection("system_logs").add({
                    "timestamp": firestore.SERVER_TIMESTAMP,
                    "document_id": "LEGACY_REPROCESS_SYSTEM",
                    "step_name": "LEGACY_REPROCESS_SKIPPED",
                    "details": {"reason": "disabled_by_settings", "source": source},
                    "status": "INFO"
                })
            except Exception as log_err:
                logger.warning(f"Failed to log legacy reprocess skip: {log_err}")
            return

    redis_client = get_redis_client()
    if not redis_client:
        debug_error(f"[{source}] Redis unavailable. Cannot trigger {job_type}.")
        return

    # Select appropriate lock key based on job type
    lock_key = BATCH_PROCESS_LOCK_KEY if job_type == "batch_run" else LEGACY_PROCESS_LOCK_KEY
    
    # Try to acquire lock
    lock_acquired = redis_client.set(lock_key, "running", nx=True, ex=LOCK_TIMEOUT_SECONDS)

    if lock_acquired:
        debug_success("LOCK_ACQUIRED", details=f"Lock secured for {job_type}")
        try:
            if job_type == "batch_run":
                default_bucket = batch_processing_service.get_default_bucket()
                if not default_bucket:
                    debug_error(f"[{source}] No default bucket configured for batch run.")
                    return
                
                debug_trigger("RUNNING_BATCH", details=f"Target Bucket: {default_bucket}", source=source)
                run_id = batch_processing_service.start_run(default_bucket)
                debug_success("BATCH_STARTED", details=f"Run ID: {run_id}")
                
            elif job_type == "reprocess_legacy":
                system_user = "cloud_scheduler_system"
                debug_trigger("RUNNING_LEGACY", details="Scanning for legacy chunks...", source=source)
                try:
                    db.collection("system_logs").add({
                        "timestamp": firestore.SERVER_TIMESTAMP,
                        "document_id": "LEGACY_REPROCESS_SYSTEM",
                        "step_name": "LEGACY_REPROCESS_TRIGGERED",
                        "details": {
                            "source": source,
                            "skip_aged_chunks": True,
                            "max_age_days": config.LEGACY_CHUNK_MAX_AGE_DAYS
                        },
                        "status": "INFO"
                    })
                except Exception as log_err:
                    logger.warning(f"Failed to log legacy reprocess trigger: {log_err}")
                # This queues the orchestrator task in the worker
                # Skip chunks aged >= config.LEGACY_CHUNK_MAX_AGE_DAYS to optimize reprocessing
                bulk_processing_utils.start_legacy_reprocess_job(
                    system_user,
                    skip_aged_chunks=True,
                    max_age_days=config.LEGACY_CHUNK_MAX_AGE_DAYS,
                    trigger_source="CLOUD_SCHEDULER"
                )
                debug_success("LEGACY_JOB_QUEUED", details=f"Worker will now scan and reprocess (skipping chunks >= {config.LEGACY_CHUNK_MAX_AGE_DAYS} days old).")

        except Exception as e:
            debug_error(f"[{source}] Critical failure in {job_type}", error=str(e))
            logger.error(f"Scheduled job error: {e}", exc_info=True)
            # Log error to system_logs for failures
            if job_type == "reprocess_legacy":
                try:
                    db.collection("system_logs").add({
                        "timestamp": firestore.SERVER_TIMESTAMP,
                        "document_id": "LEGACY_REPROCESS_SYSTEM",
                        "step_name": "LEGACY_REPROCESS_FAILED",
                        "details": {
                            "error": str(e),
                            "source": source,
                            "skip_aged_chunks": True,
                            "max_age_days": config.LEGACY_CHUNK_MAX_AGE_DAYS
                        },
                        "status": "ERROR"
                    })
                except Exception as log_err:
                    logger.warning(f"Failed to log legacy reprocess error: {log_err}")
        finally:
            debug_trigger("CLEANUP", details=f"Releasing lock for {job_type}", source=source)
            redis_client.delete(lock_key)
    else:
        debug_trigger("SKIP", details=f"{job_type} already running on another pod.", source=source)

def message_callback(message):
    """
    Standardizes the reception of Pub/Sub messages.
    Supports JSON: {"action": "batch_run"} or {"action": "reprocess_legacy"}
    
    Note: 'reprocess_legacy' action will skip chunks aged >= config.LEGACY_CHUNK_MAX_AGE_DAYS
    during processing as configured in the constants above.
    """
    source = "PUB/SUB"
    try:
        data_str = message.data.decode('utf-8')
        debug_trigger("SIGNAL_RECEIVED", details=data_str, source=source)
        
        try:
            payload = json.loads(data_str)
            action = payload.get("action")
        except json.JSONDecodeError:
            # Fallback for plain text triggers
            action = data_str.strip().lower()

        if action in ["batch_run", "nightly_batch"]:
            trigger_scheduled_job("batch_run")
        elif action in ["reprocess_legacy", "upgrade_docs"]:
            trigger_scheduled_job("reprocess_legacy")
        else:
            debug_trigger("UNKNOWN_ACTION", details=f"No handler for '{action}'", source=source)
            
        message.ack()
        debug_success("SIGNAL_ACKNOWLEDGED")

    except Exception as e:
        debug_error(f"[{source}] Failed to process signal", error=str(e))
        # Nack so it can be retried if it was a transient parsing issue
        message.nack()

def subscribe_to_scheduler_services():
    """
    Initializes listeners for all configured scheduler topics.
    This provides professional startup logs and connection diagnostics.
    """
    project_id = config.PROJECT_ID
    
    # 1. Batch Processing Subscription
    sub_batch = config.PUBSUB_BATCH_SUBSCRIPTION_ID
    
    # 2. Legacy Reprocess Subscription
    sub_legacy = config.PUBSUB_LEGACY_SUBSCRIPTION_ID

    if not project_id:
        debug_error("[SYSTEM] PROJECT_ID missing. Scheduler services disabled.")
        return

    debug_banner("SCHEDULER SERVICE INITIALIZATION", {
        "Project ID": project_id,
        "Batch Topic": config.PUBSUB_BATCH_TOPIC_ID,
        "Batch Sub": sub_batch,
        "Legacy Topic": config.PUBSUB_LEGACY_TOPIC_ID,
        "Legacy Sub": sub_legacy
    })

    subscriber = pubsub_v1.SubscriberClient()
    subscriptions = []
    
    if sub_batch:
        path = subscriber.subscription_path(project_id, sub_batch)
        subscriptions.append(path)
        
    if sub_legacy and sub_legacy != sub_batch:
        path = subscriber.subscription_path(project_id, sub_legacy)
        subscriptions.append(path)

    if not subscriptions:
        debug_connection("Scheduler", "OFFLINE", "No subscriptions configured.")
        return

    for sub_path in subscriptions:
        try:
            subscriber.subscribe(sub_path, callback=message_callback)
            debug_connection("Pub/Sub", "CONNECTED", f"Listening on {sub_path}")
        except Exception as e:
            debug_error(f"[CONNECTION] Failed to connect to {sub_path}", error=str(e))

    debug_success("SCHEDULER_SERVICES", "Wait loop engaged. Listening for signals...")
    
    # Run resource validation in background to not block startup
    validate_infrastructure(project_id)


def check_resource_existence(resource_name, client_method, **kwargs):
    """
    Helper to safely check if a GCP resource exists.
    Returns: True if exists, False if not found, Exception if permission denied/error.
    """
    try:
        client_method(**kwargs)
        return True
    except google.api_core.exceptions.NotFound:
        return False
    except google.api_core.exceptions.PermissionDenied as e:
        return e  # Return exception to log details
    except Exception as e:
        return e


def validate_infrastructure(project_id):
    """
    Validates that configured Pub/Sub topics, subscriptions, and Scheduler Jobs actually exist.
    """
    debug_banner("INFRASTRUCTURE VALIDATION", {"Status": "Checking GCP Resources..."})

    # 1. Pub/Sub Validation
    pubsub_publisher = pubsub_v1.PublisherClient()
    pubsub_subscriber = pubsub_v1.SubscriberClient()

    # Define resources to check
    topics_to_check = [
        ("Batch Topic", config.PUBSUB_BATCH_TOPIC_ID),
        ("Legacy Topic", config.PUBSUB_LEGACY_TOPIC_ID)
    ]
    
    subs_to_check = [
        ("Batch Sub", config.PUBSUB_BATCH_SUBSCRIPTION_ID),
        ("Legacy Sub", config.PUBSUB_LEGACY_SUBSCRIPTION_ID)
    ]

    for label, topic_id in topics_to_check:
        if topic_id:
            path = pubsub_publisher.topic_path(project_id, topic_id)
            # Pass as keyword argument 'topic'
            exists = check_resource_existence(label, pubsub_publisher.get_topic, topic=path)
            
            if exists is True:
                debug_success(f"TOPIC: {topic_id}")
            elif exists is False:
                debug_warning(f"TOPIC: {topic_id}", f"NOT FOUND. Please create this topic in project {project_id}.")
            else:
                debug_error(f"topic_validation", f"Error checking {topic_id}: {exists}")

    for label, sub_id in subs_to_check:
        if sub_id:
            path = pubsub_subscriber.subscription_path(project_id, sub_id)
            # Pass as keyword argument 'subscription'
            exists = check_resource_existence(label, pubsub_subscriber.get_subscription, subscription=path)
            
            if exists is True:
                debug_success(f"SUBSCRIPTION: {sub_id}")
            elif exists is False:
                debug_warning(f"SUBSCRIPTION: {sub_id}", f"NOT FOUND. Please create this subscription.")
            else:
                debug_error(f"sub_validation", f"Error checking {sub_id}: {exists}")

    # 2. Cloud Scheduler Job Validation (Optional)
    location = config.SCHEDULER_LOCATION
    jobs_to_check = [
        ("Batch Job", config.SCHEDULER_JOB_BATCH_RUN),
        ("Legacy Job", config.SCHEDULER_JOB_LEGACY_REPROCESS)
    ]
    
    # Only initialize if at least one job is configured
    if any(job_name for _, job_name in jobs_to_check):
        try:
            scheduler_client = scheduler_v1.CloudSchedulerClient()
            
            for label, job_name in jobs_to_check:
                if job_name:
                    job_path = scheduler_client.job_path(project_id, location, job_name)
                    # Pass job_path as keyword argument 'name'
                    exists = check_resource_existence(label, scheduler_client.get_job, name=job_path)
                    
                    if exists is True:
                        debug_success(f"SCHEDULER_JOB: {job_name}")
                    elif exists is False:
                        debug_warning(f"SCHEDULER_JOB: {job_name}", f"NOT FOUND in {location}. Ensure the Job ID matches.")
                    else:
                        debug_error(f"job_validation", f"Error checking {job_name}: {exists}")
        except Exception as e:
            debug_warning("SCHEDULER_CLIENT", f"Could not initialize CloudSchedulerClient: {e}")
    else:
        debug_warning("SCHEDULER_JOB", f"Validation skipped. Configure SCHEDULER_JOB_* in env to enable checks.")
