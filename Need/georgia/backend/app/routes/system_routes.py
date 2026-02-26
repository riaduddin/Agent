# backend/app/routes/system_routes.py
import logging # Add logging import
from flask import Blueprint, jsonify, current_app, request
from flask_jwt_extended import jwt_required, get_jwt_identity # Import get_jwt_identity
from google.cloud import storage, firestore, documentai
from google.api_core.exceptions import NotFound, GoogleAPICallError, AlreadyExists, PermissionDenied
import google.auth # For getting credentials for Admin API
from googleapiclient.discovery import build # For Admin API client
from googleapiclient.errors import HttpError # Import HttpError
import os
import json
from flask_cors import CORS # Import CORS

# Import specific model functions needed
from app.models.metadata_model import get_all_chunk_ids
from app.models.log_model import add_log_entry # Import log function
# Import services
from app.services import vertex_ai_service 
from app.services.doc_processing_helpers import bulk_processing_utils 
from app import config # Import config
from app import db # Import db instance
from app.utils.redis_client import get_redis_client, QUEUE_NAME # Import redis client getter and QUEUE_NAME
from app.utils.debug_logger import debug_log, debug_warn, debug_error

# Initialize logger
logger = logging.getLogger(__name__)

# Remove url_prefix from blueprint definition, will be added during registration
system_bp = Blueprint('system_bp', __name__)

# Removed blueprint-level CORS configuration. Relying on global CORS in __init__.py


# --- Helper to check configuration values (from config.py) ---
def check_config_vars():
    results = {}
    # Check variables loaded in config.py
    config_vars_to_check = {
        'PROJECT_ID': config.PROJECT_ID,
        'BUCKET_NAME': config.BUCKET_NAME,
        'VERTEX_LOCATION': config.VERTEX_LOCATION,
        'DOCAI_LOCATION': config.DOCAI_LOCATION,
        'DOCUMENT_API_PROCESSOR_ID': config.DOCUMENT_API_PROCESSOR_ID,
        'VECTOR_INDEX_NAME': config.VECTOR_INDEX_NAME, # Keep checking if set
        'VECTOR_INDEX_ENDPOINT_ID': config.VECTOR_INDEX_ENDPOINT_ID, # Keep checking if set
        'VECTOR_DEPLOYED_INDEX_ID': config.VECTOR_DEPLOYED_INDEX_ID, # Keep checking if set
        'JWT_SECRET_KEY': config.JWT_SECRET_KEY,
        'GOOGLE_APPLICATION_CREDENTIALS': config.GOOGLE_APPLICATION_CREDENTIALS,
        'FIRESTORE_DATABASE_ID': config.FIRESTORE_DATABASE_ID # Added Firestore DB ID check
    }
    for name, value in config_vars_to_check.items():
        if value:
            display_value = '******' if 'KEY' in name or 'SECRET' in name else value
            # Check credentials path specifically
            if name == 'GOOGLE_APPLICATION_CREDENTIALS' and not os.path.exists(value):
                 results[name] = {"status": "Error", "detail": f"File not found at path: {value}"}
            else:
                 results[name] = {"status": "OK", "value": display_value}
        else:
            # Only mark as error if it's not an optional Vector Search var
            if 'VECTOR_' not in name:
                 results[name] = {"status": "Error", "detail": "Not loaded or missing in config"}
            else:
                 results[name] = {"status": "Info", "detail": "Not set (Optional - can be created via Setup)"}
    return results

# --- Helper to check GCS Bucket (using config) ---
def check_gcs_bucket():
    # Config vars already validated in config.py
    try:
        # Use project_id from config
        storage_client = storage.Client(project=config.PROJECT_ID)
        bucket = storage_client.get_bucket(config.BUCKET_NAME)
        return {"status": "OK", "detail": f"Bucket '{config.BUCKET_NAME}' accessible"}
    except NotFound:
        return {"status": "Error", "detail": f"Bucket '{config.BUCKET_NAME}' not found or no access."}
    except Exception as e:
        return {"status": "Error", "detail": f"GCS connection error: {e}"}

# --- Helper to check Firestore (using imported db instance) ---
def check_firestore():
    """Checks connection to the configured Firestore database."""
    db_id_to_check = config.FIRESTORE_DATABASE_ID if config.FIRESTORE_DATABASE_ID != '(default)' else 'default'
    try:
        # db instance is initialized in app.__init__ with the correct database ID
        # Attempt to list collections as a basic connectivity test
        collections = list(db.collections()) # Ensure limit=1 is removed
        return {"status": "OK", "detail": f"Firestore connection successful for database '{db_id_to_check}' in project {config.PROJECT_ID}"}
    except NotFound as e:
        # This might indicate the database itself doesn't exist or has access issues
         # Check if the error message specifically mentions the database ID
        if f"database projects/{config.PROJECT_ID}/databases/{config.FIRESTORE_DATABASE_ID}" in str(e).lower():
             error_detail = f"Database '{config.FIRESTORE_DATABASE_ID}' not found or inaccessible in project {config.PROJECT_ID}. It must be created manually (e.g., via GCP Console or 'gcloud firestore databases create --database={config.FIRESTORE_DATABASE_ID}')."
             return {"status": "Error", "detail": error_detail, "needs_manual_creation": True} # Add flag
        else:
             # Different NotFound error
             error_detail = f"Firestore resource not found: {e}"
             return {"status": "Error", "detail": error_detail}
    except Exception as e:
        # Capture other potential errors (permissions, network, etc.)
        error_detail = f"Firestore connection error: {e}"
        # Check if it's a permission error related to the specific database
        if "permission denied" in str(e).lower() and config.FIRESTORE_DATABASE_ID in str(e).lower():
             error_detail += f" Ensure the service account has permissions for database '{config.FIRESTORE_DATABASE_ID}'."

        return {"status": "Error", "detail": error_detail}


# --- Helper function to create Firestore Database (using Admin API) ---
# Rewritten with consistent 4-space indentation
def _create_firestore_database_if_not_exists(project_id, database_id, location_id):
    """Uses the Admin API to create the Firestore database if it doesn't exist."""
    if database_id == '(default)':
        return {"status": "Info", "message": "Using the default Firestore database. No creation needed via this route."}

    try:
        credentials, effective_project_id = google.auth.default(
            scopes=['https://www.googleapis.com/auth/datastore', 'https://www.googleapis.com/auth/cloud-platform']
        )
        if not effective_project_id: effective_project_id = project_id
        if not effective_project_id:
            return {"status": "Error", "message": "Could not determine Google Cloud project ID for Admin API."}

        firestore_admin = build('firestore', 'v1', credentials=credentials, cache_discovery=False)
        parent = f"projects/{effective_project_id}"
        db_name = f"{parent}/databases/{database_id}"

        proceed_to_create = False
        # 1. Check if database exists
        try:
            debug_log(f"Checking existence of Firestore database: {db_name}")
            firestore_admin.projects().databases().get(name=db_name).execute()
            debug_log(f"Firestore database '{database_id}' already exists.")
            return {"status": "Exists", "message": f"Firestore database '{database_id}' already exists in project '{effective_project_id}'."}
        except HttpError as e:
            if e.resp.status == 404:
                debug_log(f"Firestore database '{database_id}' not found (404). Will attempt creation.")
                proceed_to_create = True
            else:
                debug_error(f"HTTP error during database check: {e}")
                return {"status": "Error", "message": f"HTTP error checking Firestore database '{database_id}'.", "error_details": str(e)}
        except Exception as e:
            debug_error(f"Unexpected error during database check: {e}")
            return {"status": "Error", "message": f"Unexpected error checking Firestore database '{database_id}'.", "error_details": str(e)}

        # 2. Create the database ONLY if the check indicated it was needed
        if proceed_to_create:
            try:
                create_body = {
                    'locationId': location_id,
                    'type': 'FIRESTORE_NATIVE',
                }
                request_obj = firestore_admin.projects().databases().create(
                    parent=parent,
                    databaseId=database_id,
                    body=create_body
                )
                operation = request_obj.execute()
                debug_log(f"Database creation operation started for {database_id}: {operation.get('name')}")
                return {"status": "Initiated", "message": f"Firestore database '{database_id}' creation initiated in location '{location_id}'. Operation: {operation.get('name')}. It may take a few minutes to become available."}
            except AlreadyExists:
                debug_log(f"Database {database_id} already exists (detected during create).")
                return {"status": "Exists", "message": f"Firestore database '{database_id}' already exists."}
            except PermissionDenied as e:
                debug_error(f"Permission denied creating database {database_id}: {e}")
                return {"status": "Error", "message": f"Permission denied creating Firestore database '{database_id}'. Ensure service account has 'Cloud Datastore Owner' or 'Editor' role on the project.", "error_details": str(e)}
            except GoogleAPICallError as e:
                debug_error(f"API error creating database {database_id}: {e}")
                error_detail = str(e)
                try:
                    error_info = json.loads(e.content).get('error', {})
                    error_detail = error_info.get('message', str(e))
                except: pass
                return {"status": "Error", "message": f"API error creating Firestore database '{database_id}'.", "error_details": error_detail}
            except Exception as e:
                debug_error(f"Unexpected error creating database {database_id}: {e}")
                return {"status": "Error", "message": f"Unexpected error creating Firestore database '{database_id}'.", "error_details": str(e)}
        else:
            # Should not be reached if logic is correct
            debug_warn(f"Database check completed but proceed_to_create flag is false. Database '{database_id}' status uncertain.")
            return {"status": "Error", "message": f"Internal logic error during database check for '{database_id}'. Status uncertain."}

    except Exception as e:
        debug_error(f"Failed to get default credentials for Admin API: {e}")
        return {"status": "Error", "message": f"Failed to get Google Cloud credentials for Admin API: {str(e)}"}


# --- Helper to check Document AI Processor (using config) ---
def check_docai_processor():
    # Config vars already validated in config.py
    try:
        # Re-initializing client for isolation, using config values
        from google.api_core.client_options import ClientOptions
        docai_client = documentai.DocumentProcessorServiceClient(
            client_options=ClientOptions(api_endpoint=f"{config.DOCAI_LOCATION}-documentai.googleapis.com")
        )
        processor_name = docai_client.processor_path(config.PROJECT_ID, config.DOCAI_LOCATION, config.DOCUMENT_API_PROCESSOR_ID)
        processor_info = docai_client.get_processor(name=processor_name)
        return {"status": "OK", "detail": f"Processor '{config.DOCUMENT_API_PROCESSOR_ID}' found (Type: {processor_info.type_})"}
    except NotFound:
         return {"status": "Error", "detail": f"Processor '{config.DOCUMENT_API_PROCESSOR_ID}' not found in location '{config.DOCAI_LOCATION}'"}
    except GoogleAPICallError as e:
         return {"status": "Error", "detail": f"DocAI API error: {e.message}"}
    except Exception as e:
        return {"status": "Error", "detail": f"DocAI connection error: {e}"}

# --- Helper to check Gemini Model ---
def check_gemini_model():
    """Checks if the Gemini API Key client is available and functional."""
    from app.llm.gemini_api_key_client import gemini_client
    if gemini_client.is_available():
        return {"status": "OK", "detail": f"Gemini API Key client initialized and available (Model: {config.GEMINI_MODEL_NAME})"}
    else:
        return {"status": "Error", "detail": f"Gemini API Key missing or client failed to initialize. Gemini features will be disabled."}


@system_bp.route('/diagnosis', methods=['GET'])
@jwt_required() # Ensure user is logged in
def run_diagnosis():
    """Runs system configuration and connectivity checks."""
    # Run the check function from vertex_ai_service
    vector_search_status = vertex_ai_service.check_vector_search_availability()

    diagnosis_results = {
        "Configuration Variables": check_config_vars(),
        "Google Cloud Storage": check_gcs_bucket(),
        "Firestore": check_firestore(),
        "Document AI": check_docai_processor(),
        "Vertex AI Vector Search": vector_search_status, # Use result from service
        "Vertex AI Gemini": check_gemini_model(),
    }
    return jsonify(diagnosis_results), 200

@system_bp.route('/create-indexes', methods=['POST', 'OPTIONS']) # Keep OPTIONS for CORS preflight
@jwt_required(optional=True) # Allow OPTIONS request to proceed, POST will require valid token implicitly
def create_firestore_indexes():
    """
    Attempts to create Firestore composite indexes programmatically using the Admin API.
    """
    # Handle OPTIONS preflight request (Flask-CORS should handle this)
    if request.method == 'OPTIONS':
        return jsonify({"status": "OK"}), 200

    # If we reach here for a POST request, @jwt_required(optional=True) allowed it.
    # We rely on the frontend sending a valid token for POST requests.
    # A missing/invalid token would result in an error handled by Flask-JWT-Extended earlier.

    # --- Proceed with POST logic ---

    # Construct the path to firestore.indexes.json relative to the app's root
    backend_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    index_file_path = os.path.join(backend_root, 'firestore.indexes.json')

    if not os.path.exists(index_file_path):
        return jsonify({"status": "Error", "message": f"Index file not found at {index_file_path}"}), 404

    try:
        with open(index_file_path, 'r') as f:
            index_config = json.load(f)
    except Exception as e:
        return jsonify({"status": "Error", "message": f"Failed to read or parse index file: {str(e)}"}), 500

    indexes_to_create = index_config.get("indexes", [])
    if not indexes_to_create:
        return jsonify({"status": "Info", "message": "No composite indexes defined in firestore.indexes.json."}), 200

    # Get credentials (use the same logic as in __init__.py or rely on ADC)
    try:
        credentials, project_id = google.auth.default(
            scopes=['https://www.googleapis.com/auth/datastore', 'https://www.googleapis.com/auth/cloud-platform']
        )
        # Ensure project_id matches config if ADC is used
        if not project_id: project_id = config.PROJECT_ID
        if not project_id:
             return jsonify({"status": "Error", "message": "Could not determine Google Cloud project ID."}), 500
    except Exception as e:
         debug_error(f"Failed to get default credentials: {e}")
         return jsonify({"status": "Error", "message": f"Failed to get Google Cloud credentials: {str(e)}"}), 500


    # Build the Firestore Admin API client
    try:
        # Use version v1 for index management
        firestore_admin = build('firestore', 'v1', credentials=credentials, cache_discovery=False)
        # Use the configured database ID from config
        parent = f"projects/{project_id}/databases/{config.FIRESTORE_DATABASE_ID}/collectionGroups"
        debug_log(f"Using Firestore Admin API parent path: {parent}")
    except Exception as e:
        debug_error(f"Failed to build Firestore Admin API client: {e}")
        return jsonify({"status": "Error", "message": f"Failed to build Firestore Admin API client: {str(e)}"}), 500

    results = []
    has_errors = False

    for index_def in indexes_to_create:
        collection_id = index_def.get("collectionGroup")
        if not collection_id:
            results.append({"index": "Unknown", "status": "Skipped", "detail": "Missing 'collectionGroup' in definition."})
            has_errors = True
            continue

        # Construct the body for the API request
        # Map fields from JSON to API format
        api_fields = []
        for field in index_def.get("fields", []):
            api_field = {}
            if "fieldPath" in field: api_field["fieldPath"] = field["fieldPath"]
            if "order" in field: api_field["order"] = field["order"]
            # Add support for arrayConfig if needed later
            # if "arrayConfig" in field: api_field["arrayConfig"] = field["arrayConfig"]
            if api_field: api_fields.append(api_field)

        if not api_fields:
             results.append({"index": collection_id, "status": "Skipped", "detail": "No valid 'fields' defined for index."})
             has_errors = True
             continue

        index_body = {
            "queryScope": index_def.get("queryScope", "COLLECTION"), # Default to COLLECTION
            "fields": api_fields
        }

        index_name_str = f"{collection_id} ({', '.join([f['fieldPath']+' '+f.get('order','ASC') for f in api_fields])})" # For logging/reporting

        try:
            debug_log(f"Attempting to create index: {index_name_str}")
            request_obj = firestore_admin.projects().databases().collectionGroups().indexes().create(
                parent=f"{parent}/{collection_id}",
                body=index_body
            )
            # This returns a long-running operation object
            operation = request_obj.execute()
            debug_log(f"Index creation operation started for {index_name_str}: {operation.get('name')}")
            # Note: Index creation is asynchronous. We report initiation here.
            # Polling the operation status is possible but complex for a simple request.
            results.append({"index": index_name_str, "status": "Initiated", "detail": f"Operation: {operation.get('name')}"})

        except AlreadyExists:
            debug_log(f"Index already exists: {index_name_str}")
            results.append({"index": index_name_str, "status": "Exists", "detail": "Index already exists."})
        except GoogleAPICallError as e:
            debug_error(f"API error creating index {index_name_str}: {e}")
            # Attempt to parse the error message for more specific info if possible
            error_detail = str(e)
            try:
                 # Google API errors often have structured details
                 error_info = json.loads(e.content).get('error', {})
                 error_detail = error_info.get('message', str(e))
                 if 'details' in error_info:
                      error_detail += f" Details: {json.dumps(error_info['details'])}"
            except:
                 pass # Keep original error string if parsing fails
            results.append({"index": index_name_str, "status": "Error", "detail": error_detail})
            has_errors = True
        except Exception as e:
            debug_error(f"Unexpected error creating index {index_name_str}: {e}")
            results.append({"index": index_name_str, "status": "Error", "detail": f"Unexpected error: {str(e)}"})
            has_errors = True

    final_status_code = 207 if has_errors else 200 # Multi-Status if errors/skips occurred
    return jsonify({
        "status": "Completed" if not has_errors else "Completed with Errors/Skips",
        "message": "Firestore index creation process finished. Check details.",
        "results": results
        }), final_status_code

@system_bp.route('/setup-vector-search', methods=['POST', 'OPTIONS'])
@jwt_required(optional=True) # Allow OPTIONS, check POST manually
def setup_vector_search():
    """
    Attempts to get or create Vector Search Index Endpoint and Index,
    and deploy the index to the endpoint.
    """
    # Handle OPTIONS preflight request (Flask-CORS should handle this)
    if request.method == 'OPTIONS':
        return jsonify({"status": "OK"}), 200

    # If we reach here for a POST request, @jwt_required(optional=True) allowed it.
    # We rely on the frontend sending a valid token for POST requests.
    # A missing/invalid token would result in an error handled by Flask-JWT-Extended earlier.
    # No need for manual get_jwt_identity() check here for this setup endpoint.

    try:
        results = vertex_ai_service.setup_vector_search_resources()
        status_code = 500 if results.get("error") else 200
        return jsonify(results), status_code
    except Exception as e:
        debug_error(f"Unexpected error during Vector Search setup: {e}")
        return jsonify({"status": "Error", "message": f"An unexpected error occurred: {str(e)}"}), 500


@system_bp.route('/setup-firestore-database', methods=['POST']) # Removed 'OPTIONS' from methods
@jwt_required(optional=True) # Re-added JWT protection (optional=True allows CORS preflight)
def setup_firestore_database():
    """
    Checks if the configured Firestore database exists and attempts to create it
    using the Admin API if it doesn't. Requires locationId in request body.
    Relies on global Flask-CORS config to handle OPTIONS preflight requests.
    """
    # Removed manual OPTIONS check: if request.method == 'OPTIONS': return jsonify({"status": "OK"}), 200

    # POST request logic starts here. JWT is checked by the decorator.

    # Get locationId from request body - REQUIRED for creation
    location_id = request.json.get('locationId') if request.is_json else None
    if not location_id:
        return jsonify({"status": "Error", "message": "Missing 'locationId' (e.g., 'us-central1') in request body."}), 400

    # Get project and database ID from config
    project_id = config.PROJECT_ID
    database_id = config.FIRESTORE_DATABASE_ID

    if not project_id or not database_id:
         return jsonify({"status": "Error", "message": "PROJECT_ID or FIRESTORE_DATABASE_ID not configured in backend."}), 500

    # Call the helper function
    result = _create_firestore_database_if_not_exists(project_id, database_id, location_id)

    status_code = 500 if result.get("status") == "Error" else 409 if result.get("status") == "Exists" else 202 if result.get("status") == "Initiated" else 200

    return jsonify(result), status_code


@system_bp.route('/start-bulk-process', methods=['POST'])
@jwt_required() # Ensure user is logged in (consider adding admin role check later)
def start_bulk_processing():
    """
    Initiates the bulk processing workflow by scanning GCS and enqueueing tasks.
    Accepts an optional 'gcs_prefix' in the JSON body.
    """
    # Optional: Add role-based access control here if needed
    # current_user = get_jwt_identity() # Example: Get user identity if needed for logging/checks

    # Get prefix from request body, default to None if not provided or empty
    gcs_prefix_from_request = request.json.get('gcs_prefix') if request.is_json else None
    # Treat empty string from request as None, so config default is used
    gcs_prefix_override = gcs_prefix_from_request if gcs_prefix_from_request else None

    debug_log(f"Received request to start bulk processing. API prefix override: '{gcs_prefix_override}'")

    try:
        
        # Call the function from the correct module
        bulk_processing_utils.initiate_bulk_processing(gcs_prefix_override=gcs_prefix_override)
        
        # The message should reflect the prefix actually used (which might be from config)
        final_prefix = gcs_prefix_override if gcs_prefix_override is not None else config.GCS_BULK_PROCESSING_PREFIX
        return jsonify({"status": "Success", "message": f"Bulk processing initiated for prefix '{final_prefix}'. Tasks are being enqueued."}), 202 # Accepted
    except Exception as e:
        debug_error(f"Failed to initiate bulk processing: {e}")
        # Log the exception traceback for detailed debugging
        current_app.logger.error(f"Bulk processing initiation failed: {e}", exc_info=True)
        return jsonify({"status": "Error", "message": f"Failed to initiate bulk processing: {str(e)}"}), 500

@system_bp.route('/bulk-process-stats', methods=['GET'])
# @jwt_required(optional=True) # Temporarily removed decorator

def get_bulk_process_stats():
    """
    Retrieves statistics for the latest or currently running bulk processing run.
    (NOTE: JWT protection temporarily removed for debugging CORS preflight issues)
    """
    # Removed explicit OPTIONS handling.
    # NOTE: JWT protection is temporarily removed above.
    try:
        debug_log("into the get bulk process stats")
        # Query Firestore for the latest run document, ordered by start_time descending
        def get_count(query):
            """Helper function to safely extract count values from Firestore aggregation queries."""
            result = query.get()
            #print(f"Raw count result:result")

            # Check if result is a valid structure (list of lists containing Aggregation objects)
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], list) and len(result[0]) > 0:
                    # Ensure value is being accessed from Aggregation object
                    aggregation_value = result[0][0].value
                    return aggregation_value if aggregation_value is not None else 0
            return 0  # Default to 0 if structure is unexpected
        completed = get_count(db.collection("document_metadata").where(
                "source", "==", "gcs_bulk"
            ).where("status", "==", "completed").count())
        
        stats_ref = db.collection("bulk_process_runs").order_by(
            "start_time", direction=firestore.Query.DESCENDING
        ).limit(1)
        docs = list(stats_ref.stream()) # Use list() to execute the stream

        if not docs:
            return jsonify({"status": "No runs found", "message": "No bulk processing runs have been initiated yet."}), 404
        
        
        #print("completed count: ",completed)
        latest_doc = docs[0]
        latest_run_data = docs[0].to_dict()
        #print("latest run data: ", latest_run_data)
        update_payload={"completed": completed}
        if completed == latest_run_data.get("files_found"):
            update_payload["status"] = "completed"
        latest_doc.reference.update(update_payload)
        # Convert datetime objects to ISO strings for JSON serialization
        if 'start_time' in latest_run_data and latest_run_data['start_time']:
            latest_run_data['start_time'] = latest_run_data['start_time'].isoformat()
        if 'end_time' in latest_run_data and latest_run_data['end_time']:
            latest_run_data['end_time'] = latest_run_data['end_time'].isoformat()
        latest_run_data.update(update_payload)
        # print("into the get bulk process stats")
        return jsonify(latest_run_data), 200

    except Exception as e:
        current_app.logger.error(f"Failed to retrieve bulk process stats: {e}", exc_info=True)
        return jsonify({"status": "Error", "message": f"Failed to retrieve bulk process statistics: {str(e)}"}), 500


@system_bp.route('/processing-dashboard-stats', methods=['GET'])
@jwt_required(optional=True) # Re-added JWT protection as optional
def get_processing_dashboard_stats():
    """
    Retrieves aggregated counts of documents by their processing status.
    """
    try:
        # Define the four main display categories for the dashboard
        main_categories = ["Pending", "Processing", "Completed", "Failed", "Other"] 
        status_counts = {category: 0 for category in main_categories}
        total_docs = 0

        # Granular to broad category mapping
        granular_to_main_map = {
            "queued_for_splitting": "Pending",
            "pending": "Pending",
            "splitting": "Processing", 
            "splitting_in_progress": "Processing",
            "pending_chunk_processing": "Processing", 
            "ocr_pending": "Processing", 
            "ocr_in_progress": "Processing", 
            "pending_vectorization": "Processing", 
            "vectorizing": "Processing", 
            "processing": "Processing", 
            "completed": "Completed",
            "error": "Failed", 
            "error_splitting": "Failed",
            "error_creating_chunks": "Failed",
            "incomplete": "Failed",
            "error_worker_failure": "Failed",
            "ocr_failed": "Failed",
            "embedding_failed": "Failed",
            "vectorization_failed": "Failed",
            "upload_failed": "Failed",
            "processing_error": "Failed",
            "unknown": "Failed", 
        }

        docs_stream = db.collection("document_metadata").stream()

        for doc in docs_stream:
            total_docs += 1
            doc_data = doc.to_dict()
            granular_status = doc_data.get("status")
            
            mapped_category = None
            if granular_status and isinstance(granular_status, str):
                clean_status = granular_status.strip()
                mapped_category = granular_to_main_map.get(clean_status)
                
                if not mapped_category: # Fallback for unmapped granular statuses
                    if "error" in clean_status.lower() or "fail" in clean_status.lower():
                        mapped_category = "Failed"
                    elif "pending" in clean_status.lower() or "queued" in clean_status.lower():
                        mapped_category = "Pending"
                    elif clean_status == "completed": # Should be caught by map
                        mapped_category = "Completed"
                    else: # Default other active-like or truly unknown to "Processing" or "Other"
                        mapped_category = "Processing" 
                        logger.warning(f"Document {doc.id} has unmapped status '{clean_status}', categorizing as '{mapped_category}'.")
            
            if mapped_category and mapped_category in status_counts:
                status_counts[mapped_category] += 1
            else: # Handles None status or if mapped_category became None (e.g. invalid status type)
                 status_counts["Other"] += 1 
                 logger.warning(f"Document {doc.id} has null, empty, or unclassifiable status '{granular_status}', categorizing as 'Other'.")
        
        # Remove "Other" if its count is 0 for a cleaner response
        if status_counts.get("Other") == 0:
            status_counts.pop("Other", None)

        # Count total chunks (can be slow on very large collections)
        # Consider maintaining a counter document for better performance if needed.
        try:
            # Using count_matching_results which is generally more efficient than streaming all
            chunk_query = db.collection("document_chunks")
            total_chunks_agg = chunk_query.count() # Use aggregation query
            total_chunks = total_chunks_agg.get()[0][0].value
            logger.info(f"Retrieved total chunk count: {total_chunks}")
        except Exception as chunk_e:
            logger.error(f"Failed to count document chunks: {chunk_e}", exc_info=True)
            total_chunks = -1 # Indicate error fetching count

        dashboard_stats = {
            "total_documents": total_docs,
            "total_chunks": total_chunks, # Add chunk count
            "status_counts": status_counts
        }
        return jsonify(dashboard_stats), 200

    except Exception as e:
        logger.error(f"Failed to retrieve processing dashboard stats: {e}", exc_info=True)
        return jsonify({"status": "Error", "message": f"Failed to retrieve processing dashboard statistics: {str(e)}"}), 500


@system_bp.route('/process-pending', methods=['POST'])
@jwt_required() # Require authentication
def process_pending_documents():
    """
    Finds all documents with status 'pending' in Firestore and enqueues
    the initial 'split' task for them via the document processing service.
    """
    try:
        # Call from the correct module
        enqueued_count, error_count = bulk_processing_utils.enqueue_pending_documents()
        return jsonify({
            "status": "Success",
            "message": f"Initiated processing for pending documents. Enqueued: {enqueued_count}, Errors/Skipped: {error_count}."
        }), 202 # Accepted
    except Exception as e:
        current_app.logger.error(f"Error initiating processing for pending documents: {e}", exc_info=True)
        return jsonify({"status": "Error", "message": f"Failed to initiate processing for pending documents: {str(e)}"}), 500


@system_bp.route('/reset-stuck-documents', methods=['POST'])
@jwt_required() # Require authentication
def reset_stuck_documents():
    """
    Finds documents stuck in intermediate or error states (not 'pending' or 'completed')
    and resets their status to 'pending'.
    """
    logger.info("Received request to reset stuck documents.")
    reset_count = 0
    error_count = 0
    # Define terminal/initial states that should NOT be reset
    non_stuck_statuses = ["pending", "completed"]

    try:
        # Query for documents NOT in 'pending' or 'completed' state.
        # Firestore doesn't directly support complex 'NOT IN' queries efficiently across many documents.
        # A more scalable approach might involve querying for each intermediate/error state individually,
        # but for simplicity now, we'll iterate (might be slow for very large datasets).
        # Alternative: Query all, filter in code (less efficient). Let's try querying specific states.

        stuck_statuses_to_query = [
            "processing", # Generic processing state
            "splitting", "splitting_in_progress", # Covers both specific and generic splitting
            "pending_chunk_processing", # New parent state after splitting
            "ocr_pending", "ocr_in_progress", 
            "pending_vectorization", "vectorizing", 
            "error", # Generic error
            "error_splitting", # New specific error
            "error_creating_chunks", # New specific error
            "incomplete", # New state for partial success/failure
            "error_worker_failure", # New specific error
            "ocr_failed", "embedding_failed", "vectorization_failed", 
            "upload_failed", "processing_error", 
            "unknown"
        ]
        # Remove duplicates just in case by converting to set and back to list
        stuck_statuses_to_query = list(set(stuck_statuses_to_query))

        docs_to_reset_data = []
        for status in stuck_statuses_to_query:
            query = db.collection("document_metadata").where("status", "==", status)
            docs_stream = query.stream()
            for doc in docs_stream:
                 doc_data = doc.to_dict()
                 doc_data['id'] = doc.id # Add ID to data for easier access
                 docs_to_reset_data.append(doc_data)

        logger.info(f"Found {len(docs_to_reset_data)} documents in stuck states to potentially reset.")

        for doc_data in docs_to_reset_data:
            doc_id = doc_data['id']
            gcs_uri = doc_data.get('gcs_uri')

            if not gcs_uri:
                logger.warning(f"Document {doc_id} missing GCS URI, cannot delete chunks/embeddings. Skipping cleanup for this doc.")
                add_log_entry("WARNING", f"Doc {doc_id} missing GCS URI for reset cleanup.", step="reset_stuck_docs_skip_cleanup", worker_id="SYSTEM_API", details={"doc_id": doc_id})
                error_count += 1
                continue # Skip cleanup for this doc, but still try to reset status

            # 1. Delete existing chunks and embeddings
            logger.info(f"Attempting to delete chunks and embeddings for doc {doc_id} before reset.")
            delete_result = bulk_processing_utils.delete_all_chunks_for_parent(doc_id)
            if delete_result.get("status") == "success":
                logger.info(f"Successfully deleted chunks and initiated vector removal for doc {doc_id}.")
                add_log_entry("INFO", f"Chunks and embeddings deleted for doc {doc_id} during reset.", step="reset_stuck_docs_cleanup_success", worker_id="SYSTEM_API", details={"doc_id": doc_id, "delete_result": delete_result})
            else:
                logger.error(f"Failed to delete chunks/embeddings for doc {doc_id} during reset: {delete_result.get('error_message', 'Unknown error')}")
                add_log_entry("ERROR", f"Failed to delete chunks/embeddings for doc {doc_id} during reset.", step="reset_stuck_docs_cleanup_failed", worker_id="SYSTEM_API", details={"doc_id": doc_id, "delete_result": delete_result})
                error_count += 1
                # Decide whether to continue resetting status if cleanup failed. For now, we proceed.

            # 2. Reset parent document status
            doc_ref = db.collection("document_metadata").document(doc_id)
            doc_ref.update({
                "status": "pending",
                "status_message": "Reset to pending by user action.",
                "error_message": firestore.DELETE_FIELD, # Clear any previous error message
                "total_chunks": 0, # Reset chunk counts
                "completed_chunks": 0,
                "last_status_update": firestore.SERVER_TIMESTAMP
            })
            reset_count += 1
            logger.info(f"Document {doc_id} status reset to 'pending'.")
            add_log_entry("INFO", f"Document {doc_id} status reset to 'pending'.", step="reset_stuck_docs_status_reset", worker_id="SYSTEM_API", details={"doc_id": doc_id})

        success_msg = f"Reset {reset_count} stuck documents to 'pending' state. Encountered {error_count} errors during cleanup/reset."
        logger.info(success_msg)
        add_log_entry("INFO", success_msg, step="reset_stuck_docs_complete", worker_id="SYSTEM_API", details={"reset_count": reset_count, "error_count": error_count})
        return jsonify({"status": "Success", "message": success_msg, "reset_count": reset_count, "cleanup_errors": error_count}), 200

    except Exception as e:
        error_msg = f"Error resetting stuck documents: {e}"
        logger.error(error_msg, exc_info=True)
        add_log_entry("ERROR", "Error resetting stuck documents", step="reset_stuck_docs", worker_id="SYSTEM_API", details={"error": str(e)})
        return jsonify({"status": "Error", "message": error_msg}), 500


# --- Helper for Batch Deletion ---
def delete_collection(coll_ref, batch_size):
    """Deletes a collection in batches to avoid exceeding limits."""
    # Ensure db is accessible, might need current_app context if run outside request?
    # Assuming 'db' is globally accessible here as initialized in __init__
    deleted_count = 0
    docs = coll_ref.limit(batch_size).stream()
    while True:
        batch = db.batch()
        count = 0
        for doc in docs:
            batch.delete(doc.reference)
            count += 1
        if count == 0:
            break # No more documents to delete
        batch.commit()
        deleted_count += count
        # Get the next batch
        docs = coll_ref.limit(batch_size).stream()
    return deleted_count

# @system_bp.route('/delete-all-embeddings', methods=['POST'])
# @jwt_required()
def delete_all_embeddings_route():
    """Deletes all embeddings from the Vertex Vector Search index. Requires admin privileges."""
    # Add admin role check here if needed
    # Example: if not is_admin(get_jwt_identity()): return jsonify({"msg": "Admin access required."}), 403

    try:
        vertex_ai_service.delete_vectors_in_batches_from_firestore()
        return jsonify({"msg": "Successfully initiated deletion of all embeddings."}), 200
    except Exception as e:
        return jsonify({"msg": f"Failed to initiate deletion: {e}"}), 500

@system_bp.route('/clear-processing-data', methods=['DELETE'])
@jwt_required() # Protect this potentially destructive endpoint
def clear_processing_data():
    """
    Deletes all documents from processing-related collections.
    Deletes all documents from processing-related collections AND associated Vector Search datapoints.
    USE WITH CAUTION - INTENDED FOR DEVELOPMENT/TESTING ONLY.
    """
    collections_to_clear = [
        "document_metadata",
        "document_chunks",
        "processing_logs",
        "bulk_process_runs"
    ]
    results = {}
    batch_size = 100 # Firestore batch limit is 500, use smaller for safety
    vector_ids_to_remove = []
    vector_removal_success = False
    vector_removal_error = None

    current_app.logger.warning("Attempting to clear processing data including Vector Search. THIS IS A DESTRUCTIVE OPERATION.")
    add_log_entry("WARNING", "Attempting to clear processing data (Firestore + Vector Search).", step="clear_data_start", worker_id="SYSTEM_API")

    try:
        # --- Step 1: Get all chunk IDs (which are the vector datapoint IDs) ---
        vector_ids_to_remove, fetch_ids_error = get_all_chunk_ids()
        if fetch_ids_error:
            raise Exception(f"Failed to fetch chunk IDs before clearing: {fetch_ids_error}")
        results["vector_search"] = {"ids_found": len(vector_ids_to_remove)}
        current_app.logger.info(f"Found {len(vector_ids_to_remove)} chunk/vector IDs to remove.")

        # --- Step 2: Remove datapoints from Vector Search ---
        if vector_ids_to_remove:
            vector_removal_success, vector_removal_error = vertex_ai_service.remove_vector_datapoints(vector_ids_to_remove)
            results["vector_search"]["removed"] = vector_removal_success
            if vector_removal_error:
                results["vector_search"]["error"] = vector_removal_error
                # Decide whether to proceed if vector removal fails. For safety, let's stop.
                raise Exception(f"Failed to remove datapoints from Vector Search: {vector_removal_error}")
            else:
                 current_app.logger.info("Successfully initiated removal of datapoints from Vector Search.")
                 add_log_entry("INFO", f"Initiated removal of {len(vector_ids_to_remove)} datapoints from Vector Search.", step="clear_data_vector", worker_id="SYSTEM_API")
        else:
            vector_removal_success = True # No IDs to remove, so technically successful
            current_app.logger.info("No vector datapoints found to remove.")
            add_log_entry("INFO", "No vector datapoints found to remove.", step="clear_data_vector", worker_id="SYSTEM_API")

        # --- Step 3: Clear Firestore Collections (only if vector removal succeeded or was skipped) ---
        if vector_removal_success:
            current_app.logger.info("Proceeding to clear Firestore collections.")
            for coll_name in collections_to_clear:
                coll_ref = db.collection(coll_name)
                deleted = delete_collection(coll_ref, batch_size)
                results[coll_name] = {"deleted": deleted}
                current_app.logger.info(f"Deleted {deleted} documents from collection '{coll_name}'.")
                add_log_entry("INFO", f"Deleted {deleted} documents from collection '{coll_name}'.", step="clear_data_firestore", worker_id="SYSTEM_API")
        else:
             # This case should ideally be caught by the exception above, but as a safeguard:
             current_app.logger.error("Skipping Firestore deletion because Vector Search removal failed.")
             raise Exception("Skipped Firestore deletion due to Vector Search removal failure.")


        # --- Step 4: Clear Redis Queue ---
        redis_deleted_count = 0
        redis_error = None
        try:
            redis_client = get_redis_client()
            if redis_client:
                # Use the imported QUEUE_NAME directly
                redis_deleted_count = redis_client.delete(QUEUE_NAME)
                if redis_deleted_count > 0:
                    current_app.logger.info(f"Successfully deleted Redis queue '{QUEUE_NAME}'.")
                    add_log_entry("INFO", f"Deleted Redis queue '{QUEUE_NAME}'.", step="clear_data_redis", worker_id="SYSTEM_API")
                else:
                    current_app.logger.info(f"Redis queue '{QUEUE_NAME}' did not exist or was empty.")
                    add_log_entry("INFO", f"Redis queue '{QUEUE_NAME}' not found or empty.", step="clear_data_redis", worker_id="SYSTEM_API")
                results["redis_queue"] = {"deleted_keys": redis_deleted_count}
            else:
                 redis_error = "Could not get Redis client."
                 current_app.logger.error(redis_error)
                 add_log_entry("ERROR", redis_error, step="clear_data_redis", worker_id="SYSTEM_API")
                 results["redis_queue"] = {"error": redis_error}

        except Exception as redis_e:
             redis_error = f"Error clearing Redis queue: {redis_e}"
             current_app.logger.error(redis_error, exc_info=True)
             add_log_entry("ERROR", "Error clearing Redis queue.", step="clear_data_redis", worker_id="SYSTEM_API", details={"error": str(redis_e)})
             results["redis_queue"] = {"error": redis_error} # Corrected indentation
        # --- End Clear Redis Queue ---

        # Construct final success message
        success_msg = "Successfully cleared Firestore processing data"
        if vector_ids_to_remove:
            success_msg += f" and initiated removal of {len(vector_ids_to_remove)} Vector Search datapoints."
        else:
            success_msg += " (no Vector Search datapoints found)."

        if redis_deleted_count > 0:
             success_msg += " Redis queue also cleared."
        elif redis_error:
             success_msg += f" Failed to clear Redis queue: {redis_error}"
        else:
             success_msg += " Redis queue was empty or not found."

        add_log_entry("INFO", success_msg, step="clear_data_complete", worker_id="SYSTEM_API")
        return jsonify({"status": "Success","message": success_msg, "details": results}), 200

    except Exception as e:
        # Log the specific error that occurred (could be fetching IDs, vector removal, or Firestore deletion)
        error_msg = f"Error during clear processing data operation: {e}"
        current_app.logger.error(error_msg, exc_info=True)
        add_log_entry("ERROR", "Error clearing processing data.", step="clear_data_error", worker_id="SYSTEM_API", details={"error": str(e)})
        return jsonify({"status": "Error", "message": error_msg}), 500
