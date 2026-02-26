from flask import Blueprint, request, jsonify, Response, stream_with_context
from flask_jwt_extended import jwt_required, get_jwt_identity, get_jwt
from app.services import gcs_service
from app.services.progress_service import get_progress_tracker
from app.config import FILE_MANAGEMENT_BUCKET_NAME, GCS_SOURCE_ROOT, ROUTE_PREFIX, ADMIN_SERVER_URL, JWT_SECRET_KEY
from app.services.path_config_service import (
    get_path_config,
    set_path_config,
    get_dynamic_source_path,
    get_dynamic_base_path,
    get_all_path_configs,
    get_path_configs_for_user
)
from app.utils.utils import superadmin_required
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor
from flask_cors import CORS
from google.cloud import storage
from datetime import datetime, timedelta, timezone
import urllib.parse
import io
import logging
from app.services.activity_log_service import log_file_activity, log_navigation_activity
from app.models.activity_log_model import ActivityTypes
from app.models import file_management_model
import zipfile
from app.utils.debug_logger import debug_log, debug_error, debug_warn
from app import db  # Import Firestore client
import requests

logger = logging.getLogger(__name__)

MAX_FILENAME_LENGTH = 50


# Helper function to get user's file management permissions
def get_user_file_management_permissions(user_id: str) -> dict:
    """
    Get file management permissions for a user by calling Admin Backend API by User ID.
    Superadmin users get all permissions.
    Freshly fetches permissions every time (cache disabled).
    """
    default_permissions = {
        'can_rename_source': False,
        'can_delete_source': False,
        'can_upload': False,
        'can_create_root_folder_source': False,
        'can_create_folder_source': False,
        'can_delete_destination': False,
        'can_create_root_folder_destination': False,
        'can_create_folder_destination': False,
        'can_transfer': True,
        'can_view_all_transfer_history': False
    }
    
    all_permissions = {
        'can_rename_source': True,
        'can_delete_source': True,
        'can_upload': True,
        'can_create_root_folder_source': True,
        'can_create_folder_source': True,
        'can_delete_destination': True,
        'can_create_root_folder_destination': True,
        'can_create_folder_destination': True,
        'can_transfer': True,
        'can_view_all_transfer_history': True
    }
    
    try:
        # Check if user is superadmin from JWT claims
        claims = get_jwt()
        if claims.get('role') == 'superadmin':
            return all_permissions
        
        api_url = f"{ADMIN_SERVER_URL}/users/{urllib.parse.quote(user_id)}/permissions"

        headers = {
            "Content-Type": "application/json",
            "X-API-Key": JWT_SECRET_KEY
        }
        
        # debug_log(f"[PERMISSIONS] Fetching permissions from: {api_url}")
        response = requests.get(api_url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            # If user is superadmin according to admin backend, give all permissions
            if data.get('role') == 'superadmin':
                return all_permissions
                
            permissions_data = data.get('permissions', default_permissions)
            
            # Ensure all keys exist
            for key in default_permissions:
                if key not in permissions_data:
                    permissions_data[key] = default_permissions[key]
                    
            return permissions_data
        else:
            debug_error(f"[PERMISSIONS] Admin API returned status {response.status_code} for {user_id}")
            return default_permissions
            
    except requests.exceptions.RequestException as e:
        debug_error(f"[PERMISSIONS] Error calling Admin API for {user_id}: {e}")
        return default_permissions
    except Exception as e:
        debug_error(f"[PERMISSIONS] Error getting file management permissions for {user_id}: {e}")
        return default_permissions


def check_file_permission(permission_key: str, user_id: str) -> tuple[bool, str]:
    """
    Check if user has a specific file management permission.
    Returns (has_permission: bool, error_message: str)
    """
    permissions = get_user_file_management_permissions(user_id)
    
    # DEBUG: Log permissions and key being checked
    # debug_log(f"[PERMISSIONS DEBUG] Checking {permission_key} for {user_id}")
    # debug_log(f"[PERMISSIONS DEBUG] User permissions: {permissions}")
    
    has_permission = permissions.get(permission_key, False)
    
    if not has_permission:
        # DEBUG: Log failure
        debug_error(f"[PERMISSIONS FAIL] User {user_id} DENIED {permission_key}. Permissions: {permissions}")
        
        permission_labels = {
            'can_rename_source': 'rename files/folders in source',
            'can_delete_source': 'delete files/folders in source',
            'can_upload': 'upload files to source',
            'can_create_root_folder_source': 'create root folders in source',
            'can_create_folder_source': 'create subfolders in source',
            'can_delete_destination': 'delete files/folders in destination',
            'can_create_root_folder_destination': 'create root folders in destination',
            'can_create_folder_destination': 'create subfolders in destination',
            'can_transfer': 'transfer files/folders',
            'can_view_all_transfer_history': 'view all transfer history'
        }
        error_message = f"You don't have permission to {permission_labels.get(permission_key, permission_key)}."
        return False, error_message
    return True, ""

def validate_filename(filename: str) -> tuple[bool, str]:
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
    if "  " in filename:
        return False, "Filename must not contain consecutive spaces."
    forbidden_chars = set('\\/:*?"<>|')
    if any(ch in forbidden_chars for ch in filename):
        return False, "Filename contains invalid characters."
    allowed_chars = set("-_.")
    # Allow spaces (v3.13.1)
    if not all(ch.isalnum() or ch in allowed_chars or ch == ' ' for ch in filename):
        return False, "Filename contains invalid characters."
    if "." not in filename:
        return False, "Filename must have a .pdf extension."
    base, ext = filename.rsplit(".", 1)
    if not base:
        return False, "Filename must not start with a dot."
    if ext.lower() != "pdf":
        return False, "Only .pdf files are allowed."
    return True, ""


gcs_bp = Blueprint('gcs_bp', __name__)
CORS(gcs_bp, origins="*")  # Allow all origins

@gcs_bp.route('/config', methods=['GET'])
@jwt_required()
def get_gcs_config():
    """Get GCS configuration including source root."""
    return jsonify({
        "sourceRoot": GCS_SOURCE_ROOT,
        "bucketName": FILE_MANAGEMENT_BUCKET_NAME
    })

#
@gcs_bp.route('/source-directory', methods=['GET'])
@jwt_required()
def list_source_directory():
    path = request.args.get('path', '')
    page = int(request.args.get('page', 1))
    page_size = int(request.args.get('page_size', 50))

    # Validate pagination parameters
    if page < 1:
        page = 1
    if page_size < 1 or page_size > 200:  # Limit max page size
        page_size = 50

    # The full path is now passed from the frontend
    full_path = path

    contents, error = gcs_service.list_gcs_directory(FILE_MANAGEMENT_BUCKET_NAME, full_path, delimiter='/', page=page, page_size=page_size)
    if error:
        return jsonify({"message": "Error listing source directory", "error": error}), 500
    return jsonify(contents)

@gcs_bp.route('/destinations', methods=['GET'])
@jwt_required()
def list_destinations():
    path = request.args.get('path', '')
    page = int(request.args.get('page', 1))
    page_size = int(request.args.get('page_size', 50))

    # Validate pagination parameters
    if page < 1:
        page = 1
    if page_size < 1 or page_size > 200:  # Limit max page size
        page_size = 50

    if not path:
        # If no path specified, get root destinations excluding source root and its parents
        destinations, error = gcs_service.get_destination_roots(FILE_MANAGEMENT_BUCKET_NAME, GCS_SOURCE_ROOT)
        if error:
            return jsonify({"message": "Error listing destination roots", "error": error}), 500
        return jsonify({"folders": destinations.get("destinations", []), "files": [], "pagination": {"current_page": 1, "total_pages": 1, "page_size": page_size, "total_items": len(destinations.get("destinations", [])), "total_folders": len(destinations.get("destinations", [])), "total_files": 0, "has_next": False, "has_prev": False}})
    else:
        # If path specified, list directory contents normally with pagination
        contents, error = gcs_service.list_gcs_directory(FILE_MANAGEMENT_BUCKET_NAME, path, delimiter='/', page=page, page_size=page_size)
        if error:
            return jsonify({"message": "Error listing destinations", "error": error}), 500
        return jsonify(contents)

@gcs_bp.route('/destinations-with-subfolders', methods=['GET'])
@jwt_required()
def list_destinations_with_subfolders():
    """Get all destination folders and files in a tree structure including all subfolders recursively.
    Supports optional search filter to find files/folders by name.
    """
    try:
        path = request.args.get('path', '')
        search_term = request.args.get('search', '').strip()  # Add search parameter
        page = int(request.args.get('page', 1))
        page_size = int(request.args.get('page_size', 50))

        # Validate pagination parameters
        if page < 1:
            page = 1
        if page_size < 1 or page_size > 200:  # Limit max page size
            page_size = 50

        # Strip leading slashes - GCS paths don't have leading slashes
        path = path.lstrip('/')

        logger.info(f"Listing destinations recursively with path: '{path}', page: {page}, page_size: {page_size}, search: '{search_term}'")
        debug_log(f"Listing destinations recursively with path: '{path}', page: {page}, page_size: {page_size}, search: '{search_term}'")

        # List destinations recursively at specified path with optional search
        contents, error = gcs_service.list_destinations_recursive(
            FILE_MANAGEMENT_BUCKET_NAME,
            GCS_SOURCE_ROOT,
            path=path,
            page=page,
            page_size=page_size,
            search_term=search_term if search_term else None  # Pass search term if provided
        )

        if error:
            return jsonify({
                "message": f"Error listing destinations recursively at path: {path}",
                "error": error
            }), 500

        logger.info(f"Found {contents.get('total_folders', 0)} folders and {contents.get('total_files', 0)} files recursively")

        # Transform the tree structure to match the hierarchical format
        root_tree = contents.get("tree", {})
        root_children = root_tree.get("children", []) if root_tree else []
        
        # Get pagination info from contents
        pagination_info = contents.get("pagination", {})
        
        # Transform the tree structure to match the exact format
        transformed_tree = []
        for child in root_children:
            transformed_item = transform_to_list_with_subfolders_format(child, depth=0)
            if transformed_item:
                transformed_tree.append(transformed_item)
        
        # Get totals from contents (already calculated by the service)
        total_folders = contents.get("total_folders", 0)
        total_files = contents.get("total_files", 0)
        
        # Return the transformed array with pagination metadata
        response_data = {
            "items": transformed_tree,
            "pagination": pagination_info,
            "totals": {
                "total_folders": total_folders,
                "total_files": total_files,
                "total_items": total_folders + total_files
            }
        }
        
        # Include search term in response if used
        if search_term:
            response_data["search"] = search_term
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"Error listing destinations recursively at path {path}: {e}", exc_info=True)
        return jsonify({
            "message": "Error listing destinations recursively",
            "error": str(e)
        }), 500

@gcs_bp.route('/destination-directory', methods=['GET'])
@jwt_required()
def list_destination_directory():
    
    root = request.args.get('root')
    path = request.args.get('path', '')
    if not root:
        return jsonify({"message": "Missing 'root' parameter"}), 400
        
    full_path = f"{root}{path}"
    
    contents, error = gcs_service.list_gcs_directory(FILE_MANAGEMENT_BUCKET_NAME, full_path, delimiter='/')
   
    if error:
        return jsonify({"message": "Error listing destination directory", "error": error}), 500
    return jsonify(contents)

@gcs_bp.route('/check-transfer-conflicts', methods=['POST'])
@jwt_required()
def check_transfer_conflicts_route():
    """Check for file conflicts before transfer."""
    data = request.get_json()
    if not data or 'source_paths' not in data or 'destination' not in data:
        return jsonify({"message": "Invalid request. 'source_paths' and 'destination' are required"}), 400
    
    source_paths = data['source_paths']
    dest_root = data['destination'].get('root')
    dest_path = data['destination'].get('path', '')
    
    if not source_paths or not dest_root:
         return jsonify({"message": "Missing source paths or destination root"}), 400
         
    conflicts, error = gcs_service.check_transfer_conflicts(
        FILE_MANAGEMENT_BUCKET_NAME,
        source_paths,
        dest_root,
        dest_path
    )
    
    if error:
        return jsonify({"message": "Error checking conflicts", "error": error}), 500
        
    return jsonify({"conflicts": conflicts}), 200

@gcs_bp.route('/transfer', methods=['POST'])
@jwt_required()
def initiate_transfer():
    debug_log("request for transfer")
    
    # Get current user ID and email from JWT
    user_id = get_jwt_identity()
    user_email = get_jwt().get('email', 'unknown_user')
    
    # Check transfer permission
    has_permission, error_msg = check_file_permission('can_transfer', user_id)
    if not has_permission:
        return jsonify({"message": error_msg}), 403
    
    data = request.get_json()
    if not data or 'source' not in data or 'destination' not in data or 'operation' not in data:
        return jsonify({"message": "Invalid transfer request"}), 400

    source_path = data['source'].get('path')
    dest_root = data['destination'].get('root')
    dest_path = data['destination'].get('path', '')
    operation = data.get('operation')
    
    if not source_path or not dest_root:
        return jsonify({"message": "Missing source or destination path"}), 400

    file_name = source_path.split('/')[-1]
    destination_blob_name = f"{dest_root}{dest_path}{file_name}"

    result, error = gcs_service.transfer_gcs_object(FILE_MANAGEMENT_BUCKET_NAME, source_path, destination_blob_name, operation)
    if error:
        return jsonify({"message": "Error initiating transfer", "error": error}), 500
    
    # Update or create file metadata based on operation
    dest_folder_path = '/'.join(destination_blob_name.split('/')[:-1])
    if dest_folder_path:
        dest_folder_path += '/'
    
    if operation == 'move':
        # Update existing file metadata (file moved)
        metadata_success, metadata_error = file_management_model.update_file_path(
            old_path=source_path,
            new_path=destination_blob_name,
            new_folder_path=dest_folder_path
        )
        
        if not metadata_success:
            # Log warning but don't fail the request since file was transferred in GCS
            logger.warning(f"File transferred in GCS but metadata update failed: {metadata_error}")
    elif operation == 'copy':
        # Create new metadata for copied file (source file still exists)
        try:
            # Get file info from GCS to create metadata
            storage_client = storage.Client()
            bucket = storage_client.bucket(FILE_MANAGEMENT_BUCKET_NAME)
            dest_blob = bucket.blob(destination_blob_name)
            
            if dest_blob.exists():
                file_id, metadata_error = file_management_model.create_file_metadata(
                    file_path=destination_blob_name,
                    file_name=file_name,
                    folder_path=dest_folder_path,
                    gcs_uri=f"gs://{FILE_MANAGEMENT_BUCKET_NAME}/{destination_blob_name}",
                    file_size=dest_blob.size,
                    content_type=dest_blob.content_type or 'application/pdf',
                    user_email=user_email
                )
                
                if metadata_error:
                    logger.warning(f"File copied in GCS but metadata creation failed: {metadata_error}")
        except Exception as e:
            logger.warning(f"Could not create metadata for copied file: {e}")
    
    # Log successful file transfer initiation
    log_file_activity(
        user_email=user_email,
        activity_type=ActivityTypes.FILE_TRANSFER,
        filename=file_name,
        file_info={
            'source_path': source_path,
            'destination_path': destination_blob_name,
            'operation': operation,
            'transfer_id': result.get('transferId') if result else None
        },
        request_obj=request
    )
        
    return jsonify(result), 202

@gcs_bp.route('/transfers/<string:transfer_id>', methods=['GET'])
@jwt_required()
def get_transfer_status(transfer_id):
    status, error = gcs_service.get_transfer_status(transfer_id)
    if error:
        return jsonify({"message": "Error getting transfer status", "error": error}), 500
    return jsonify(status)

@gcs_bp.route('/transfers', methods=['GET'])
@jwt_required()
def get_transfers():
    transfers, error = gcs_service.get_user_transfers()
    if error:
        return jsonify({"message": "Error getting transfers", "error": error}), 500
    return jsonify(transfers)

@gcs_bp.route('/user-transfers', methods=['GET'])
@jwt_required()
def get_user_transfers_paginated():
    """
    Get user transfers with pagination and filtering support.
    
    Query Parameters:
    - page: Page number (1-based, default: 1)
    - limit: Number of transfers per page (1-200, default: 50)
    - status: Filter by status (e.g., 'succeeded', 'failed', 'running', 'queued')
    - operation: Filter by operation type ('move', 'undo' or 'delete')
    - transfer_type: Filter by transfer type ('Folder' for folder transfers, 'File' for file transfers)
    - date: Filter by specific date (ISO format: YYYY-MM-DD). Filters transfers on this exact date
    - start_date: Start date for date range filter (ISO format: YYYY-MM-DD). Inclusive.
    - end_date: End date for date range filter (ISO format: YYYY-MM-DD). Inclusive.
    
    Note: Use either 'date' OR 'start_date'/'end_date', not both. 'date' takes precedence.
    
    Returns:
    - JSON response with transfers list, pagination info, and filters applied
    """
    try:
        # Get current user ID from JWT
        user_id = get_jwt_identity()
        if not user_id:
            return jsonify({"message": "Could not identify user from token"}), 401
        
        # Get query parameters
        page = request.args.get('page', 1, type=int)
        limit = request.args.get('limit', 50, type=int)
        status = request.args.get('status', None, type=str)
        operation = request.args.get('operation', None, type=str)
        transfer_type = request.args.get('transfer_type', None, type=str)
        date = request.args.get('date', None, type=str)
        start_date = request.args.get('start_date', None, type=str)
        end_date = request.args.get('end_date', None, type=str)
        
        # Validate limit
        limit = max(1, min(limit, 200))
        page = max(1, page)
        
        # Validate operation if provided
        if operation and operation not in ['move', 'undo', 'delete']:
            return jsonify({
                "message": "Invalid operation. Must be 'move', 'undo', or 'delete'"
            }), 400
        
        # Validate date parameters
        if date and (start_date or end_date):
            return jsonify({
                "message": "Cannot use both 'date' and 'start_date'/'end_date' filters. Use either 'date' OR 'start_date'/'end_date'"
            }), 400
        
        # Check if user has permission to view all transfer history
        permissions = get_user_file_management_permissions(user_id)
        can_view_all = permissions.get('can_view_all_transfer_history', False)
        
        # If user has permission to view all, pass user_id=None to service to fetch all
        # Otherwise, pass current user_id
        service_user_id = None if can_view_all else user_id
        
        # Call service function
        transfers, pagination_info, error = gcs_service.get_user_transfers_paginated(
            user_id=service_user_id,
            limit=limit,
            page=page,
            status=status,
            operation=operation,
            transfer_type=transfer_type,
            date=date,
            start_date=start_date,
            end_date=end_date
        )
        
        if error:
            logger.error(f"Error getting user transfers: {error}")
            return jsonify({"message": "Error getting transfers", "error": error}), 500
        
        # Build response
        response_data = {
            "transfers": transfers,
            "pagination": pagination_info
        }
        
        # Include applied filters in response
        filters = {}
        if status:
            filters['status'] = status
        if operation:
            filters['operation'] = operation
        if transfer_type:
            filters['transfer_type'] = transfer_type
        if date:
            filters['date'] = date
        if start_date:
            filters['start_date'] = start_date
        if end_date:
            filters['end_date'] = end_date
        if filters:
            response_data['filters'] = filters
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"Error in get_user_transfers_paginated endpoint: {e}", exc_info=True)
        return jsonify({"message": "An unexpected error occurred while fetching transfers"}), 500


@gcs_bp.route('/transfer-folder', methods=['POST'])
@jwt_required()
def transfer_folder():
    """Transfer an entire folder recursively with progress tracking."""
    debug_log("request for folder transfer")
    
    # Capture user_id before background thread (JWT context will be lost)
    user_id = get_jwt_identity()
    debug_log(f"Captured user_id for folder transfer: {user_id}")
    
    data = request.get_json()
    
    if not data or 'sourceFolderPath' not in data or 'destination' not in data:
        return jsonify({"message": "Invalid folder transfer request"}), 400

    source_folder_path = data['sourceFolderPath']
    dest_root = data['destination'].get('root')
    dest_path = data['destination'].get('path', '')
    operation = data.get('operation', 'move')
    
    if not source_folder_path or not dest_root:
        return jsonify({"message": "Missing source folder or destination path"}), 400

    # Extract folder name from source path
    folder_name = source_folder_path.rstrip('/').split('/')[-1]
    destination_folder_path = f"{dest_root}{dest_path}{folder_name}/"

    debug_log(f"=== FOLDER TRANSFER DEBUG ===")
    debug_log(f"Source folder path: {source_folder_path}")
    debug_log(f"Dest root: {dest_root}")
    debug_log(f"Dest path: {dest_path}")
    debug_log(f"Folder name: {folder_name}")
    debug_log(f"Final destination: {destination_folder_path}")
    debug_log(f"Operation: {operation}")
    debug_log(f"=== END DEBUG ===")
    
    # Create transfer ID and progress tracking
    transfer_id = str(uuid.uuid4())
    progress_tracker = get_progress_tracker()
    
    # Count files and folders first for progress tracking
    try:
        from google.cloud import storage
        storage_client = storage.Client()
        bucket = storage_client.bucket(FILE_MANAGEMENT_BUCKET_NAME)
        source_path = source_folder_path if source_folder_path.endswith('/') else source_folder_path + '/'
        all_blobs = list(storage_client.list_blobs(bucket, prefix=source_path))
        files_to_transfer = [blob for blob in all_blobs if not (blob.name.endswith('/') and blob.size == 0)]
        folder_markers = [blob for blob in all_blobs if blob.name.endswith('/') and blob.size == 0]
        total_files = len(files_to_transfer)
        total_folders = len(folder_markers)
        total_items = total_files + total_folders
        
        # Create progress entry (using total_items for accurate progress tracking)
        progress_tracker.create_transfer(transfer_id, 'folder', operation, total_items)
        
    except Exception as e:
        return jsonify({"message": "Error counting files", "error": str(e)}), 500
    
    def run_transfer():
        """Run the transfer in a background thread."""
        debug_log(f"=== STARTING FOLDER TRANSFER THREAD ===")
        debug_log(f"Transfer ID: {transfer_id}")
        debug_log(f"Folder: {folder_name}")
        debug_log(f"Total files: {total_files}")
        try:
            progress_tracker.start_transfer(transfer_id)
            progress_callback = progress_tracker.create_progress_callback(transfer_id)
            
            result, error = gcs_service.transfer_gcs_folder(
                FILE_MANAGEMENT_BUCKET_NAME, 
                source_folder_path, 
                destination_folder_path, 
                operation,
                progress_callback,
                user_id  # Pass user_id to handle JWT context loss in background thread
            )
            
            # Handle transfer completion (individual files are logged in gcs_service.py)
            if error and result is None:
                # Complete failure
                progress_tracker.fail_transfer(transfer_id, error)
                debug_error(f"=== FOLDER TRANSFER FAILED ===")
                debug_error(f"Folder: {folder_name}, Error: {error}")
                debug_log(f"Individual file failures are logged separately")
                
            else:
                # Success or partial success
                progress_tracker.complete_transfer(transfer_id)
                if error:
                    debug_warn(f"=== FOLDER TRANSFER COMPLETED WITH SOME ERRORS ===")
                    debug_warn(f"Folder: {folder_name}, Files: {total_files}, Partial Error: {error}")
                else:
                    debug_log(f"=== FOLDER TRANSFER COMPLETED SUCCESSFULLY ===")
                    debug_log(f"Folder: {folder_name}, Files: {total_files}")
                    
                    # Log successful folder transfer
                    log_file_activity(
                        user_email=user_id,
                        activity_type=ActivityTypes.FOLDER_TRANSFER,
                        filename=folder_name,
                        file_info={
                            'source_folder': source_folder_path,
                            'destination_folder': destination_folder_path,
                            'operation': operation,
                            'total_files': total_files,
                            'transfer_id': transfer_id
                        },
                        request_obj=None  # No request object in background thread
                    )
                debug_log(f"Individual file transfers are logged with 'Folder' tag")
                
        except Exception as e:
            progress_tracker.fail_transfer(transfer_id, str(e))
            debug_error(f"=== FOLDER TRANSFER EXCEPTION ===")
            debug_error(f"Folder: {folder_name}, Exception: {str(e)}")
            debug_log(f"Individual file failures are logged separately with 'Folder' tag")
            
    
    # Start transfer in background thread
    transfer_thread = threading.Thread(target=run_transfer)
    transfer_thread.daemon = True
    transfer_thread.start()
    
    return jsonify({
        "message": "Folder transfer started",
        "transferId": transfer_id,
        "totalFiles": total_files,
        "totalFolders": total_folders,
        "totalItems": total_items
    }), 202


@gcs_bp.route('/progress/<string:transfer_id>', methods=['GET'])
@jwt_required()
def get_transfer_progress(transfer_id):
    """Get progress of a specific transfer."""
    progress_tracker = get_progress_tracker()
    transfer = progress_tracker.get_transfer(transfer_id)
    
    if not transfer:
        return jsonify({"message": "Transfer not found"}), 404
    
    return jsonify(transfer)


@gcs_bp.route('/progress', methods=['GET'])
@jwt_required()
def get_all_progress():
    """Get progress of all transfers."""
    progress_tracker = get_progress_tracker()
    transfers = progress_tracker.get_all_transfers()
    
    return jsonify(list(transfers.values()))


@gcs_bp.route('/undo-transfer', methods=['POST'])
@jwt_required()
def undo_transfer():
    """
    Undo a file transfer by moving it back to source.
    Only works if file hasn't been batch processed.
    
    Request Body:
    {
        "transfer_id": "<transfer_id>"
    }
    
    Returns:
    - 200: File successfully moved back to source
    - 400: Error (file processed, not found, etc.)
    """
    data = request.get_json()
    if not data or 'transfer_id' not in data:
        return jsonify({"message": "Missing transfer_id in request"}), 400
    
    transfer_id = data['transfer_id']
    user_id = get_jwt_identity()
    user_email = get_jwt().get('email', 'unknown_user')
    
    # Check if user has permission to view all transfer history (and thus undo any)
    permissions = get_user_file_management_permissions(user_id)
    can_view_all = permissions.get('can_view_all_transfer_history', False)
    
    result, error = gcs_service.undo_file_transfer(
        FILE_MANAGEMENT_BUCKET_NAME, 
        transfer_id, 
        user_id=None if can_view_all else user_id
    )
    
    if error:
        return jsonify({"message": "Error undoing transfer", "error": error}), 400
    
    # Log activity
    log_file_activity(
        user_email=user_email,
        activity_type=ActivityTypes.FILE_TRANSFER,
        filename=result.get('fileName', 'unknown'),
        file_info={
            'transfer_id': transfer_id,
            'operation': 'undo',
            'source_path': result.get('source_path'),
            'destination_path': result.get('destination_path')
        },
        request_obj=request
    )
    
    return jsonify(result), 200


@gcs_bp.route('/delete-destination-file', methods=['DELETE'])
@jwt_required()
def delete_destination_file_route():
    """
    Delete a file from destination.
    Only works if file hasn't been batch processed.
    
    Request Body:
    {
        "destination_path": "destination/folder/file.pdf"
    }
    
    Returns:
    - 200: File successfully deleted
    - 400: Error (file processed, not found, etc.)
    """
    data = request.get_json()
    if not data or 'destination_path' not in data:
        return jsonify({"message": "Missing destination_path in request"}), 400
    
    destination_path = data['destination_path']
    user_id = get_jwt_identity()
    user_email = get_jwt().get('email', 'unknown_user')
    
    result, error = gcs_service.delete_destination_file(
        FILE_MANAGEMENT_BUCKET_NAME, 
        destination_path, 
        user_id # Service now expects user_id for permission checks
    )
    
    if error:
        return jsonify({"message": "Error deleting file", "error": error}), 400
    
    # Log activity
    log_file_activity(
        user_email=user_email,
        activity_type=ActivityTypes.FILE_DELETE,
        filename=destination_path.split('/')[-1],
        file_info={
            'destination_path': destination_path,
            'operation': 'delete'
        },
        request_obj=request
    )
    
    return jsonify(result), 200


@gcs_bp.route('/check-file-eligibility', methods=['POST'])
@jwt_required()
def check_file_eligibility():
    """
    Check if a file can be undone or deleted.
    Returns whether file has been batch processed.
    
    Request Body:
    {
        "destination_path": "destination/folder/file.pdf"
    }
    
    Returns:
    - 200: Eligibility status
    """
    data = request.get_json()
    if not data or 'destination_path' not in data:
        return jsonify({"message": "Missing destination_path in request"}), 400
    
    destination_path = data['destination_path']
    is_processed = gcs_service.is_file_batch_processed(
        FILE_MANAGEMENT_BUCKET_NAME, 
        destination_path
    )
    
    return jsonify({
        "destination_path": destination_path,
        "is_batch_processed": is_processed,
        "can_undo": not is_processed,
        "can_delete": not is_processed
    }), 200


@gcs_bp.route('/create-folder', methods=['POST'])
@jwt_required()
def create_folder():
    """Create a new folder in GCS."""
    debug_log("request for folder creation")
    user_id = get_jwt_identity()
    user_email = get_jwt().get('email', 'unknown_user')
    data = request.get_json()
    
    if not data or 'folderName' not in data or 'currentPath' not in data:
        return jsonify({"message": "Invalid folder creation request"}), 400

    folder_name = data['folderName'].strip()
    current_path = data['currentPath']
    
    debug_log(f"[CREATE FOLDER DEBUG] Request: folderName='{folder_name}', currentPath='{current_path}'")
    
    # Determine if this is source or destination based on path
    # Source paths typically start with source_root (e.g., "Georgia 14/Pending Files/")
    # Destination paths are everything else
    source_root = get_dynamic_source_path()
    is_source = current_path.startswith(source_root) if source_root else False
    
    debug_log(f"[CREATE FOLDER DEBUG] source_root='{source_root}', is_source={is_source}")

    # Determine if this is a root-level or subfolder-level creation
    # Root-level: currentPath equals the source_root or destination base path (no additional nesting)
    # Subfolder-level: currentPath is deeper than the base path
    if is_source:
        is_root_level = (current_path.rstrip('/') == source_root.rstrip('/'))
        permission_key = 'can_create_root_folder_source' if is_root_level else 'can_create_folder_source'
    else:
        # For destination, check if creating at the destination root
        dest_root = source_root.split('/')[0] + '/' if source_root else ''
        is_root_level = (current_path.rstrip('/') == dest_root.rstrip('/')) or current_path.count('/') <= 1
        permission_key = 'can_create_root_folder_destination' if is_root_level else 'can_create_folder_destination'
    
    debug_log(f"[CREATE FOLDER DEBUG] is_root_level={is_root_level}, required_permission={permission_key}")

    has_permission, error_msg = check_file_permission(permission_key, user_id)
    if not has_permission:
        debug_error(f"[CREATE FOLDER DENIED] User {user_id} denied. Required: {permission_key}")
        return jsonify({"message": error_msg}), 403
    
    if not folder_name:
        return jsonify({"message": "Folder name is required"}), 400

    # Validate folder name
    if '/' in folder_name or '\\' in folder_name:
        return jsonify({"message": "Folder name cannot contain slashes"}), 400

    # Create full folder path
    if current_path and not current_path.endswith('/'):
        current_path += '/'
    
    full_folder_path = f"{current_path}{folder_name}/"

    # Check if folder with same name already exists in the parent path
    existing_folder, _ = file_management_model.get_folder_metadata(full_folder_path)
    if existing_folder:
        return jsonify({"message": f"Folder '{folder_name}' already exists in this location"}), 409

    debug_log(f"Creating folder: {full_folder_path}")
    
    result, error = gcs_service.create_gcs_folder(FILE_MANAGEMENT_BUCKET_NAME, full_folder_path)
    
    if error:
        return jsonify({"message": "Error creating folder", "error": error}), 500
    
    # Create folder metadata in Firestore
    folder_id, metadata_error = file_management_model.create_folder_metadata(
        folder_path=full_folder_path,
        folder_name=folder_name,
        parent_path=current_path,
        user_email=user_id # Using user_id as the primary identifier in metadata
    )
    
    if metadata_error:
        # Log warning but don't fail the request since folder was created in GCS
        debug_warn(f"Folder created in GCS but metadata creation failed: {metadata_error}")
    
    # Log successful folder creation
    log_file_activity(
        user_email=user_email,
        activity_type=ActivityTypes.FOLDER_CREATE,
        filename=folder_name,
        file_info={
            'folder_path': full_folder_path,
            'parent_path': current_path,
            'folder_id': folder_id
        },
        request_obj=request
    )
    
    # Add metadata info to result
    if folder_id:
        created_at = datetime.now(tz=timezone.utc).isoformat()
        result['folder_id'] = folder_id
        result['created_at'] = created_at
        result['created_date'] = created_at
        result['updated_date'] = created_at
    
    return jsonify(result), 201


from app import config # Import config
import os
from google.cloud import storage
from google.cloud.exceptions import NotFound
# Initialize GCS client using project ID from config
storage_client = storage.Client(project=config.PROJECT_ID)


def upload_to_gcs(file_stream, filename, content_type, target_path):
    """Uploads a file stream to specific path in Google Cloud Storage."""
    try:
        bucket = storage_client.get_bucket(FILE_MANAGEMENT_BUCKET_NAME)
    except NotFound:
        debug_error(f"Bucket '{FILE_MANAGEMENT_BUCKET_NAME}' not found.")
        return None, None, None, f"Bucket '{FILE_MANAGEMENT_BUCKET_NAME}' not found."
    except Exception as e:
        debug_error(f"Could not get bucket '{FILE_MANAGEMENT_BUCKET_NAME}': {e}")
        return None, None, None, f"Could not access bucket '{FILE_MANAGEMENT_BUCKET_NAME}'."

    # Sanitize filename to prevent path traversal and invalid characters
    safe_filename = "".join(c if c.isalnum() or c in ['.', '_', '-', ' '] else '_' for c in filename)
    
    # Use the exact target path provided (already includes source root folder structure)
    blob_name = target_path.replace("\\", "/")  # Ensure forward slashes for GCS
    
    blob = bucket.blob(blob_name)
    
    try:
        # Reset stream position
        file_stream.seek(0)
        
        # Get file size
        file_stream.seek(0, os.SEEK_END)
        file_size = file_stream.tell()
        file_stream.seek(0)  # Reset to beginning
        
        # Upload the stream
        blob.upload_from_file(file_stream, content_type=content_type)
        debug_log(f"File {filename} uploaded to {blob_name} in bucket {FILE_MANAGEMENT_BUCKET_NAME}. Size: {file_size} bytes.")

        # Return the GCS URI, blob name, and file size
        gcs_uri = f"gs://{FILE_MANAGEMENT_BUCKET_NAME}/{blob_name}"
        return gcs_uri, blob_name, file_size, None  # gcs_uri, blob_name, file_size, error
        
    except Exception as e:
        debug_error(f"Failed to upload {filename} to GCS: {e}")
        return None, None, None, f"Failed to upload file to storage: {e}"


@gcs_bp.route('/upload-files', methods=['POST'])
@jwt_required()
def upload_files():
    try:
        user_id = get_jwt_identity()
        user_email = get_jwt().get('email', 'unknown_user')
        
        # Check upload permission
        has_permission, error_msg = check_file_permission('can_upload', user_id)
        if not has_permission:
            return jsonify({"message": error_msg, "error": "Permission denied"}), 403
        
        # Get the target path from form data or query params
        target_path = request.form.get('path', '') or request.args.get('path', '')
        
        # Validate that the path is within the configured source root
        source_root_normalized = GCS_SOURCE_ROOT.rstrip('/')
        if not target_path.startswith(f'{source_root_normalized}/') and target_path != source_root_normalized:
            return jsonify({
                "message": f"Upload only allowed within {source_root_normalized} directory",
                "error": "Invalid path"
            }), 400
        
        # Check if files were uploaded
        if 'files' not in request.files:
            return jsonify({
                "message": "No files provided",
                "error": "No files in request"
            }), 400
        
        files = request.files.getlist('files')
        
        if not files or all(file.filename == '' for file in files):
            return jsonify({
                "message": "No files selected",
                "error": "Empty file list"
            }), 400
        
        uploaded_files = []
        errors = []
        
        for file in files:
            if file.filename == '':
                continue
                
            # Validate file type (PDF only)
            if not file.filename.lower().endswith('.pdf'):
                errors.append(f"File '{file.filename}' is not a PDF")
                continue
            
            # Validate file content type
            if file.content_type != 'application/pdf':
                errors.append(f"File '{file.filename}' does not have PDF content type")
                continue
            
            try:
                # Construct the full GCS path
                if target_path.endswith('/'):
                    gcs_path = f"{target_path}{file.filename}"
                else:
                    gcs_path = f"{target_path}/{file.filename}" if target_path else f"{GCS_SOURCE_ROOT}{file.filename}"
                
                # Upload file to GCS using the custom function
                gcs_uri, blob_name, file_size, upload_error = upload_to_gcs(
                    file.stream,
                    file.filename,
                    file.content_type,
                    gcs_path
                )
                
                if upload_error is None:
                    # Get file metadata if available to include dates
                    try:
                        file_metadata, _ = file_management_model.get_file_metadata(gcs_path)
                        created_date = file_metadata.get('created_at') if file_metadata else None
                        updated_date = file_metadata.get('updated_at') if file_metadata else None
                    except Exception:
                        created_date = None
                        updated_date = None
                    
                    uploaded_files.append({
                        "filename": file.filename,
                        "path": gcs_path,
                        "gcs_uri": gcs_uri,
                        "blob_name": blob_name,
                        "size": file_size,
                        "created_date": created_date,
                        "updated_date": updated_date
                    })
                    
                    # Log successful file upload to GCS
                    log_file_activity(
                        user_email=user_email,
                        activity_type=ActivityTypes.FILE_UPLOAD,
                        filename=file.filename,
                        file_info={
                            'gcs_path': gcs_path,
                            'file_size': file_size,
                            'upload_method': 'file_management'
                        },
                        request_obj=request
                    )
                else:
                    errors.append(f"Failed to upload '{file.filename}': {upload_error}")
                    
            except Exception as e:
                errors.append(f"Error uploading '{file.filename}': {str(e)}")
        
        # Prepare response
        response_data = {
            "message": f"Upload completed. {len(uploaded_files)} files uploaded successfully",
            "uploaded_files": uploaded_files,
            "total_uploaded": len(uploaded_files),
            "total_attempted": len([f for f in files if f.filename != ''])
        }
        
        if errors:
            response_data["errors"] = errors
            response_data["total_errors"] = len(errors)
        
        # Return appropriate status code
        if uploaded_files and not errors:
            return jsonify(response_data), 201  # Created
        elif uploaded_files and errors:
            return jsonify(response_data), 207  # Multi-Status (partial success)
        else:
            return jsonify({
                "message": "No files were uploaded successfully",
                "errors": errors,
                "total_errors": len(errors)
            }), 400
            
    except Exception as e:
        return jsonify({
            "message": "Internal server error during file upload",
            "error": str(e)
        }), 500

@gcs_bp.route('/upload-files-to-path', methods=['POST'])
@jwt_required()
def upload_files_to_path():
    """Upload files to a path with conflict detection. Returns conflicts if files already exist."""
    try:
        user_id = get_jwt_identity()
        user_email = get_jwt().get('email', 'unknown_user')

        # Check upload permission
        has_permission, error_msg = check_file_permission('can_upload', user_id)
        if not has_permission:
            return jsonify({"message": error_msg, "error": "Permission denied"}), 403

        # Get dynamic source path
        dynamic_source_root = get_dynamic_source_path()
        if not dynamic_source_root:
            return jsonify({
                "message": "Source path not configured",
                "error": "Dynamic source path is not set"
            }), 400
        
        # Get the target path from form data or query params
        target_path = request.form.get('path', '') or request.args.get('path', '')
        
        # Normalize the source root for comparison
        source_root_normalized = dynamic_source_root.rstrip('/')
        
        # Handle relative paths - if path doesn't start with source root, prepend it
        if target_path:
            # Check if path already starts with source root
            if target_path.startswith(source_root_normalized):
                # Path already includes source root, use as is
                pass
            else:
                # Check if path starts with a suffix of source root
                # e.g., if source is "Georgia 14/Pending Files" and path is "Pending Files/test"
                # we want to extract "test" and prepend full source root
                source_parts = source_root_normalized.rstrip('/').split('/')
                match_found = False
                
                # Check if path starts with any suffix of source root
                for i in range(len(source_parts)):
                    suffix = '/'.join(source_parts[i:])
                    if suffix and (target_path.startswith(suffix + '/') or target_path == suffix):
                        # Path starts with a suffix of source root
                        # Extract the remaining part after the suffix
                        remaining = target_path[len(suffix):].lstrip('/')
                        target_path = f"{dynamic_source_root}{remaining}"
                        match_found = True
                        break
                
                if not match_found:
                    # Path is completely relative to source root - prepend source root
                    target_path = f"{dynamic_source_root}{target_path.lstrip('/')}"
            
            # Ensure final path is within source root
            if not target_path.startswith(f'{source_root_normalized}/') and target_path != source_root_normalized:
                return jsonify({
                    "message": f"Upload only allowed within {source_root_normalized} directory",
                    "error": "Invalid path"
                }), 400
        else:
            # No path provided, use source root
            target_path = dynamic_source_root
        
        # Check if files were uploaded
        if 'files' not in request.files:
            return jsonify({
                "message": "No files provided",
                "error": "No files in request"
            }), 400
        
        files = request.files.getlist('files')
        
        if not files or all(file.filename == '' for file in files):
            return jsonify({
                "message": "No files selected",
                "error": "Empty file list"
            }), 400
        
        # Initialize GCS client for conflict detection
        storage_client = storage.Client()
        bucket = storage_client.bucket(FILE_MANAGEMENT_BUCKET_NAME)
        
        conflicts = []
        valid_files = []
        errors = []
        
        # Check for conflicts first
        for file in files:
            if file.filename == '':
                continue

            is_valid, validation_msg = validate_filename(file.filename)
            if not is_valid:
                errors.append(f"File '{file.filename}' is invalid: {validation_msg}")
                continue
                
            # Validate file type (PDF only)
            if not file.filename.lower().endswith('.pdf'):
                errors.append(f"File '{file.filename}' is not a PDF")
                continue
            
            # Validate file content type
            if file.content_type != 'application/pdf':
                errors.append(f"File '{file.filename}' does not have PDF content type")
                continue
            
            try:
                # Construct the full GCS path
                if target_path.endswith('/'):
                    gcs_path = f"{target_path}{file.filename}"
                else:
                    gcs_path = f"{target_path}/{file.filename}"
                
                # Check if file already exists
                existing_blob = bucket.blob(gcs_path)
                if existing_blob.exists():
                    # File exists - add to conflicts
                    try:
                        existing_blob.reload()  # Refresh metadata
                        conflicts.append({
                            "filename": file.filename,
                            "path": gcs_path,
                            "size": existing_blob.size,
                            "created_at": existing_blob.time_created.isoformat() if existing_blob.time_created else None,
                            "updated_at": existing_blob.updated.isoformat() if existing_blob.updated else None
                        })
                    except Exception as e:
                        # If we can't get metadata, still report conflict
                        conflicts.append({
                            "filename": file.filename,
                            "path": gcs_path,
                            "size": existing_blob.size or 0,
                            "created_at": None,
                            "updated_at": None
                        })
                else:
                    # File doesn't exist - add to valid files for upload
                    valid_files.append({
                        "file": file,
                        "gcs_path": gcs_path
                    })
            except Exception as e:
                errors.append(f"Error checking file '{file.filename}': {str(e)}")
        
        # If there are conflicts, return them without uploading
        if conflicts:
            return jsonify({
                "message": "File conflicts detected. Some files already exist in the target directory.",
                "conflicts": conflicts,
                "valid_files": len(valid_files),
                "total_attempted": len([f for f in files if f.filename != '']),
                "target_path": target_path,
                "errors": errors if errors else None,
                "total_errors": len(errors) if errors else 0
            }), 409  # Conflict status code
        
        # No conflicts - proceed with upload
        uploaded_files = []
        
        for file_info in valid_files:
            file = file_info["file"]
            gcs_path = file_info["gcs_path"]
            
            try:
                # Upload file to GCS using the custom function
                gcs_uri, blob_name, file_size, upload_error = upload_to_gcs(
                    file.stream,
                    file.filename,
                    file.content_type,
                    gcs_path
                )
                
                if upload_error is None:
                    # Determine folder path from target_path
                    folder_path = target_path if target_path.endswith('/') else target_path + '/'
                    if not folder_path:
                        folder_path = dynamic_source_root
                    
                    # Create file metadata in Firestore
                    file_id, metadata_error = file_management_model.create_file_metadata(
                        file_path=gcs_path,
                        file_name=file.filename,
                        folder_path=folder_path,
                        gcs_uri=gcs_uri,
                        file_size=file_size,
                        content_type=file.content_type,
                        user_email=user_id
                    )
                    
                    if metadata_error:
                        # Log warning but don't fail the request since file was uploaded to GCS
                        debug_warn(f"File uploaded to GCS but metadata creation failed: {metadata_error}")
                    
                    created_at = datetime.now(tz=timezone.utc).isoformat() if file_id else None
                    uploaded_files.append({
                        "filename": file.filename,
                        "path": gcs_path,
                        "gcs_uri": gcs_uri,
                        "blob_name": blob_name,
                        "size": file_size,
                        "file_id": file_id,
                        "created_at": created_at,
                        "created_date": created_at,
                        "updated_date": created_at
                    })
                    
                    # Log successful file upload to GCS
                    log_file_activity(
                        user_email=user_email,
                        activity_type=ActivityTypes.FILE_UPLOAD,
                        filename=file.filename,
                        file_info={
                            'gcs_path': gcs_path,
                            'file_size': file_size,
                            'upload_method': 'file_management',
                            'file_id': file_id,
                            'folder_path': folder_path
                        },
                        request_obj=request
                    )
                else:
                    errors.append(f"Failed to upload '{file.filename}': {upload_error}")
                    
            except Exception as e:
                errors.append(f"Error uploading '{file.filename}': {str(e)}")
        
        # Prepare response
        response_data = {
            "message": f"Upload completed. {len(uploaded_files)} files uploaded successfully",
            "uploaded_files": uploaded_files,
            "total_uploaded": len(uploaded_files),
            "total_attempted": len(valid_files),
            "target_path": target_path
        }
        
        if errors:
            response_data["errors"] = errors
            response_data["total_errors"] = len(errors)
        
        # Return appropriate status code
        if uploaded_files and not errors:
            return jsonify(response_data), 201  # Created
        elif uploaded_files and errors:
            return jsonify(response_data), 207  # Multi-Status (partial success)
        else:
            return jsonify({
                "message": "No files were uploaded successfully",
                "errors": errors,
                "total_errors": len(errors)
            }), 400
            
    except Exception as e:
        return jsonify({
            "message": "Internal server error during file upload",
            "error": str(e)
        }), 500


@gcs_bp.route('/resolve-upload-conflict', methods=['POST'])
@jwt_required()
def resolve_upload_conflict():
    """
    Handle user's decision on file conflicts (replace or keep existing file).
    
    Request Body:
    {
        "filePath": "path/to/existing/file.pdf",  // GCS path of the conflicting file
        "action": "replace" or "keep",           // User's choice
        "targetPath": "path/to/folder/"         // Path where file should be uploaded
    }
    
    Returns:
    - 200: If action is 'keep' or 'replace' successful
    - 400: If parameters are invalid
    - 409: If file is being replaced but upload fails
    - 500: Internal server error
    """
    try:
        user_id = get_jwt_identity()
        user_email = get_jwt().get('email', 'unknown_user')
        
        # Check upload permission
        has_permission, error_msg = check_file_permission('can_upload', user_id)
        if not has_permission:
            return jsonify({
                "message": error_msg,
                "error": "Permission denied"
            }), 403
        
        data = request.get_json(silent=True) or {}
        form_data = request.form or {}

        def get_param(*keys):
            for key in keys:
                if key in data and data[key] is not None:
                    return data[key]
                if key in form_data and form_data[key] is not None:
                    return form_data[key]
            return None

        file_path = get_param('filePath', 'file_path')
        action = get_param('action')
        target_path = get_param('targetPath', 'target_path')

        if not file_path or not action:
            return jsonify({
                "message": "Missing required parameters: filePath (or file_path) and action",
                "error": "Invalid request parameters"
            }), 400
        
        if action not in ['replace', 'keep']:
            return jsonify({
                "message": "action must be either 'replace' or 'keep'",
                "error": "Invalid action"
            }), 400
        
        # Initialize GCS client
        storage_client = storage.Client()
        bucket = storage_client.bucket(FILE_MANAGEMENT_BUCKET_NAME)
        
        if action == 'keep':
            # User chose to keep existing file - just return success
            return jsonify({
                "message": "Existing file has been kept",
                "action": "keep",
                "file_path": file_path,
                "target_path": target_path,
                "status": "success"
            }), 200
        
        elif action == 'replace':
            # User chose to replace file - check if file in request and upload
            if 'file' not in request.files:
                return jsonify({
                    "message": "No file provided for replacement",
                    "error": "No file in request"
                }), 400
            
            file = request.files['file']
            
            if file.filename == '':
                return jsonify({
                    "message": "File has no name",
                    "error": "Empty filename"
                }), 400
            
            # Validate file type (PDF only)
            if not file.filename.lower().endswith('.pdf'):
                return jsonify({
                    "message": f"File '{file.filename}' is not a PDF",
                    "error": "Invalid file type"
                }), 400
            
            # Validate file content type
            if file.content_type != 'application/pdf':
                return jsonify({
                    "message": f"File '{file.filename}' does not have PDF content type",
                    "error": "Invalid content type"
                }), 400
            
            try:
                # Delete existing file
                existing_blob = bucket.blob(file_path)
                if existing_blob.exists():
                    try:
                        existing_blob.delete()
                        debug_log(f"Deleted existing file: {file_path}")
                    except Exception as e:
                        debug_error(f"Failed to delete existing file {file_path}: {e}")
                        return jsonify({
                            "message": f"Failed to delete existing file: {str(e)}",
                            "error": "Delete operation failed"
                        }), 409
                
                # Upload new file
                gcs_uri, blob_name, file_size, upload_error = upload_to_gcs(
                    file.stream,
                    file.filename,
                    file.content_type,
                    file_path
                )
                
                if upload_error is not None:
                    debug_error(f"Failed to upload replacement file: {upload_error}")
                    return jsonify({
                        "message": f"Failed to upload replacement file: {upload_error}",
                        "error": "Upload failed"
                    }), 409
                
                # Update file metadata in Firestore
                # Extract folder path from file_path
                folder_path = '/'.join(file_path.split('/')[:-1]) + '/'
                file_name = file_path.split('/')[-1]
                
                file_id, metadata_error = file_management_model.create_file_metadata(
                    file_path=file_path,
                    file_name=file_name,
                    folder_path=folder_path,
                    gcs_uri=gcs_uri,
                    file_size=file_size,
                    content_type=file.content_type,
                    user_email=user_email
                )
                
                if metadata_error:
                    debug_warn(f"File replaced in GCS but metadata update failed: {metadata_error}")
                
                # Log file replacement activity
                log_file_activity(
                    user_email=user_email,
                    activity_type=ActivityTypes.FILE_UPLOAD,
                    filename=file_name,
                    file_info={
                        'gcs_path': file_path,
                        'file_size': file_size,
                        'operation': 'replace',
                        'file_id': file_id,
                        'folder_path': folder_path
                    },
                    request_obj=request
                )
                
                created_at = datetime.now(tz=timezone.utc).isoformat()
                
                return jsonify({
                    "message": "File successfully replaced",
                    "action": "replace",
                    "file_path": file_path,
                    "size": file_size,
                    "gcs_uri": gcs_uri,
                    "file_id": file_id,
                    "updated_at": created_at,
                    "status": "success"
                }), 200
            
            except Exception as e:
                debug_error(f"Error during file replacement: {str(e)}")
                return jsonify({
                    "message": f"Unexpected error during file replacement: {str(e)}",
                    "error": "Internal server error"
                }), 500
    
    except Exception as e:
        debug_error(f"Error in resolve_upload_conflict endpoint: {str(e)}")
        return jsonify({
            "message": "Internal server error",
            "error": str(e)
        }), 500

@gcs_bp.route('/preview/<path:file_path>', methods=['GET'])
@jwt_required()
def get_file_preview(file_path):
    """Return a preview URL that points to our proxy endpoint."""
    try:
        # Decode the file path
        decoded_path = urllib.parse.unquote(file_path)
        
        # Initialize GCS client
        client = storage.Client()
        bucket = client.bucket(FILE_MANAGEMENT_BUCKET_NAME)
        blob = bucket.blob(decoded_path)
        
        # Check if file exists
        if not blob.exists():
            return jsonify({
                "message": "File not found",
                "error": f"File '{decoded_path}' does not exist"
            }), 404
        
        # Get current user's token to include in proxy URL
        from flask import request
        auth_header = request.headers.get('Authorization', '')
        token = auth_header.replace('Bearer ', '') if auth_header.startswith('Bearer ') else ''
        
        # Build the proxy URL - use the same base URL that the frontend is using
        # Since the request comes from frontend, we can use the Origin header or build from request
        origin = request.headers.get('origin', 'http://localhost:3000')
        # backend_base = origin.replace(':3000', ':5001')  # Frontend on 3000, backend on 5001
        backend_base = os.getenv('BACKEND_API_URL')
        # Return full proxy URL with token parameter
        encoded_path = urllib.parse.quote(decoded_path, safe='')
        preview_url = f"{backend_base}{ROUTE_PREFIX}/backend/api/v1/gcs/proxy/{encoded_path}?token={token}"
        
        # Log file preview activity
        user_id = get_jwt_identity()
        user_email = get_jwt().get('email', 'unknown_user')
        log_file_activity(
            user_email=user_email,
            activity_type=ActivityTypes.FILE_PREVIEW,
            filename=decoded_path.split('/')[-1],
            file_info={'file_path': decoded_path, 'preview_method': 'iframe_proxy'},
            request_obj=request
        )
        
        return jsonify({
            "preview_url": preview_url,
            "filename": decoded_path.split('/')[-1],
            "expires_at": (datetime.utcnow() + timedelta(minutes=30)).isoformat() + "Z"
        })
        
    except Exception as e:
        return jsonify({
            "message": "Error generating preview URL",
            "error": str(e)
        }), 500

@gcs_bp.route('/proxy/<path:file_path>', methods=['GET'])
def proxy_file_content(file_path):
    """Proxy PDF content from GCS to avoid iframe restrictions."""
    try:
        # Get token from URL parameter
        token = request.args.get('token')
        if not token:
            return "Missing authentication token", 401
        
        # Verify JWT token manually
        from flask_jwt_extended import decode_token
        try:
            decode_token(token)
        except Exception as e:
            return "Invalid or expired token", 401
        
        # Decode the file path
        decoded_path = urllib.parse.unquote(file_path)
        
        # Initialize GCS client
        client = storage.Client()
        bucket = client.bucket(FILE_MANAGEMENT_BUCKET_NAME)
        blob = bucket.blob(decoded_path)
        
        # Check if file exists
        if not blob.exists():
            return "File not found", 404
        
        # Get file content
        file_content = blob.download_as_bytes()
        
        # Create response with proper headers for PDF viewing in iframe
        response = Response(
            file_content,
            mimetype='application/pdf',
            headers={
                'Content-Disposition': 'inline',
                'Content-Type': 'application/pdf',
                'Cache-Control': 'public, max-age=1800',
                'X-Frame-Options': 'SAMEORIGIN',
                'Content-Security-Policy': "frame-ancestors 'self'",
                'X-Content-Type-Options': 'nosniff'
            }
        )
        
        return response
        
    except Exception as e:
        return f"Error loading file: {str(e)}", 500


@gcs_bp.route('/rename-folder', methods=['PATCH'])
@jwt_required()
def rename_folder():
    """Rename a folder in GCS and update metadata."""
    user_id = get_jwt_identity()
    user_email = get_jwt().get('email', 'unknown_user')
    data = request.get_json()
    
    if not data or 'folderPath' not in data or 'newFolderName' not in data:
        return jsonify({"message": "Invalid rename request. 'folderPath' and 'newFolderName' are required"}), 400
    
    folder_path = data['folderPath']
    new_folder_name = data['newFolderName'].strip()
    
    if not new_folder_name:
        return jsonify({"message": "New folder name is required"}), 400
    
    # Check permission - renaming is only allowed in source panel
    has_permission, error_msg = check_file_permission('can_rename_source', user_id)
    if not has_permission:
        return jsonify({"message": error_msg}), 403
    
    # Validate folder name
    if '/' in new_folder_name or '\\' in new_folder_name:
        return jsonify({"message": "Folder name cannot contain slashes"}), 400
    
    try:
        # Ensure folder path ends with /
        if not folder_path.endswith('/'):
            folder_path += '/'
        
        # Extract parent path and construct new path
        parent_path = '/'.join(folder_path.rstrip('/').split('/')[:-1])
        if parent_path:
            parent_path += '/'
        
        new_folder_path = f"{parent_path}{new_folder_name}/"
        
        # Check if new folder path already exists
        storage_client = storage.Client()
        bucket = storage_client.bucket(FILE_MANAGEMENT_BUCKET_NAME)
        new_folder_blob = bucket.blob(new_folder_path)
        if new_folder_blob.exists():
            return jsonify({"message": f"Folder '{new_folder_name}' already exists"}), 400
        
        # Rename folder in GCS (move all files)
        old_folder_blob = bucket.blob(folder_path)
        if not old_folder_blob.exists():
            # Check if there are files in the folder
            blobs = list(storage_client.list_blobs(bucket, prefix=folder_path))
            if not blobs:
                return jsonify({"message": f"Folder '{folder_path}' not found"}), 404
        
        # Move all files in the folder
        blobs = list(storage_client.list_blobs(bucket, prefix=folder_path))
        for blob in blobs:
            if blob.name == folder_path:
                # This is the folder marker, skip it for now
                continue
            
            # Calculate new blob name
            relative_path = blob.name[len(folder_path):]
            new_blob_name = new_folder_path + relative_path
            
            # Copy blob to new location
            new_blob = bucket.blob(new_blob_name)
            new_blob.rewrite(blob)
            
            # Delete old blob
            blob.delete()
            
            # Update file metadata if it exists
            file_management_model.update_file_path(
                old_path=blob.name,
                new_path=new_blob_name,
                new_folder_path=new_folder_path
            )
        
        # Move folder marker
        if old_folder_blob.exists():
            new_folder_blob.upload_from_string('', content_type='application/x-directory')
            old_folder_blob.delete()
        
        # Update folder name and path in metadata (using old path to find the document)
        success, error = file_management_model.update_folder_name(
            folder_path=folder_path,
            new_folder_name=new_folder_name,
            user_email=user_email,
            new_folder_path=new_folder_path
        )
        
        if not success:
            debug_warn(f"Folder renamed in GCS but metadata update failed: {error}")
        
        # Log folder rename
        log_file_activity(
            user_email=user_email,
            activity_type=ActivityTypes.FOLDER_CREATE,  # Using FOLDER_CREATE as closest match
            filename=new_folder_name,
            file_info={
                'old_folder_path': folder_path,
                'new_folder_path': new_folder_path,
                'operation': 'rename'
            },
            request_obj=request
        )
        
        return jsonify({
            "message": f"Folder renamed successfully",
            "old_path": folder_path,
            "new_path": new_folder_path,
            "new_folder_name": new_folder_name
        }), 200
        
    except Exception as e:
        logger.error(f"Error renaming folder {folder_path}: {e}", exc_info=True)
        return jsonify({
            "message": "Error renaming folder",
            "error": str(e)
        }), 500


@gcs_bp.route('/folder/<path:folder_path>/metadata', methods=['GET'])
@jwt_required()
def get_folder_metadata_endpoint(folder_path):
    """Get metadata for a specific folder."""
    try:
        # Ensure folder path ends with /
        if not folder_path.endswith('/'):
            folder_path += '/'
        
        folder_data, error = file_management_model.get_folder_metadata(folder_path)
        
        if error:
            return jsonify({"message": error}), 404
        
        return jsonify(folder_data), 200
        
    except Exception as e:
        logger.error(f"Error getting folder metadata for {folder_path}: {e}", exc_info=True)
        return jsonify({
            "message": "Error getting folder metadata",
            "error": str(e)
        }), 500


@gcs_bp.route('/file/<path:file_path>/metadata', methods=['GET'])
@jwt_required()
def get_file_metadata_endpoint(file_path):
    """Get metadata for a specific file."""
    try:
        file_data, error = file_management_model.get_file_metadata(file_path)
        
        if error:
            return jsonify({"message": error}), 404
        
        return jsonify(file_data), 200
        
    except Exception as e:
        logger.error(f"Error getting file metadata for {file_path}: {e}", exc_info=True)
        return jsonify({
            "message": "Error getting file metadata",
            "error": str(e)
        }), 500


@gcs_bp.route('/folder/<path:folder_path>/files', methods=['GET'])
@jwt_required()
def get_files_in_folder(folder_path):
    """Get all files in a specific folder."""
    try:
        # Ensure folder path ends with /
        if not folder_path.endswith('/'):
            folder_path += '/'
        
        files_list, error = file_management_model.get_files_in_folder(folder_path)
        
        if error:
            return jsonify({"message": error}), 500
        
        return jsonify({
            "folder_path": folder_path,
            "files": files_list,
            "total_files": len(files_list)
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting files in folder {folder_path}: {e}", exc_info=True)
        return jsonify({
            "message": "Error getting files in folder",
            "error": str(e)
        }), 500


@gcs_bp.route('/delete-folder', methods=['DELETE'])
@jwt_required()
def delete_folder():
    """Delete a folder and all its contents from GCS and Firestore."""
    user_id = get_jwt_identity()
    user_email = get_jwt().get('email', 'unknown_user')
    data = request.get_json()
    
    if not data or 'folderPath' not in data:
        return jsonify({"message": "Invalid delete request. 'folderPath' is required"}), 400
    
    folder_path = data['folderPath']
    
    # Ensure folder path ends with /
    if not folder_path.endswith('/'):
        folder_path += '/'
    
    # Determine if this is source or destination based on path
    source_root = get_dynamic_source_path()
    is_source = folder_path.startswith(source_root) if source_root else False
    
    # Check permission based on source/destination
    permission_key = 'can_delete_source' if is_source else 'can_delete_destination'
    has_permission, error_msg = check_file_permission(permission_key, user_id)
    if not has_permission:
        return jsonify({"message": error_msg}), 403
    
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(FILE_MANAGEMENT_BUCKET_NAME)
        
        # List all blobs in the folder
        blobs = list(storage_client.list_blobs(bucket, prefix=folder_path))
        
        if not blobs:
            # Check if folder marker exists
            folder_blob = bucket.blob(folder_path)
            if not folder_blob.exists():
                return jsonify({"message": f"Folder '{folder_path}' not found"}), 404
        
        deleted_files = []
        deleted_folders = []
        errors = []
        
        # Delete all files and subfolders
        for blob in blobs:
            try:
                # Delete file metadata if it exists
                if not blob.name.endswith('/'):
                    # This is a file
                    file_management_model.delete_file_metadata(blob.name)
                    deleted_files.append(blob.name)
                else:
                    # This is a folder marker
                    deleted_folders.append(blob.name)
                
                # Delete from GCS
                blob.delete()
                
            except Exception as e:
                error_msg = f"Error deleting {blob.name}: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)
        
        # Delete folder metadata
        metadata_success, metadata_error = file_management_model.delete_folder_metadata(folder_path)
        if not metadata_success:
            logger.warning(f"Folder deleted from GCS but metadata deletion failed: {metadata_error}")
        
        # Also delete all file metadata in the folder (in case some weren't found by blob listing)
        deleted_metadata_count, _ = file_management_model.delete_all_files_metadata_in_folder(folder_path)
        
        # Log folder deletion
        log_file_activity(
            user_email=user_email,
            activity_type=ActivityTypes.FOLDER_CREATE,  # Using FOLDER_CREATE as closest match
            filename=folder_path.split('/')[-2] if folder_path.count('/') > 1 else folder_path,
            file_info={
                'folder_path': folder_path,
                'deleted_files_count': len(deleted_files),
                'deleted_folders_count': len(deleted_folders),
                'operation': 'delete'
            },
            request_obj=request
        )
        
        response_data = {
            "message": "Folder deleted successfully",
            "folder_path": folder_path,
            "deleted_files_count": len(deleted_files),
            "deleted_folders_count": len(deleted_folders),
            "deleted_metadata_count": deleted_metadata_count
        }
        
        if errors:
            response_data["errors"] = errors
            response_data["partial_success"] = True
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"Error deleting folder {folder_path}: {e}", exc_info=True)
        return jsonify({
            "message": "Error deleting folder",
            "error": str(e)
        }), 500


@gcs_bp.route('/delete-file', methods=['DELETE'])
@jwt_required()
def delete_file():
    """Delete a file from GCS and Firestore."""
    user_id = get_jwt_identity()
    user_email = get_jwt().get('email', 'unknown_user')
    data = request.get_json()
    
    if not data or 'filePath' not in data:
        return jsonify({"message": "Invalid delete request. 'filePath' is required"}), 400
    
    file_path = data['filePath']
    
    # Determine if this is source or destination based on path
    source_root = get_dynamic_source_path()
    is_source = file_path.startswith(source_root) if source_root else False
    
    # Check permission based on source/destination
    permission_key = 'can_delete_source' if is_source else 'can_delete_destination'
    has_permission, error_msg = check_file_permission(permission_key, user_id)
    if not has_permission:
        return jsonify({"message": error_msg}), 403
    
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(FILE_MANAGEMENT_BUCKET_NAME)
        blob = bucket.blob(file_path)
        
        if not blob.exists():
            return jsonify({"message": f"File '{file_path}' not found"}), 404
        
        # Delete file from GCS
        blob.delete()
        
        # Delete file metadata
        metadata_success, metadata_error = file_management_model.delete_file_metadata(file_path)
        if not metadata_success:
            logger.warning(f"File deleted from GCS but metadata deletion failed: {metadata_error}")
        
        # Extract filename for logging
        file_name = file_path.split('/')[-1]
        
        # Log file deletion
        log_file_activity(
            user_email=user_email,
            activity_type=ActivityTypes.FILE_UPLOAD,  # Using FILE_UPLOAD as closest match
            filename=file_name,
            file_info={
                'file_path': file_path,
                'operation': 'delete'
            },
            request_obj=request
        )
        
        return jsonify({
            "message": "File deleted successfully",
            "file_path": file_path,
            "file_name": file_name
        }), 200
        
    except Exception as e:
        logger.error(f"Error deleting file {file_path}: {e}", exc_info=True)
        return jsonify({
            "message": "Error deleting file",
            "error": str(e)
        }), 500


@gcs_bp.route('/rename-file', methods=['PATCH'])
@jwt_required()
def rename_file():
    """Rename a file in GCS and update metadata."""
    user_id = get_jwt_identity()
    user_email = get_jwt().get('email', 'unknown_user')
    data = request.get_json()
    
    if not data or 'filePath' not in data or 'newFileName' not in data:
        return jsonify({"message": "Invalid rename request. 'filePath' and 'newFileName' are required"}), 400
    
    file_path = data['filePath']
    new_file_name = data['newFileName'].strip()
    
    if not new_file_name:
        return jsonify({"message": "New file name is required"}), 400

    is_valid, validation_msg = validate_filename(new_file_name)
    if not is_valid:
        return jsonify({"message": validation_msg}), 400
    
    # Check permission - renaming is only allowed in source panel
    has_permission, error_msg = check_file_permission('can_rename_source', user_id)
    if not has_permission:
        return jsonify({"message": error_msg}), 403
    
    # Validate file name
    if '/' in new_file_name or '\\' in new_file_name:
        return jsonify({"message": "File name cannot contain slashes"}), 400
    
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(FILE_MANAGEMENT_BUCKET_NAME)
        old_blob = bucket.blob(file_path)
        
        # Check if file exists
        if not old_blob.exists():
            return jsonify({"message": f"File '{file_path}' not found"}), 404
        
        # Extract folder path and construct new file path
        folder_path = '/'.join(file_path.split('/')[:-1])
        if folder_path:
            folder_path += '/'
        
        new_file_path = f"{folder_path}{new_file_name}"
        
        # Check if new file path already exists
        new_blob = bucket.blob(new_file_path)
        if new_blob.exists():
            return jsonify({"message": f"File '{new_file_name}' already exists in this folder"}), 400
        
        # Rename file in GCS (copy to new location and delete old)
        new_blob.rewrite(old_blob)
        old_blob.delete()
        
        # Update file metadata (file_path, file_name, and folder_path)
        metadata_success, metadata_error = file_management_model.update_file_name_and_path(
            old_path=file_path,
            new_path=new_file_path,
            new_file_name=new_file_name,
            new_folder_path=folder_path
        )
        
        if not metadata_success:
            logger.warning(f"File renamed in GCS but metadata update failed: {metadata_error}")
        
        # Log file rename
        log_file_activity(
            user_email=user_email,
            activity_type=ActivityTypes.FILE_UPLOAD,  # Using FILE_UPLOAD as closest match
            filename=new_file_name,
            file_info={
                'old_file_path': file_path,
                'new_file_path': new_file_path,
                'old_file_name': file_path.split('/')[-1],
                'new_file_name': new_file_name,
                'operation': 'rename'
            },
            request_obj=request
        )
        
        return jsonify({
            "message": "File renamed successfully",
            "old_path": file_path,
            "new_path": new_file_path,
            "old_file_name": file_path.split('/')[-1],
            "new_file_name": new_file_name
        }), 200
        
    except Exception as e:
        logger.error(f"Error renaming file {file_path}: {e}", exc_info=True)
        return jsonify({
            "message": "Error renaming file",
            "error": str(e)
        }), 500


@gcs_bp.route('/bulk-delete-files', methods=['DELETE'])
@jwt_required()
def bulk_delete_files():
    """Delete multiple files from GCS and Firestore in a single API call."""
    user_id = get_jwt_identity()
    user_email = get_jwt().get('email', 'unknown_user')
    data = request.get_json()
    
    if not data or 'filePaths' not in data:
        return jsonify({"message": "Invalid bulk delete request. 'filePaths' array is required"}), 400
    
    file_paths = data['filePaths']
    
    if not isinstance(file_paths, list):
        return jsonify({"message": "'filePaths' must be an array"}), 400
    
    if not file_paths:
        return jsonify({"message": "'filePaths' array cannot be empty"}), 400
    
    # Check permission - need to check both source and destination delete permissions
    # For simplicity, we require both if files are from different locations
    # Check first file to determine location, but a strict implementation would check each
    source_root = get_dynamic_source_path()
    
    # Check if user has permission to delete from the locations of the files
    has_source_permission, _ = check_file_permission('can_delete_source', user_id)
    has_dest_permission, _ = check_file_permission('can_delete_destination', user_id)
    
    # Check if any file is in source and any is in destination
    has_source_files = any(fp.startswith(source_root) if source_root else False for fp in file_paths)
    has_dest_files = any(not fp.startswith(source_root) if source_root else True for fp in file_paths)
    
    if has_source_files and not has_source_permission:
        return jsonify({"message": "You don't have permission to delete files from source"}), 403
    if has_dest_files and not has_dest_permission:
        return jsonify({"message": "You don't have permission to delete files from destination"}), 403
    
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(FILE_MANAGEMENT_BUCKET_NAME)
        
        results = {
            "successful": [],
            "failed": [],
            "total_requested": len(file_paths),
            "total_successful": 0,
            "total_failed": 0
        }
        
        for file_path in file_paths:
            if not isinstance(file_path, str) or not file_path.strip():
                results["failed"].append({
                    "file_path": file_path,
                    "error": "Invalid file path"
                })
                results["total_failed"] += 1
                continue
            
            try:
                blob = bucket.blob(file_path)
                
                if not blob.exists():
                    results["failed"].append({
                        "file_path": file_path,
                        "error": "File not found"
                    })
                    results["total_failed"] += 1
                    continue
                
                # Delete file from GCS
                blob.delete()
                
                # Delete file metadata
                metadata_success, metadata_error = file_management_model.delete_file_metadata(file_path)
                if not metadata_success:
                    logger.warning(f"File {file_path} deleted from GCS but metadata deletion failed: {metadata_error}")
                
                # Extract filename for logging
                file_name = file_path.split('/')[-1]
                
                # Log individual file deletion
                log_file_activity(
                    user_email=user_email,
                    activity_type=ActivityTypes.FILE_DELETE,
                    filename=file_name,
                    file_info={
                        'file_path': file_path,
                        'operation': 'bulk_delete'
                    },
                    request_obj=request
                )
                
                results["successful"].append({
                    "file_path": file_path,
                    "file_name": file_name
                })
                results["total_successful"] += 1
                
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Error deleting file {file_path}: {e}", exc_info=True)
                results["failed"].append({
                    "file_path": file_path,
                    "error": error_msg
                })
                results["total_failed"] += 1
        
        # Determine response status code
        if results["total_successful"] == 0:
            status_code = 400  # All failed
        elif results["total_failed"] == 0:
            status_code = 200  # All succeeded
        else:
            status_code = 207  # Partial success (Multi-Status)
        
        response_data = {
            "message": f"Bulk delete completed. {results['total_successful']} files deleted successfully, {results['total_failed']} failed",
            "total_requested": results["total_requested"],
            "total_successful": results["total_successful"],
            "total_failed": results["total_failed"],
            "successful": results["successful"],
            "failed": results["failed"]
        }
        
        return jsonify(response_data), status_code
        
    except Exception as e:
        logger.error(f"Error in bulk delete files: {e}", exc_info=True)
        return jsonify({
            "message": "Error processing bulk delete request",
            "error": str(e)
        }), 500


@gcs_bp.route('/bulk-delete-folders', methods=['DELETE'])
@jwt_required()
def bulk_delete_folders():
    """Delete multiple folders and all their contents from GCS and Firestore in a single API call."""
    user_id = get_jwt_identity()
    user_email = get_jwt().get('email', 'unknown_user')
    data = request.get_json()
    
    if not data or 'folderPaths' not in data:
        return jsonify({"message": "Invalid bulk delete request. 'folderPaths' array is required"}), 400
    
    folder_paths = data['folderPaths']
    
    if not isinstance(folder_paths, list):
        return jsonify({"message": "'folderPaths' must be an array"}), 400
    
    if not folder_paths:
        return jsonify({"message": "'folderPaths' array cannot be empty"}), 400
    
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(FILE_MANAGEMENT_BUCKET_NAME)
        
        results = {
            "successful": [],
            "failed": [],
            "total_requested": len(folder_paths),
            "total_successful": 0,
            "total_failed": 0
        }
        
        for folder_path in folder_paths:
            if not isinstance(folder_path, str) or not folder_path.strip():
                results["failed"].append({
                    "folder_path": folder_path,
                    "error": "Invalid folder path"
                })
                results["total_failed"] += 1
                continue
            
            # Ensure folder path ends with /
            if not folder_path.endswith('/'):
                folder_path += '/'
            
            try:
                # List all blobs in the folder
                blobs = list(storage_client.list_blobs(bucket, prefix=folder_path))
                
                # Check if folder exists
                if not blobs:
                    # Check if folder marker exists
                    folder_blob = bucket.blob(folder_path)
                    if not folder_blob.exists():
                        results["failed"].append({
                            "folder_path": folder_path,
                            "error": "Folder not found"
                        })
                        results["total_failed"] += 1
                        continue
                
                deleted_files = []
                deleted_folders = []
                errors = []
                
                # Delete all files and subfolders
                for blob in blobs:
                    try:
                        # Delete file metadata if it exists
                        if not blob.name.endswith('/'):
                            # This is a file
                            file_management_model.delete_file_metadata(blob.name)
                            deleted_files.append(blob.name)
                        else:
                            # This is a folder marker
                            deleted_folders.append(blob.name)
                        
                        # Delete from GCS
                        blob.delete()
                        
                    except Exception as e:
                        error_msg = f"Error deleting {blob.name}: {str(e)}"
                        logger.error(error_msg)
                        errors.append(error_msg)
                
                # Delete folder metadata
                metadata_success, metadata_error = file_management_model.delete_folder_metadata(folder_path)
                if not metadata_success:
                    logger.warning(f"Folder {folder_path} deleted from GCS but metadata deletion failed: {metadata_error}")
                
                # Also delete all file metadata in the folder (in case some weren't found by blob listing)
                deleted_metadata_count, _ = file_management_model.delete_all_files_metadata_in_folder(folder_path)
                
                # Extract folder name for logging
                folder_name = folder_path.split('/')[-2] if folder_path.count('/') > 1 else folder_path
                
                # Log folder deletion
                log_file_activity(
                    user_email=user_email,
                    activity_type=ActivityTypes.FOLDER_DELETE,
                    filename=folder_name,
                    file_info={
                        'folder_path': folder_path,
                        'deleted_files_count': len(deleted_files),
                        'deleted_folders_count': len(deleted_folders),
                        'operation': 'bulk_delete'
                    },
                    request_obj=request
                )
                
                result_item = {
                    "folder_path": folder_path,
                    "folder_name": folder_name,
                    "deleted_files_count": len(deleted_files),
                    "deleted_folders_count": len(deleted_folders),
                    "deleted_metadata_count": deleted_metadata_count
                }
                
                if errors:
                    result_item["errors"] = errors
                    result_item["partial_success"] = True
                
                results["successful"].append(result_item)
                results["total_successful"] += 1
                
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Error deleting folder {folder_path}: {e}", exc_info=True)
                results["failed"].append({
                    "folder_path": folder_path,
                    "error": error_msg
                })
                results["total_failed"] += 1
        
        # Determine response status code
        if results["total_successful"] == 0:
            status_code = 400  # All failed
        elif results["total_failed"] == 0:
            status_code = 200  # All succeeded
        else:
            status_code = 207  # Partial success (Multi-Status)
        
        response_data = {
            "message": f"Bulk delete completed. {results['total_successful']} folders deleted successfully, {results['total_failed']} failed",
            "total_requested": results["total_requested"],
            "total_successful": results["total_successful"],
            "total_failed": results["total_failed"],
            "successful": results["successful"],
            "failed": results["failed"]
        }
        
        return jsonify(response_data), status_code
        
    except Exception as e:
        logger.error(f"Error in bulk delete folders: {e}", exc_info=True)
        return jsonify({
            "message": "Error processing bulk delete request",
            "error": str(e)
        }), 500


@gcs_bp.route('/list-root', methods=['GET'])
@jwt_required()
def list_root_directory():
    """Get all files and folders in the root directory."""
    try:
        page = int(request.args.get('page', 1))
        page_size = int(request.args.get('page_size', 50))
        
        # Validate pagination parameters
        if page < 1:
            page = 1
        if page_size < 1 or page_size > 200:
            page_size = 50
        
        # List root directory (empty prefix)
        contents, error = gcs_service.list_gcs_directory(
            FILE_MANAGEMENT_BUCKET_NAME,
            prefix='',
            delimiter='/',
            page=page,
            page_size=page_size
        )
        
        if error:
            return jsonify({
                "message": "Error listing root directory",
                "error": error
            }), 500
        
        return jsonify({
            "path": "",
            "folders": contents.get("folders", []),
            "files": contents.get("files", []),
            "pagination": contents.get("pagination", {})
        }), 200
        
    except Exception as e:
        logger.error(f"Error listing root directory: {e}", exc_info=True)
        return jsonify({
            "message": "Error listing root directory",
            "error": str(e)
        }), 500


def get_path_depth(path_str):
    """Calculate the depth of a path by counting non-empty segments."""
    if not path_str or path_str == '/':
        return 0
    # Remove leading/trailing slashes and split
    normalized = path_str.strip('/')
    if not normalized:
        return 0
    return len([p for p in normalized.split('/') if p])


@gcs_bp.route('/list-path', methods=['GET'])
@jwt_required()
def list_path_directory():
    """Get all files and folders in a specific path and all parent paths up to root."""
    try:
        path = request.args.get('path', '')
        
        if not path:
            return jsonify({
                "message": "Path parameter is required. Use /list-root for root directory."
            }), 400
        
        page = int(request.args.get('page', 1))
        page_size = int(request.args.get('page_size', 50))
        
        # Validate pagination parameters
        if page < 1:
            page = 1
        if page_size < 1 or page_size > 200:
            page_size = 50
        
        # Strip leading slashes - GCS paths don't have leading slashes
        path = path.lstrip('/')
        
        # Ensure path ends with / for directory listing
        if not path.endswith('/'):
            path += '/'
        
        logger.info(f"Listing directory with path: '{path}' (after normalization)")
        debug_log(f"Listing directory with path: '{path}'")
        
        # Determine the root path and its depth based on whether we're in source or destination
        # Source root: GCS_SOURCE_ROOT (e.g., "Georgia 14/Pending Files/")
        # Destination root: First part of source root (e.g., "Georgia 14/")
        source_root_normalized = GCS_SOURCE_ROOT.rstrip('/') + '/'
        source_parts = GCS_SOURCE_ROOT.rstrip('/').split('/')
        destination_root = f"{source_parts[0]}/" if len(source_parts) > 0 else ""
        
        # Determine which root we're working with
        if path.startswith(source_root_normalized):
            # We're in the source root area
            root_path = source_root_normalized
            root_depth = get_path_depth(root_path)
        elif destination_root and path.startswith(destination_root):
            # We're in the destination root area
            root_path = destination_root
            root_depth = get_path_depth(root_path)
        else:
            # Fallback: use current path as root
            root_path = path
            root_depth = get_path_depth(root_path)
        
        # Generate list of paths to query: root path and all parent paths up to current path
        paths_to_query = []
        
        # If current path is the root, only query root
        if path == root_path:
            paths_to_query = [root_path]
        else:
            # Build all parent paths from root to current path
            # Split path into segments
            path_segments = path.rstrip('/').split('/')
            root_segments = root_path.rstrip('/').split('/')
            
            # Build paths incrementally from root to current
            current_build = root_path
            paths_to_query.append(current_build)
            
            # Add each intermediate path
            for i in range(len(root_segments), len(path_segments)):
                if path_segments[i]:  # Skip empty segments
                    current_build = current_build + path_segments[i] + '/'
                    paths_to_query.append(current_build)
        
        # Collect all folders and files from all paths
        all_folders = {}
        all_files = {}
        last_contents = None
        
        # Query each path and collect results
        for query_path in paths_to_query:
            contents, error = gcs_service.list_gcs_directory(
                FILE_MANAGEMENT_BUCKET_NAME,
                prefix=query_path,
                delimiter='/',
                page=page,
                page_size=page_size
            )
            
            if error:
                logger.warning(f"Error listing directory at path: {query_path}, error: {error}")
                continue
            
            # Store last contents for pagination info
            last_contents = contents
            
            # Collect folders (use path as key to avoid duplicates)
            for folder in contents.get("folders", []):
                folder_path = folder.get("path", folder.get("name", ""))
                # Only include folders that are direct children of the query_path
                if folder_path.startswith(query_path):
                    # Check if it's a direct child (not deeper)
                    relative_path = folder_path[len(query_path):]
                    if '/' not in relative_path.rstrip('/'):
                        all_folders[folder_path] = folder
            
            # Collect files (use path as key to avoid duplicates)
            for file in contents.get("files", []):
                file_path = file.get("path", file.get("name", ""))
                # Only include files that are direct children of the query_path
                if file_path.startswith(query_path):
                    # Check if it's a direct child (not in a subfolder)
                    relative_path = file_path[len(query_path):]
                    if '/' not in relative_path:
                        all_files[file_path] = file
        
        logger.info(f"Found {len(all_folders)} folders and {len(all_files)} files across all paths")
        
        # Build tree structure from folders and files
        # Create a map of folder paths to their tree nodes
        folder_nodes = {}
        
        # First, create all folder nodes
        for folder_path, folder in all_folders.items():
            folder_name = folder_path.rstrip('/').split('/')[-1] if folder_path else ""
            folder_depth = get_path_depth(folder_path)
            folder_type = "Folder" if (folder_depth - root_depth == 1) else "Sub Folder"
            
            folder_nodes[folder_path] = {
                "id": folder_path,
                "name": folder_name,
                "type": folder_type,
                "created_date": folder.get("created_date"),
                "path": folder_path,
                "children": []
            }
        
        # Organize folders into tree structure
        # Sort folders by depth (shallow to deep) to build tree correctly
        sorted_folders = sorted(all_folders.keys(), key=lambda p: (get_path_depth(p), p))
        
        # Build parent-child relationships
        root_items = []
        for folder_path in sorted_folders:
            folder_node = folder_nodes[folder_path]
            
            # Find parent folder path
            # Get the parent by removing the last segment
            path_parts = folder_path.rstrip('/').split('/')
            if len(path_parts) > len(root_path.rstrip('/').split('/')):
                # This folder has a parent (it's deeper than root)
                parent_path = '/'.join(path_parts[:-1]) + '/'
                
                if parent_path in folder_nodes:
                    # Add to parent's children
                    folder_nodes[parent_path]["children"].append(folder_node)
                else:
                    # Parent folder not in our collection, treat as root-level
                    root_items.append(folder_node)
            else:
                # This is a root-level folder (direct child of root_path)
                root_items.append(folder_node)
        
        # Add files to their parent folders
        for file_path, file in all_files.items():
            file_name = file_path.split('/')[-1] if file_path else ""
            
            file_item = {
                "id": file_path,
                "name": file_name,
                "type": "File",
                "created_date": file.get("created_date"),
                "path": file_path,
                "metadata": {
                    "updatedAt": file.get("updatedAt"),
                    "contentType": file.get("contentType")
                }
            }
            
            # Find the parent folder for this file
            # Get the directory path (parent folder) - where the file is located
            file_dir = '/'.join(file_path.split('/')[:-1]) + '/' if '/' in file_path else root_path
            
            # Find the closest parent folder that exists in our folder_nodes
            parent_folder = None
            current_dir = file_dir
            
            # Check if the file's directory is one of the folders we collected
            if current_dir in folder_nodes:
                parent_folder = folder_nodes[current_dir]
            else:
                # Try to find a parent folder by going up the directory tree
                # But only check folders that are in our collected folders
                path_parts = current_dir.rstrip('/').split('/')
                for i in range(len(path_parts), 0, -1):
                    check_dir = '/'.join(path_parts[:i]) + '/'
                    if check_dir in folder_nodes:
                        parent_folder = folder_nodes[check_dir]
                        break
            
            if parent_folder:
                # Add file to parent folder's children
                parent_folder["children"].append(file_item)
            else:
                # File is at root level (not in any collected folder), add to root items
                root_items.append(file_item)
        
        # Sort root items: folders first, then files, both alphabetically
        root_items.sort(key=lambda x: (x.get("type") != "Folder" and x.get("type") != "Sub Folder", x.get("name", "").lower()))
        
        # Sort children within each folder: folders first, then files
        def sort_children(node):
            if "children" in node:
                node["children"].sort(key=lambda x: (x.get("type") != "Folder" and x.get("type") != "Sub Folder", x.get("name", "").lower()))
                # Recursively sort nested children
                for child in node["children"]:
                    if "children" in child:
                        sort_children(child)
        
        for item in root_items:
            sort_children(item)
        
        return jsonify({
            "path": path,
            "items": root_items,
            "pagination": last_contents.get("pagination", {}) if last_contents else {}
        }), 200
        
    except Exception as e:
        logger.error(f"Error listing directory at path {path}: {e}", exc_info=True)
        return jsonify({
            "message": "Error listing directory",
            "error": str(e)
        }), 500


@gcs_bp.route('/list-destination-path', methods=['GET'])
@jwt_required()
def list_destination_path_directory():
    """Get all files and folders in a specific destination path and all parent paths up to root. Root is 'Georgia 14/' and excludes 'Pending Files/' folder."""
    path = ''
    try:
        # Determine destination root (first part of source root, e.g., "Georgia 14/")
        source_parts = GCS_SOURCE_ROOT.rstrip('/').split('/')
        destination_root = f"{source_parts[0]}/" if len(source_parts) > 0 else ""
        source_root_normalized = GCS_SOURCE_ROOT.rstrip('/') + '/'
        
        path = request.args.get('path', '')
        
        # If no path provided, use destination root
        if not path:
            path = destination_root
        else:
            # Ensure path starts with destination root
            path = path.lstrip('/')
            if not path.startswith(destination_root):
                path = destination_root + path.lstrip('/')
        
        page = int(request.args.get('page', 1))
        page_size = int(request.args.get('page_size', 50))
        
        # Validate pagination parameters
        if page < 1:
            page = 1
        if page_size < 1 or page_size > 200:
            page_size = 50
        
        # Ensure path ends with / for directory listing
        if not path.endswith('/'):
            path += '/'
        
        logger.info(f"Listing destination directory with path: '{path}' (after normalization)")
        debug_log(f"Listing destination directory with path: '{path}'")
        
        root_path = destination_root
        root_depth = get_path_depth(root_path)
        
        # Generate list of paths to query: root path and all parent paths up to current path
        paths_to_query = []
        
        # If current path is the root, only query root
        if path == root_path:
            paths_to_query = [root_path]
        else:
            # Build all parent paths from root to current path
            # Split path into segments
            path_segments = path.rstrip('/').split('/')
            root_segments = root_path.rstrip('/').split('/')
            
            # Build paths incrementally from root to current
            current_build = root_path
            paths_to_query.append(current_build)
            
            # Add each intermediate path
            for i in range(len(root_segments), len(path_segments)):
                if path_segments[i]:  # Skip empty segments
                    current_build = current_build + path_segments[i] + '/'
                    paths_to_query.append(current_build)
        
        # Collect all folders and files from all paths
        all_folders = {}
        all_files = {}
        last_contents = None
        
        # Query each path and collect results
        for query_path in paths_to_query:
            contents, error = gcs_service.list_gcs_directory(
                FILE_MANAGEMENT_BUCKET_NAME,
                prefix=query_path,
                delimiter='/',
                page=page,
                page_size=page_size
            )
            
            if error:
                logger.warning(f"Error listing directory at path: {query_path}, error: {error}")
                continue
            
            # Store last contents for pagination info
            last_contents = contents
            
            # Collect folders (use path as key to avoid duplicates)
            for folder in contents.get("folders", []):
                folder_path = folder.get("path", folder.get("name", ""))
                # Exclude "Pending Files/" folder
                if folder_path != source_root_normalized and not folder_path.startswith(source_root_normalized):
                    # Only include folders that are direct children of the query_path
                    if folder_path.startswith(query_path):
                        # Check if it's a direct child (not deeper)
                        relative_path = folder_path[len(query_path):]
                        if '/' not in relative_path.rstrip('/'):
                            all_folders[folder_path] = folder
            
            # Collect files (use path as key to avoid duplicates)
            for file in contents.get("files", []):
                file_path = file.get("path", file.get("name", ""))
                # Only include files that are direct children of the query_path
                if file_path.startswith(query_path):
                    # Check if it's a direct child (not in a subfolder)
                    relative_path = file_path[len(query_path):]
                    if '/' not in relative_path:
                        all_files[file_path] = file
        
        logger.info(f"Found {len(all_folders)} folders and {len(all_files)} files across all paths")
        
        # Build tree structure from folders and files
        # Create a map of folder paths to their tree nodes
        folder_nodes = {}
        
        # First, create all folder nodes
        for folder_path, folder in all_folders.items():
            folder_name = folder_path.rstrip('/').split('/')[-1] if folder_path else ""
            folder_depth = get_path_depth(folder_path)
            folder_type = "Folder" if (folder_depth - root_depth == 1) else "Sub Folder"
            
            folder_nodes[folder_path] = {
                "id": folder_path,
                "name": folder_name,
                "type": folder_type,
                "created_date": folder.get("created_date"),
                "path": folder_path,
                "children": []
            }
        
        # Organize folders into tree structure
        # Sort folders by depth (shallow to deep) to build tree correctly
        sorted_folders = sorted(all_folders.keys(), key=lambda p: (get_path_depth(p), p))
        
        # Build parent-child relationships
        root_items = []
        for folder_path in sorted_folders:
            folder_node = folder_nodes[folder_path]
            
            # Find parent folder path
            # Get the parent by removing the last segment
            path_parts = folder_path.rstrip('/').split('/')
            if len(path_parts) > len(root_path.rstrip('/').split('/')):
                # This folder has a parent (it's deeper than root)
                parent_path = '/'.join(path_parts[:-1]) + '/'
                
                if parent_path in folder_nodes:
                    # Add to parent's children
                    folder_nodes[parent_path]["children"].append(folder_node)
                else:
                    # Parent folder not in our collection, treat as root-level
                    root_items.append(folder_node)
            else:
                # This is a root-level folder (direct child of root_path)
                root_items.append(folder_node)
        
        # Add files to their parent folders
        for file_path, file in all_files.items():
            file_name = file_path.split('/')[-1] if file_path else ""
            
            file_item = {
                "id": file_path,
                "name": file_name,
                "type": "File",
                "created_date": file.get("created_date"),
                "path": file_path,
                "metadata": {
                    "updatedAt": file.get("updatedAt"),
                    "contentType": file.get("contentType")
                }
            }
            
            # Find the parent folder for this file
            # Get the directory path (parent folder) - where the file is located
            file_dir = '/'.join(file_path.split('/')[:-1]) + '/' if '/' in file_path else root_path
            
            # Find the closest parent folder that exists in our folder_nodes
            parent_folder = None
            current_dir = file_dir
            
            # Check if the file's directory is one of the folders we collected
            if current_dir in folder_nodes:
                parent_folder = folder_nodes[current_dir]
            else:
                # Try to find a parent folder by going up the directory tree
                # But only check folders that are in our collected folders
                path_parts = current_dir.rstrip('/').split('/')
                for i in range(len(path_parts), 0, -1):
                    check_dir = '/'.join(path_parts[:i]) + '/'
                    if check_dir in folder_nodes:
                        parent_folder = folder_nodes[check_dir]
                        break
            
            if parent_folder:
                # Add file to parent folder's children
                parent_folder["children"].append(file_item)
            else:
                # File is at root level (not in any collected folder), add to root items
                root_items.append(file_item)
        
        # Sort root items: folders first, then files, both alphabetically
        root_items.sort(key=lambda x: (x.get("type") != "Folder" and x.get("type") != "Sub Folder", x.get("name", "").lower()))
        
        # Sort children within each folder: folders first, then files
        def sort_children(node):
            if "children" in node:
                node["children"].sort(key=lambda x: (x.get("type") != "Folder" and x.get("type") != "Sub Folder", x.get("name", "").lower()))
                # Recursively sort nested children
                for child in node["children"]:
                    if "children" in child:
                        sort_children(child)
        
        for item in root_items:
            sort_children(item)
        
        return jsonify({
            "path": path,
            "items": root_items,
            "pagination": last_contents.get("pagination", {}) if last_contents else {}
        }), 200
        
    except Exception as e:
        logger.error(f"Error listing destination directory at path {path}: {e}", exc_info=True)
        return jsonify({
            "message": "Error listing destination directory",
            "error": str(e)
        }), 500


def transform_tree_to_hierarchical(node, depth=0):
    """
    Transform the tree structure to match the hierarchical format with id, name, type, path, and children.
    Type mapping:
    - folders: "folder"
    - files: "file"
    Preserves metadata fields: created_date, updated_date, size
    """
    # Get the path from the node - use it as the id
    node_path = node.get("path", "")
    node_name = node.get("name", "")
    
    if node.get("type") == "file":
        # For files, use the full path as id
        file_id = node_path if node_path else node_name
        file_item = {
            "id": file_id,
            "name": node_name,
            "type": "file",
            "path": node_path
        }
        # Preserve metadata fields if they exist
        if "size" in node:
            file_item["size"] = node["size"]
        if "created_date" in node:
            file_item["created_date"] = node["created_date"]
        if "updated_date" in node:
            file_item["updated_date"] = node["updated_date"]
        return file_item
    
    # For folders, use "folder" as type
    # Use the path as the id (with slashes)
    folder_id = node_path if node_path else node_name
    
    # Transform children
    children = []
    for child in node.get("children", []):
        transformed_child = transform_tree_to_hierarchical(child, depth + 1)
        if transformed_child:
            children.append(transformed_child)
    
    # Folders always have children array (even if empty)
    folder_item = {
        "id": folder_id,
        "name": node_name,
        "type": "folder",
        "path": node_path,
        "children": children
    }
    # Preserve metadata fields if they exist
    if "created_date" in node:
        folder_item["created_date"] = node["created_date"]
    if "updated_date" in node:
        folder_item["updated_date"] = node["updated_date"]
    
    return folder_item


def transform_to_list_with_subfolders_format(node, depth=0):
    """
    Transform the tree structure to match the exact format required for /list-with-subfolders.
    Returns items with: checked, id, name, type, created_date, children, metadata (for files)
    Type mapping:
    - top-level folders: "folder"
    - nested folders: "sub folder"
    - files: "file"
    """
    node_path = node.get("path", "")
    node_name = node.get("name", "")
    
    # Ensure folder paths end with /
    if node.get("type") == "folder" and node_path and not node_path.endswith('/'):
        node_path = node_path + '/'
    
    if node.get("type") == "file":
        # For files
        file_item = {
            "checked": False,
            "id": node_path if node_path else node_name,
            "path": node_path if node_path else node_name,
            "name": node_name,
            "type": "file",
            "created_date": node.get("created_date"),
            "metadata": {
                "size": node.get("size", 0),
                "updatedAt": node.get("updatedAt"),
                "contentType": node.get("contentType")
            }
        }
        return file_item
    
    # For folders, determine if it's a top-level folder or sub folder
    folder_type = "folder" if depth == 0 else "sub folder"
    
    # Transform children recursively
    children = []
    for child in node.get("children", []):
        transformed_child = transform_to_list_with_subfolders_format(child, depth + 1)
        if transformed_child:
            children.append(transformed_child)
    
    folder_item = {
        "checked": False,
        "id": node_path if node_path else node_name,
        "path": node_path if node_path else node_name,
        "name": node_name,
        "type": folder_type,
        "created_date": node.get("created_date"),
        "children": children
    }
    
    return folder_item


@gcs_bp.route('/list-with-subfolders', methods=['GET'])
@jwt_required()
def list_with_subfolders():
    """Get all files and folders in a path including all subfolders and their contents recursively.
    Supports optional search filter to find files/folders by name and pagination.
    """
    try:
        path = request.args.get('path', '')
        search_term = request.args.get('search', '').strip()  # Add search parameter
        page = int(request.args.get('page', 1))
        page_size = int(request.args.get('page_size', 50))
        
        # Validate pagination parameters
        if page < 1:
            page = 1
        if page_size < 1 or page_size > 200:  # Limit max page size
            page_size = 50
        
        # Strip leading slashes - GCS paths don't have leading slashes
        path = path.lstrip('/')
        
        logger.info(f"Listing directory recursively with path: '{path}', page: {page}, page_size: {page_size}, search: '{search_term}'")
        debug_log(f"Listing directory recursively with path: '{path}', page: {page}, page_size: {page_size}, search: '{search_term}'")
        
        # List directory recursively at specified path with optional search
        contents, error = gcs_service.list_gcs_directory_recursive(
            FILE_MANAGEMENT_BUCKET_NAME, 
            prefix=path,
            search_term=search_term if search_term else None  # Pass search term if provided
        )
        
        if error:
            return jsonify({
                "message": f"Error listing directory recursively at path: {path}",
                "error": error
            }), 500
        
        logger.info(f"Found {contents.get('total_folders', 0)} folders and {contents.get('total_files', 0)} files recursively")
        
        # Get root tree and extract root children
        root_tree = contents.get("tree", {})
        root_children = root_tree.get("children", []) if root_tree else []
        
        # Count root-level items before pagination
        total_root_items = len(root_children)
        total_root_folders = sum(1 for child in root_children if child.get("type") == "folder")
        total_root_files = sum(1 for child in root_children if child.get("type") == "file")
        
        # Apply pagination to root children
        start_index = (page - 1) * page_size
        end_index = start_index + page_size
        paginated_root_children = root_children[start_index:end_index]
        
        # Calculate pagination metadata
        total_pages = (total_root_items + page_size - 1) // page_size if total_root_items > 0 else 1
        has_next = page < total_pages
        has_prev = page > 1
        
        # Transform the paginated root children
        transformed_tree = []
        for child in paginated_root_children:
            transformed_item = transform_to_list_with_subfolders_format(child, depth=0)
            if transformed_item:
                transformed_tree.append(transformed_item)
        
        # Calculate total folders and files in entire tree (including nested)
        def count_items(node):
            """Recursively count folders and files in tree."""
            folders = 1 if node.get("type") == "folder" else 0
            files = 1 if node.get("type") == "file" else 0
            
            for child in node.get("children", []):
                child_folders, child_files = count_items(child)
                folders += child_folders
                files += child_files
            
            return folders, files
        
        total_folders = 0
        total_files = 0
        for child in root_children:  # Use all children for counting, not paginated
            child_folders, child_files = count_items(child)
            total_folders += child_folders
            total_files += child_files
        
        # Return the transformed array with pagination metadata
        response_data = {
            "items": transformed_tree,
            "pagination": {
                "current_page": page,
                "total_pages": total_pages,
                "page_size": page_size,
                "total_items": total_root_items,
                "total_folders": total_root_folders,
                "total_files": total_root_files,
                "has_next": has_next,
                "has_prev": has_prev
            },
            "totals": {
                "total_folders": total_folders,
                "total_files": total_files,
                "total_items": total_folders + total_files
            }
        }
        
        # Include search term in response if used
        if search_term:
            response_data["search"] = search_term
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"Error listing directory recursively at path {path}: {e}", exc_info=True)
        return jsonify({
            "message": "Error listing directory recursively",
            "error": str(e)
        }), 500


@gcs_bp.route('/search', methods=['GET'])
@jwt_required()
def search_files_and_folders():
    """
    Search for files and folders by name across all subfolders using Firestore.
    
    Query Parameters:
    - search (required): Search term (substring match, case-insensitive)
    - type (optional): 'file', 'folder', or 'both' (default: 'both')
    - path (optional): Filter results within specific path prefix (e.g., "Georgia 14/Pending Files/")
    - page (optional): Page number (default: 1, min: 1)
    - page_size (optional): Items per page (default: 50, min: 1, max: 200)
    
    Returns:
    - JSON response with matching files and folders, pagination info, and filters applied
    """
    try:
        # Get query parameters
        search_term = request.args.get('search', '').strip()
        if not search_term:
            return jsonify({"message": "Search term 'search' is required"}), 400
        
        search_type = request.args.get('type', 'both').lower()
        path_filter = request.args.get('path', None)
        page = int(request.args.get('page', 1))
        page_size = int(request.args.get('page_size', 50))
        
        # Validate parameters
        if search_type not in ['file', 'folder', 'both']:
            return jsonify({
                "message": "Invalid type. Must be 'file', 'folder', or 'both'"
            }), 400
        
        page = max(1, page)
        page_size = max(1, min(page_size, 200))
        
        # Normalize path filter if provided
        if path_filter:
            path_filter = path_filter.lstrip('/')
            if path_filter and not path_filter.endswith('/'):
                path_filter += '/'
        
        results = {
            "search_term": search_term,
            "type": search_type,
            "files": [],
            "folders": [],
            "files_pagination": {},
            "folders_pagination": {}
        }
        
        # Search files
        if search_type in ['file', 'both']:
            files, files_pagination, error = file_management_model.search_files_by_name(
                search_term=search_term,
                path_filter=path_filter,
                page=page,
                page_size=page_size
            )
            if error:
                logger.warning(f"Error searching files: {error}")
                results["files_error"] = error
            else:
                results["files"] = files
                results["files_pagination"] = files_pagination
        
        # Search folders
        if search_type in ['folder', 'both']:
            folders, folders_pagination, error = file_management_model.search_folders_by_name(
                search_term=search_term,
                path_filter=path_filter,
                page=page,
                page_size=page_size
            )
            if error:
                logger.warning(f"Error searching folders: {error}")
                results["folders_error"] = error
            else:
                results["folders"] = folders
                results["folders_pagination"] = folders_pagination
        
        # Add path filter to response if provided
        if path_filter:
            results["path_filter"] = path_filter
        
        # Calculate totals
        total_files = results.get("files_pagination", {}).get("total_items", 0)
        total_folders = results.get("folders_pagination", {}).get("total_items", 0)
        results["total_items"] = total_files + total_folders
        results["total_files"] = total_files
        results["total_folders"] = total_folders
        
        return jsonify(results), 200
        
    except ValueError as e:
        logger.error(f"Invalid parameter in search endpoint: {e}", exc_info=True)
        return jsonify({
            "message": "Invalid parameter",
            "error": str(e)
        }), 400
    except Exception as e:
        logger.error(f"Error in search endpoint: {e}", exc_info=True)
        return jsonify({
            "message": "Error performing search",
            "error": str(e)
        }), 500
        
@gcs_bp.route('/download-folders', methods=['POST'])
@jwt_required()
def download_folders_as_zip():
    """
    Download one or multiple folders as a zip file.
    
    Request Body:
    {
        "folderPaths": ["folder1/", "folder2/subfolder/"]  // Array of folder paths
    }
    
    Returns:
    - ZIP file stream with all files from the specified folders
    """
    try:
        user_id = get_jwt_identity()
        user_email = get_jwt().get('email', 'unknown_user')
        data = request.get_json()
        
        if not data or 'folderPaths' not in data:
            return jsonify({"message": "Invalid request. 'folderPaths' array is required"}), 400
        
        folder_paths = data['folderPaths']
        
        if not isinstance(folder_paths, list):
            return jsonify({"message": "'folderPaths' must be an array"}), 400
        
        if not folder_paths:
            return jsonify({"message": "'folderPaths' array cannot be empty"}), 400
        
        # Normalize folder paths (ensure they end with /)
        normalized_paths = []
        for folder_path in folder_paths:
            if not isinstance(folder_path, str) or not folder_path.strip():
                continue
            normalized_path = folder_path.strip()
            if not normalized_path.endswith('/'):
                normalized_path += '/'
            normalized_paths.append(normalized_path)
        
        if not normalized_paths:
            return jsonify({"message": "No valid folder paths provided"}), 400
        
        # Initialize GCS client
        storage_client = storage.Client()
        bucket = storage_client.bucket(FILE_MANAGEMENT_BUCKET_NAME)
        
        # Create in-memory zip file
        zip_buffer = io.BytesIO()
        
        total_files = 0
        errors = []
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Process each folder
            for folder_path in normalized_paths:
                try:
                    # List all blobs in the folder recursively
                    blobs = list(storage_client.list_blobs(bucket, prefix=folder_path))
                    
                    if not blobs:
                        # Check if folder marker exists
                        folder_blob = bucket.blob(folder_path)
                        if not folder_blob.exists():
                            errors.append(f"Folder not found: {folder_path}")
                            continue
                    
                    # Extract folder name for zip path structure
                    folder_name = folder_path.rstrip('/').split('/')[-1]
                    if not folder_name:
                        # If folder_name is empty, use parent folder name or a default
                        parts = [p for p in folder_path.rstrip('/').split('/') if p]
                        folder_name = parts[-1] if parts else 'root'
                    
                    # Process each file in the folder
                    for blob in blobs:
                        # Skip folder markers (empty blobs ending with /)
                        if blob.name.endswith('/') and blob.size == 0:
                            continue
                        
                        try:
                            # Download file content
                            file_content = blob.download_as_bytes()
                            
                            # Calculate relative path within zip
                            # Remove the folder prefix to get relative path
                            relative_path = blob.name[len(folder_path):]
                            
                            # Remove leading slashes from relative path
                            relative_path = relative_path.lstrip('/')
                            
                            if not relative_path:
                                # Skip if relative path is empty (shouldn't happen, but be safe)
                                continue
                            
                            # If multiple folders, include folder name in zip path
                            if len(normalized_paths) > 1:
                                zip_path = f"{folder_name}/{relative_path}"
                            else:
                                zip_path = relative_path
                            
                            # Add file to zip
                            zip_file.writestr(zip_path, file_content)
                            total_files += 1
                            
                        except Exception as e:
                            error_msg = f"Error processing file {blob.name}: {str(e)}"
                            logger.error(error_msg, exc_info=True)
                            errors.append(error_msg)
                
                except Exception as e:
                    error_msg = f"Error processing folder {folder_path}: {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    errors.append(error_msg)
        
        if total_files == 0:
            if errors:
                return jsonify({
                    "message": "No files found in the specified folders",
                    "errors": errors
                }), 404
            return jsonify({
                "message": "No files found in the specified folders"
            }), 404
        
        # Prepare zip file for download
        zip_buffer.seek(0)
        
        # Generate zip filename
        def sanitize_filename(name):
            """Sanitize filename to remove invalid characters."""
            invalid_chars = '<>:"/\\|?*'
            for char in invalid_chars:
                name = name.replace(char, '_')
            return name[:100]  # Limit length
        
        if len(normalized_paths) == 1:
            folder_name = normalized_paths[0].rstrip('/').split('/')[-1]
            if not folder_name:
                folder_name = 'folder'
            zip_filename = f"{sanitize_filename(folder_name)}.zip"
        else:
            timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
            zip_filename = f"folders_{timestamp}.zip"
        
        # Log download activity
        log_file_activity(
            user_email=user_email,
            activity_type=ActivityTypes.FILE_DOWNLOAD,
            filename=zip_filename,
            file_info={
                'folder_paths': normalized_paths,
                'total_files': total_files,
                'operation': 'download_folders_as_zip',
                'errors': errors if errors else None
            },
            request_obj=request
        )
        
        # Return zip file as download
        return Response(
            zip_buffer.getvalue(),
            mimetype='application/zip',
            headers={
                'Content-Disposition': f'attachment; filename="{zip_filename}"',
                'Content-Type': 'application/zip'
            }
        )
        
    except Exception as e:
        logger.error(f"Error downloading folders as zip: {e}", exc_info=True)
        return jsonify({
            "message": "Error downloading folders as zip",
            "error": str(e)
        }), 500


# ============================================================================
# Dynamic Path Configuration APIs
# ============================================================================

@gcs_bp.route('/dynamic-path-config', methods=['GET'])
@jwt_required()
def get_dynamic_path_config():
    """Get the current dynamic path configuration. All authenticated users can view."""
    try:
        config = get_path_config()
        
        if not config:
            # Return default from env if no config exists
            source_parts = GCS_SOURCE_ROOT.rstrip('/').split('/')
            base_path_display = f"{source_parts[0]}/" if len(source_parts) > 0 else ""
            
            return jsonify({
                "base_path": "",
                "source_folder": "",
                "last_updated_by": None,
                "last_updated_at": None,
                "source_path": GCS_SOURCE_ROOT,
                "base_path_display": base_path_display
            }), 200
        
        source_path = get_dynamic_source_path()
        base_path_display = get_dynamic_base_path()
        
        return jsonify({
            "base_path": config.get("base_path", ""),
            "source_folder": config.get("source_folder", ""),
            "last_updated_by": config.get("last_updated_by"),
            "last_updated_at": config.get("last_updated_at").isoformat() if config.get("last_updated_at") else None,
            "source_path": source_path,
            "base_path_display": base_path_display
        }), 200
    except Exception as e:
        logger.error(f"Error getting dynamic path config: {e}", exc_info=True)
        return jsonify({
            "message": "Error getting path configuration",
            "error": str(e)
        }), 500

@gcs_bp.route('/dynamic-path-config', methods=['PUT'])
@superadmin_required
def update_dynamic_path_config():
    """Update the dynamic path configuration. Only superadmin can update."""
    try:
        data = request.get_json()
        base_path = data.get('base_path', '').strip()
        source_folder = data.get('source_folder', '').strip()
        
        if not base_path and not source_folder:
            return jsonify({
                "message": "At least one of base_path or source_folder must be provided"
            }), 400
        
        user_id = get_jwt_identity()
        user_email = get_jwt().get('email', 'unknown_user')
        success, error = set_path_config(base_path, source_folder, user_id)
        
        if not success:
            return jsonify({
                "message": "Error updating path configuration",
                "error": error
            }), 500
        
        # Return updated config
        config = get_path_config()
        source_path = get_dynamic_source_path()
        base_path_display = get_dynamic_base_path()
        
        return jsonify({
            "message": "Path configuration updated successfully",
            "config": {
                "base_path": config.get("base_path", ""),
                "source_folder": config.get("source_folder", ""),
                "last_updated_by": config.get("last_updated_by"),
                "last_updated_at": config.get("last_updated_at").isoformat() if config.get("last_updated_at") else None,
                "source_path": source_path,
                "base_path_display": base_path_display
            }
        }), 200
    except Exception as e:
        logger.error(f"Error updating dynamic path config: {e}", exc_info=True)
        return jsonify({
            "message": "Error updating path configuration",
            "error": str(e)
        }), 500

@gcs_bp.route('/dynamic-source-path', methods=['GET'])
@jwt_required()
def get_dynamic_source_path_endpoint():
    """Get all files and folders in a specific dynamic source path and all parent paths up to root. Similar to /list-path but uses dynamic configuration."""
    try:
        path = request.args.get('path', '')
        
        # Get dynamic source path
        source_root_normalized = get_dynamic_source_path()
        if not source_root_normalized:
            return jsonify({
                "message": "Source path not configured"
            }), 400
        
        # Normalize source root
        source_root_normalized = source_root_normalized.rstrip('/') + '/'
        
        # If no path provided, use source root
        if not path:
            path = source_root_normalized
        else:
            # Ensure path starts with source root
            path = path.lstrip('/')
            if not path.startswith(source_root_normalized):
                path = source_root_normalized + path.lstrip('/')
        
        page = int(request.args.get('page', 1))
        page_size = int(request.args.get('page_size', 50))
        
        # Validate pagination parameters
        if page < 1:
            page = 1
        if page_size < 1 or page_size > 200:
            page_size = 50
        
        # Ensure path ends with / for directory listing
        if not path.endswith('/'):
            path += '/'
        
        # Check if we should fetch parents (default: true for backward compatibility)
        fetch_parents = request.args.get('fetch_parents', 'true').lower() == 'true'
        
        logger.info(f"Listing dynamic source directory with path: '{path}' (fetch_parents={fetch_parents})")
        
        root_path = source_root_normalized
        root_depth = get_path_depth(root_path)
        
        # Generate list of paths to query
        paths_to_query = []
        
        if not fetch_parents and path != root_path:
            # Optimized mode: only query the requested path
            # The frontend will be responsible for merging this into the existing tree
            paths_to_query = [path]
        elif path == root_path:
            # If current path is the root, only query root
            paths_to_query = [root_path]
        else:
            # Standard mode: Build all parent paths from root to current path
            # Split path into segments
            path_segments = path.rstrip('/').split('/')
            root_segments = root_path.rstrip('/').split('/')
            
            # Build paths incrementally from root to current
            current_build = root_path
            paths_to_query.append(current_build)
            
            # Add each intermediate path
            for i in range(len(root_segments), len(path_segments)):
                if path_segments[i]:  # Skip empty segments
                    current_build = current_build + path_segments[i] + '/'
                    paths_to_query.append(current_build)
        
        # Collect all folders and files from all paths
        all_folders = {}
        all_files = {}
        root_contents = None  # Store root path contents for pagination
        
        # Query the root path to get pagination info.
        # list_gcs_directory auto-clamps page on overflow — no re-query ever needed.
        if root_path in paths_to_query:
            root_contents, error = gcs_service.list_gcs_directory(
                FILE_MANAGEMENT_BUCKET_NAME,
                prefix=root_path,
                delimiter='/',
                page=page,
                page_size=page_size
            )

            if error:
                logger.warning(f"Error listing directory at root path: {root_path}, error: {error}")
            elif root_contents:
                # Sync page to the value actually used inside list_gcs_directory.
                # If the requested page was out of bounds it was clamped to total_pages
                # internally, so we simply adopt that clamped value here.
                pagination = root_contents.get("pagination", {})
                clamped_page = pagination.get("current_page", page)
                if clamped_page != page:
                    logger.info(f"Source path: page clamped from {page} to {clamped_page}")
                    page = clamped_page

        # Query each path and collect results.
        # Non-root paths are fetched in parallel; root result is already in root_contents.
        non_root_paths = [p for p in paths_to_query if p != root_path]

        def _fetch_source_path(qp):
            contents, err = gcs_service.list_gcs_directory(
                FILE_MANAGEMENT_BUCKET_NAME,
                prefix=qp,
                delimiter='/',
                page=1,
                page_size=10000
            )
            return qp, contents, err

        path_results = {}
        if root_contents is not None:
            path_results[root_path] = (root_contents, None)

        if non_root_paths:
            with ThreadPoolExecutor(max_workers=min(len(non_root_paths), 5)) as executor:
                for qp, contents, err in executor.map(_fetch_source_path, non_root_paths):
                    path_results[qp] = (contents, err)

        # Collect folders and files from all path results
        for query_path in paths_to_query:
            contents, err = path_results.get(query_path, (None, None))
            if err:
                logger.warning(f"Error listing directory at path: {query_path}, error: {err}")
                continue
            if not contents:
                continue

            # Collect folders (use path as key to avoid duplicates)
            for folder in contents.get("folders", []):
                folder_path = folder.get("path", folder.get("name", ""))
                # Only include folders that are direct children of the query_path
                if folder_path.startswith(query_path):
                    # Check if it's a direct child (not deeper)
                    relative_path = folder_path[len(query_path):]
                    if '/' not in relative_path.rstrip('/'):
                        all_folders[folder_path] = folder

            # Collect files (use path as key to avoid duplicates)
            for file in contents.get("files", []):
                file_path = file.get("path", file.get("name", ""))
                # Only include files that are direct children of the query_path
                if file_path.startswith(query_path):
                    # Check if it's a direct child (not in a subfolder)
                    relative_path = file_path[len(query_path):]
                    if '/' not in relative_path:
                        all_files[file_path] = file

        logger.info(f"Found {len(all_folders)} folders and {len(all_files)} files across all paths")

        # Build tree structure from folders and files
        # Create a map of folder paths to their tree nodes
        folder_nodes = {}

        # First, create all folder nodes
        for folder_path, folder in all_folders.items():
            folder_name = folder_path.rstrip('/').split('/')[-1] if folder_path else ""
            folder_depth = get_path_depth(folder_path)
            folder_type = "Folder" if (folder_depth - root_depth == 1) else "Sub Folder"

            folder_nodes[folder_path] = {
                "id": folder_path,
                "name": folder_name,
                "type": folder_type,
                "created_date": folder.get("created_date"),
                "path": folder_path,
                "children": []
            }

        # Organize folders into tree structure
        # Sort folders by depth (shallow to deep) to build tree correctly
        sorted_folders = sorted(all_folders.keys(), key=lambda p: (get_path_depth(p), p))

        # Build parent-child relationships
        root_items = []
        for folder_path in sorted_folders:
            folder_node = folder_nodes[folder_path]

            # Find parent folder path
            # Get the parent by removing the last segment
            path_parts = folder_path.rstrip('/').split('/')
            if len(path_parts) > len(root_path.rstrip('/').split('/')):
                # This folder has a parent (it's deeper than root)
                parent_path = '/'.join(path_parts[:-1]) + '/'

                if parent_path in folder_nodes:
                    # Add to parent's children
                    folder_nodes[parent_path]["children"].append(folder_node)
                else:
                    # Parent folder not in our collection, treat as root-level
                    root_items.append(folder_node)
            else:
                # This is a root-level folder (direct child of root_path)
                root_items.append(folder_node)

        # Add files to their parent folders
        for file_path, file in all_files.items():
            file_name = file_path.split('/')[-1] if file_path else ""

            file_item = {
                "id": file_path,
                "name": file_name,
                "type": "File",
                "created_date": file.get("created_date"),
                "path": file_path,
                "metadata": {
                    "updatedAt": file.get("updatedAt"),
                    "contentType": file.get("contentType")
                }
            }

            # Find the parent folder for this file
            # Get the directory path (parent folder) - where the file is located
            file_dir = '/'.join(file_path.split('/')[:-1]) + '/' if '/' in file_path else root_path

            # Find the closest parent folder that exists in our folder_nodes
            parent_folder = None
            current_dir = file_dir

            # Check if the file's directory is one of the folders we collected
            if current_dir in folder_nodes:
                parent_folder = folder_nodes[current_dir]
            else:
                # Try to find a parent folder by going up the directory tree
                # But only check folders that are in our collected folders
                path_parts = current_dir.rstrip('/').split('/')
                for i in range(len(path_parts), 0, -1):
                    check_dir = '/'.join(path_parts[:i]) + '/'
                    if check_dir in folder_nodes:
                        parent_folder = folder_nodes[check_dir]
                        break

            if parent_folder:
                # Add file to parent folder's children
                parent_folder["children"].append(file_item)
            else:
                # File is at root level (not in any collected folder), add to root items
                root_items.append(file_item)

        # Sort root items: folders first, then files, both alphabetically
        root_items.sort(key=lambda x: (x.get("type") != "Folder" and x.get("type") != "Sub Folder", x.get("name", "").lower()))

        # Sort children within each folder: folders first, then files
        def sort_children(node):
            if "children" in node:
                node["children"].sort(key=lambda x: (x.get("type") != "Folder" and x.get("type") != "Sub Folder", x.get("name", "").lower()))
                # Recursively sort nested children
                for child in node["children"]:
                    if "children" in child:
                        sort_children(child)

        for item in root_items:
            sort_children(item)

        return jsonify({
            "path": path,
            "items": root_items,
            "pagination": root_contents.get("pagination", {}) if root_contents else {}
        }), 200

    except Exception as e:
        logger.error(f"Error listing dynamic source directory at path {path}: {e}", exc_info=True)
        return jsonify({
            "message": "Error listing source directory",
            "error": str(e)
        }), 500

@gcs_bp.route('/dynamic-destination-paths', methods=['GET'])
@jwt_required()
def list_dynamic_destination_paths():
    """Get all files and folders in a specific dynamic destination path and all parent paths up to root. Root is base path and excludes source folder. Similar to /list-destination-path but uses dynamic configuration."""
    path = ''
    try:
        # Get dynamic paths
        base_path = get_dynamic_base_path()
        source_path = get_dynamic_source_path()
        
        if not base_path:
            return jsonify({
                "message": "Base path not configured"
            }), 400
        
        # Normalize paths
        destination_root = base_path.rstrip('/') + '/' if base_path else ""
        source_root_normalized = source_path.rstrip('/') + '/' if source_path else ""
        
        path = request.args.get('path', '')
        
        # If no path provided, use destination root
        if not path:
            path = destination_root
        else:
            # Ensure path starts with destination root
            path = path.lstrip('/')
            if not path.startswith(destination_root):
                path = destination_root + path.lstrip('/')
        
        page = int(request.args.get('page', 1))
        page_size = int(request.args.get('page_size', 50))
        
        # Validate pagination parameters
        if page < 1:
            page = 1
        if page_size < 1 or page_size > 200:
            page_size = 50
        
        # Ensure path ends with / for directory listing
        if not path.endswith('/'):
            path += '/'
        
        # Check if we should fetch parents (default: true for backward compatibility)
        fetch_parents = request.args.get('fetch_parents', 'true').lower() == 'true'
        
        logger.info(f"Listing dynamic destination directory with path: '{path}' (fetch_parents={fetch_parents})")
        
        root_path = destination_root
        root_depth = get_path_depth(root_path)
        
        # Generate list of paths to query
        paths_to_query = []
        
        if not fetch_parents and path != root_path:
            # Optimized mode: only query the requested path
            paths_to_query = [path]
        elif path == root_path:
            # If current path is the root, only query root
            paths_to_query = [root_path]
        else:
            # Standard mode: Build all parent paths from root to current path
            # Split path into segments
            path_segments = path.rstrip('/').split('/')
            root_segments = root_path.rstrip('/').split('/')
            
            # Build paths incrementally from root to current
            current_build = root_path
            paths_to_query.append(current_build)
            
            # Add each intermediate path
            for i in range(len(root_segments), len(path_segments)):
                if path_segments[i]:  # Skip empty segments
                    current_build = current_build + path_segments[i] + '/'
                    paths_to_query.append(current_build)
        
        # Collect all folders and files from all paths
        all_folders = {}
        all_files = {}
        root_contents = None  # Store root path contents for pagination
        
        # Query the root path to get pagination info.
        # list_gcs_directory auto-clamps page on overflow — no re-query ever needed.
        if root_path in paths_to_query:
            root_contents, error = gcs_service.list_gcs_directory(
                FILE_MANAGEMENT_BUCKET_NAME,
                prefix=root_path,
                delimiter='/',
                page=page,
                page_size=page_size
            )

            if error:
                logger.warning(f"Error listing directory at root path: {root_path}, error: {error}")
            elif root_contents:
                # Sync page to the value actually used inside list_gcs_directory.
                # If the requested page was out of bounds it was clamped to total_pages
                # internally, so we simply adopt that clamped value here.
                pagination = root_contents.get("pagination", {})
                clamped_page = pagination.get("current_page", page)
                if clamped_page != page:
                    logger.info(f"Destination path: page clamped from {page} to {clamped_page}")
                    page = clamped_page

        # Query each path and collect results.
        # Non-root paths are fetched in parallel; root result is already in root_contents.
        non_root_paths = [p for p in paths_to_query if p != root_path]

        def _fetch_destination_path(qp):
            contents, err = gcs_service.list_gcs_directory(
                FILE_MANAGEMENT_BUCKET_NAME,
                prefix=qp,
                delimiter='/',
                page=1,
                page_size=10000
            )
            return qp, contents, err

        path_results = {}
        if root_contents is not None:
            path_results[root_path] = (root_contents, None)

        if non_root_paths:
            with ThreadPoolExecutor(max_workers=min(len(non_root_paths), 5)) as executor:
                for qp, contents, err in executor.map(_fetch_destination_path, non_root_paths):
                    path_results[qp] = (contents, err)

        # Collect folders and files from all path results
        for query_path in paths_to_query:
            contents, err = path_results.get(query_path, (None, None))
            if err:
                logger.warning(f"Error listing directory at path: {query_path}, error: {err}")
                continue
            if not contents:
                continue

            # Collect folders (use path as key to avoid duplicates)
            for folder in contents.get("folders", []):
                folder_path = folder.get("path", folder.get("name", ""))
                # Exclude source folder
                if folder_path != source_root_normalized and not folder_path.startswith(source_root_normalized):
                    # Only include folders that are direct children of the query_path
                    if folder_path.startswith(query_path):
                        # Check if it's a direct child (not deeper)
                        relative_path = folder_path[len(query_path):]
                        if '/' not in relative_path.rstrip('/'):
                            all_folders[folder_path] = folder

            # Collect files (use path as key to avoid duplicates)
            for file in contents.get("files", []):
                file_path = file.get("path", file.get("name", ""))
                # Only include files that are direct children of the query_path
                if file_path.startswith(query_path):
                    # Check if it's a direct child (not in a subfolder)
                    relative_path = file_path[len(query_path):]
                    if '/' not in relative_path:
                        all_files[file_path] = file
        
        logger.info(f"Found {len(all_folders)} folders and {len(all_files)} files across all paths")
        
        # Build tree structure from folders and files
        # Create a map of folder paths to their tree nodes
        folder_nodes = {}
        
        # First, create all folder nodes
        for folder_path, folder in all_folders.items():
            folder_name = folder_path.rstrip('/').split('/')[-1] if folder_path else ""
            folder_depth = get_path_depth(folder_path)
            folder_type = "Folder" if (folder_depth - root_depth == 1) else "Sub Folder"
            
            folder_nodes[folder_path] = {
                "id": folder_path,
                "name": folder_name,
                "type": folder_type,
                "created_date": folder.get("created_date"),
                "path": folder_path,
                "children": []
            }
        
        # Organize folders into tree structure
        # Sort folders by depth (shallow to deep) to build tree correctly
        sorted_folders = sorted(all_folders.keys(), key=lambda p: (get_path_depth(p), p))
        
        # Build parent-child relationships
        root_items = []
        for folder_path in sorted_folders:
            folder_node = folder_nodes[folder_path]
            
            # Find parent folder path
            # Get the parent by removing the last segment
            path_parts = folder_path.rstrip('/').split('/')
            if len(path_parts) > len(root_path.rstrip('/').split('/')):
                # This folder has a parent (it's deeper than root)
                parent_path = '/'.join(path_parts[:-1]) + '/'
                
                if parent_path in folder_nodes:
                    # Add to parent's children
                    folder_nodes[parent_path]["children"].append(folder_node)
                else:
                    # Parent folder not in our collection, treat as root-level
                    root_items.append(folder_node)
            else:
                # This is a root-level folder (direct child of root_path)
                root_items.append(folder_node)
        
        # Add files to their parent folders
        for file_path, file in all_files.items():
            file_name = file_path.split('/')[-1] if file_path else ""
            
            file_item = {
                "id": file_path,
                "name": file_name,
                "type": "File",
                "created_date": file.get("created_date"),
                "path": file_path,
                "metadata": {
                    "updatedAt": file.get("updatedAt"),
                    "contentType": file.get("contentType")
                }
            }
            
            # Find the parent folder for this file
            # Get the directory path (parent folder) - where the file is located
            file_dir = '/'.join(file_path.split('/')[:-1]) + '/' if '/' in file_path else root_path
            
            # Find the closest parent folder that exists in our folder_nodes
            parent_folder = None
            current_dir = file_dir
            
            # Check if the file's directory is one of the folders we collected
            if current_dir in folder_nodes:
                parent_folder = folder_nodes[current_dir]
            else:
                # Try to find a parent folder by going up the directory tree
                # But only check folders that are in our collected folders
                path_parts = current_dir.rstrip('/').split('/')
                for i in range(len(path_parts), 0, -1):
                    check_dir = '/'.join(path_parts[:i]) + '/'
                    if check_dir in folder_nodes:
                        parent_folder = folder_nodes[check_dir]
                        break
            
            if parent_folder:
                # Add file to parent folder's children
                parent_folder["children"].append(file_item)
            else:
                # File is at root level (not in any collected folder), add to root items
                root_items.append(file_item)
        
        # Sort root items: folders first, then files, both alphabetically
        root_items.sort(key=lambda x: (x.get("type") != "Folder" and x.get("type") != "Sub Folder", x.get("name", "").lower()))
        
        # Sort children within each folder: folders first, then files
        def sort_children(node):
            if "children" in node:
                node["children"].sort(key=lambda x: (x.get("type") != "Folder" and x.get("type") != "Sub Folder", x.get("name", "").lower()))
                # Recursively sort nested children
                for child in node["children"]:
                    if "children" in child:
                        sort_children(child)
        
        for item in root_items:
            sort_children(item)
        
        return jsonify({
            "path": path,
            "items": root_items,
            "pagination": root_contents.get("pagination", {}) if root_contents else {}
        }), 200
        
    except Exception as e:
        logger.error(f"Error listing dynamic destination directory at path {path}: {e}", exc_info=True)
        return jsonify({
            "message": "Error listing destination directory",
            "error": str(e)
        }), 500

@gcs_bp.route('/list-dynamic-path', methods=['GET'])
@jwt_required()
def list_dynamic_path_directory():
    """List files and folders in a dynamic path. Similar to /list-path but uses dynamic configuration."""
    try:
        path = request.args.get('path', '')
        path_type = request.args.get('type', 'source')  # 'source' or 'destination'
        
        page = int(request.args.get('page', 1))
        page_size = int(request.args.get('page_size', 50))
        
        # Validate pagination parameters
        if page < 1:
            page = 1
        if page_size < 1 or page_size > 200:
            page_size = 50
        
        # Get dynamic paths
        source_path = get_dynamic_source_path()
        base_path = get_dynamic_base_path()
        
        # Determine which root path to use
        if path_type == 'source':
            root_path = source_path
        else:  # destination
            root_path = base_path
        
        # If path is provided, combine with root
        if path:
            path = path.lstrip('/')
            if not path.endswith('/'):
                path += '/'
            # Combine with appropriate root
            if root_path:
                full_path = f"{root_path.rstrip('/')}/{path}"
            else:
                full_path = path
        else:
            full_path = root_path
        
        # Normalize path
        full_path = full_path.lstrip('/')
        if not full_path.endswith('/'):
            full_path += '/'
        
        logger.info(f"Listing dynamic directory with path: '{full_path}' (type: {path_type})")
        
        # Use existing list_gcs_directory function
        contents, error = gcs_service.list_gcs_directory(
            FILE_MANAGEMENT_BUCKET_NAME,
            full_path,
            delimiter='/',
            page=page,
            page_size=page_size
        )
        
        if error:
            return jsonify({
                "message": f"Error listing {path_type} directory",
                "error": error
            }), 500
        
        return jsonify({
            "path": full_path,
            "path_type": path_type,
            "root_path": root_path,
            "folders": contents.get("folders", []),
            "files": contents.get("files", []),
            "pagination": contents.get("pagination", {})
        }), 200
        
    except Exception as e:
        logger.error(f"Error listing dynamic directory: {e}", exc_info=True)
        return jsonify({
            "message": "Error listing directory",
            "error": str(e)
        }), 500

@gcs_bp.route('/dynamic-path-configs/all', methods=['GET'])
@superadmin_required
def get_all_path_configs_endpoint():
    """Get all path configurations from the database. Superadmin only."""
    try:
        configs, error = get_all_path_configs()
        
        if error:
            return jsonify({
                "message": "Error retrieving path configurations",
                "error": error
            }), 500
        
        # Format timestamps for JSON serialization
        formatted_configs = []
        for config in configs:
            formatted_config = {
                "id": config.get("id"),
                "base_path": config.get("base_path", ""),
                "source_folder": config.get("source_folder", ""),
                "last_updated_by": config.get("last_updated_by", ""),
                "last_updated_at": config.get("last_updated_at").isoformat() if config.get("last_updated_at") else None
            }
            formatted_configs.append(formatted_config)
        
        return jsonify({
            "configs": formatted_configs,
            "count": len(formatted_configs)
        }), 200
    except Exception as e:
        logger.error(f"Error getting all path configs: {e}", exc_info=True)
        return jsonify({
            "message": "Error retrieving path configurations",
            "error": str(e)
        }), 500

@gcs_bp.route('/dynamic-path-configs/user', methods=['GET'])
@superadmin_required
def get_path_configs_for_user_endpoint():
    """Get path configurations for the authenticated user. Superadmin only."""
    try:
        user_id = get_jwt_identity()
        user_email = get_jwt().get('email', 'unknown_user')
        
        if not user_id:
            return jsonify({
                "message": "User email not found in token"
            }), 400
        
        configs, error = get_path_configs_for_user(user_id)
        
        if error:
            return jsonify({
                "message": "Error retrieving path configurations for user",
                "error": error
            }), 500
        
        # Format timestamps for JSON serialization
        formatted_configs = []
        for config in configs:
            formatted_config = {
                "id": config.get("id"),
                "base_path": config.get("base_path", ""),
                "source_folder": config.get("source_folder", ""),
                "last_updated_by": config.get("last_updated_by", ""),
                "last_updated_at": config.get("last_updated_at").isoformat() if config.get("last_updated_at") else None
            }
            formatted_configs.append(formatted_config)
        
        return jsonify({
            "user_id": user_id,
            "user_email": user_email,
            "configs": formatted_configs,
            "count": len(formatted_configs)
        }), 200
    except Exception as e:
        logger.error(f"Error getting path configs for user: {e}", exc_info=True)
        return jsonify({
            "message": "Error retrieving path configurations for user",
            "error": str(e)
        }), 500