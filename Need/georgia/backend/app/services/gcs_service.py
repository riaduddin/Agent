# backend/app/services/gcs_service.py
from google.cloud import storage
from google.cloud.exceptions import NotFound
from google.api_core import exceptions as api_core_exceptions # For specific exception handling
from tenacity import retry, stop_after_attempt, wait_exponential # Import tenacity
import datetime
import uuid
from google.cloud import storage, firestore
import logging
from google.api_core import exceptions
from flask_jwt_extended import get_jwt_identity
import os # Keep os for path joining if needed, though GCS uses forward slashes
from typing import Optional # Import Optional for type hinting
import logging # Import logging
from concurrent.futures import ThreadPoolExecutor
from app import config # Import config
from app.utils.debug_logger import debug_log, debug_error

# Initialize GCS client using project ID from config
storage_client = storage.Client(project=config.PROJECT_ID)

# Use config values
BUCKET_NAME = config.BUCKET_NAME
# BUCKET_DIRECTORY can be defined in config if needed, or keep default here
BUCKET_DIRECTORY = os.getenv('BUCKET_DIRECTORY', '') # Keep allowing override via env if desired, or remove

# Validation for BUCKET_NAME is now in config.py

logger = logging.getLogger(__name__) # Initialize logger at module level

# Renamed function to be more specific about uploading the *original* file

def upload_original_to_gcs(file_stream, filename, content_type, user_email="unknown_user"):
    """Uploads a file stream to Google Cloud Storage."""
    try:
        bucket = storage_client.get_bucket(BUCKET_NAME)
    except NotFound:
        debug_error(f"Bucket '{BUCKET_NAME}' not found.")
        # Optionally, try to create the bucket (requires storage.buckets.create permission)
        # try:
        #     bucket = storage_client.create_bucket(BUCKET_NAME)
        #     print(f"Bucket '{BUCKET_NAME}' created.")
        # except Exception as create_e:
        #     debug_error(f" Failed to create bucket '{BUCKET_NAME}': {create_e}")
        #     return None, f"Bucket '{BUCKET_NAME}' not found and could not be created."
        return None, f"Bucket '{BUCKET_NAME}' not found."
    except Exception as e:
        debug_error(f"Could not get bucket '{BUCKET_NAME}': {e}")
        return None, f"Could not access bucket '{BUCKET_NAME}'."

    # Create a unique filename to avoid collisions, include user and timestamp
    timestamp = datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:8]
    # Sanitize filename and user_email
    safe_filename = "".join(c if c.isalnum() or c in ['.', '_', '-'] else '_' for c in filename)
    safe_user_email = "".join(c if c.isalnum() or c in ['@', '.', '_', '-'] else '_' for c in user_email)

    # Construct the blob path including optional directory
    blob_name = os.path.join(BUCKET_DIRECTORY, f"{safe_user_email}/{timestamp}_{unique_id}_{safe_filename}")
    # Ensure forward slashes for GCS paths if needed, though os.path.join might handle it
    blob_name = blob_name.replace("\\", "/")

    blob = bucket.blob(blob_name)

    try:
        # Reset stream position just in case
        file_stream.seek(0)
        # Upload the stream
        blob.upload_from_file(file_stream, content_type=content_type)
        debug_log(f"File {filename} uploaded to {blob_name} in bucket {BUCKET_NAME}.")

        # Get file size after upload (or before if stream supports it reliably)
        # Getting size from the stream before upload is safer
        file_stream.seek(0, os.SEEK_END)
        file_size = file_stream.tell()
        file_stream.seek(0) # Reset stream position

        debug_log(f"File {filename} uploaded to {blob_name} in bucket {BUCKET_NAME}. Size: {file_size} bytes.")

        # Return the GCS URI, blob name, and file size
        gcs_uri = f"gs://{BUCKET_NAME}/{blob_name}"
        return gcs_uri, blob_name, file_size, None # gcs_uri, blob_name, file_size, error

    except Exception as e:
        debug_error(f"Failed to upload {filename} to GCS: {e}")
        return None, None, None, f"Failed to upload file to storage: {e}" # gcs_uri, blob_name, file_size, error

# Shorten default expiration to 30 seconds (0.5 minutes)
def generate_download_signed_url(blob_name: str, expiration_minutes: float = 0.5) -> Optional[str]:
    """Generates a v4 signed URL for downloading a blob."""
    if not blob_name:
        # logger is not defined here, use print or import logging
        debug_error("Cannot generate signed URL: blob_name is empty.")
        return None

    try:
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(blob_name)

        # Generate a v4 signed URL for reading the object
        # The service account needs roles/iam.serviceAccountTokenCreator permission
        # and permissions to read the object (e.g., roles/storage.objectViewer)
        url = blob.generate_signed_url(
            version="v4",
            # This URL is valid for expiration_minutes from now.
            expiration=datetime.timedelta(minutes=expiration_minutes), # Use float for sub-minute values
            method="GET",
            response_disposition="inline", # Change to 'inline' to suggest opening in browser
            # response_type="application/pdf" # Optional: Set content type if needed
        )
        logger.info(f"Generated signed URL for blob: {blob_name}")
        return url
    except NotFound:
        logger.error(f"Cannot generate signed URL: Blob '{blob_name}' not found in bucket '{BUCKET_NAME}'.")
        return None
    except Exception as e:
        logger.error(f"Failed to generate signed URL for {blob_name}: {e}", exc_info=True)
        return None
    
from app.config import FIRESTORE_DATABASE_ID

logger = logging.getLogger(__name__)
db = firestore.Client(database=FIRESTORE_DATABASE_ID)

def list_gcs_directory(bucket_name, prefix, delimiter='/', page=1, page_size=50):
    """Lists files and folders in a GCS directory with pagination."""
    debug_log(f"--- list_gcs_directory ---")
    debug_log(f"Bucket: {bucket_name}, Prefix: {prefix}, Page: {page}, Page Size: {page_size}")
    try:
        from app.models import file_management_model
        
        bucket = storage_client.bucket(bucket_name)
        blobs = storage_client.list_blobs(bucket, prefix=prefix, delimiter=delimiter)

        # First pass: Collect folder paths and file info (minimal data for sorting/pagination)
        folder_paths_raw = []  # Just paths for sorting
        file_info_raw = []  # Store (file_path, blob) tuples for sorting
        
        for page_result in blobs.pages:
            # Collect folder paths
            for p in page_result.prefixes:
                folder_paths_raw.append(p)
            
            # Collect file info
            for blob in page_result:
                if not blob.name.endswith('/'):
                    file_info_raw.append((blob.name, blob))
        
        # Sort by name for consistent pagination (before fetching metadata)
        folder_paths_raw.sort()
        file_info_raw.sort(key=lambda x: x[0])  # Sort by file path
        
        # Calculate pagination on raw data
        total_folders = len(folder_paths_raw)
        total_files = len(file_info_raw)
        total_items = total_folders + total_files

        # Compute total_pages early so page can be clamped before slicing.
        # This means callers never need a second GCS scan for out-of-bounds pages.
        total_pages = (total_items + page_size - 1) // page_size if total_items > 0 else 1
        if page > total_pages and total_pages > 0:
            debug_log(f"Page {page} exceeds total_pages {total_pages}, clamping to {total_pages}")
            page = total_pages

        # Calculate start and end indices using (possibly clamped) page
        start_index = (page - 1) * page_size
        end_index = start_index + page_size
        
        # Paginate folders and files separately (folders first)
        # Determine how many folders and files to include in this page
        if start_index < total_folders:
            # Page includes some folders
            folder_end = min(end_index, total_folders)
            paginated_folder_paths = folder_paths_raw[start_index:folder_end]
            # If page extends beyond folders, include some files too
            if end_index > total_folders:
                file_count_needed = end_index - total_folders
                paginated_file_info = file_info_raw[:file_count_needed]
            else:
                paginated_file_info = []
        else:
            # Page only includes files
            paginated_folder_paths = []
            file_start = start_index - total_folders
            file_end = end_index - total_folders
            paginated_file_info = file_info_raw[file_start:file_end]
        
        # Now fetch metadata ONLY for paginated items (major performance improvement!)
        paginated_folder_paths_normalized = [p if p.endswith('/') else p + '/' for p in paginated_folder_paths]
        paginated_file_paths = [fp for fp, _ in paginated_file_info]
        
        # Fetch folder and file metadata concurrently instead of sequentially
        with ThreadPoolExecutor(max_workers=2) as executor:
            folder_future = (
                executor.submit(file_management_model.get_folders_metadata_batch, paginated_folder_paths_normalized)
                if paginated_folder_paths_normalized else None
            )
            file_future = (
                executor.submit(file_management_model.get_files_metadata_batch, paginated_file_paths)
                if paginated_file_paths else None
            )
            folder_metadata_map = folder_future.result() if folder_future else {}
            file_metadata_map = file_future.result() if file_future else {}
        
        # Build paginated folder items with metadata
        paginated_folders = []
        for p in paginated_folder_paths:
            folder_path = p if p.endswith('/') else p + '/'
            folder_metadata = folder_metadata_map.get(folder_path)
            
            folder_item = {
                "name": p,
                "path": p,
                "updatedAt": None
            }
            
            # Add created_date and updated_date from metadata if available
            if folder_metadata:
                if 'created_at' in folder_metadata:
                    folder_item['created_date'] = folder_metadata['created_at']
                elif 'created_date' in folder_metadata:
                    folder_item['created_date'] = folder_metadata['created_date']
                if 'updated_at' in folder_metadata:
                    folder_item['updated_date'] = folder_metadata['updated_at']
                elif 'updated_date' in folder_metadata:
                    folder_item['updated_date'] = folder_metadata['updated_date']
            else:
                # If no metadata, set to None
                folder_item['created_date'] = None
                folder_item['updated_date'] = None
            
            paginated_folders.append(folder_item)
        
        # Build paginated file items with metadata
        paginated_files = []
        for file_path, blob in paginated_file_info:
            file_metadata = file_metadata_map.get(file_path)
            
            file_item = {
                "name": blob.name,
                "path": blob.name,
                "size": blob.size,
                "updatedAt": blob.updated.isoformat() if blob.updated else None,
                "contentType": blob.content_type
            }
            
            # Add created_date, updated_date, and size from metadata if available
            if file_metadata:
                if 'created_at' in file_metadata:
                    file_item['created_date'] = file_metadata['created_at']
                elif 'created_date' in file_metadata:
                    file_item['created_date'] = file_metadata['created_date']
                if 'updated_at' in file_metadata:
                    file_item['updated_date'] = file_metadata['updated_at']
                elif 'updated_date' in file_metadata:
                    file_item['updated_date'] = file_metadata['updated_date']
                # Use file_size_bytes from metadata if available, otherwise use blob.size
                if 'file_size_bytes' in file_metadata:
                    file_item['size'] = file_metadata['file_size_bytes']
            else:
                # If no metadata, use blob data
                file_item['created_date'] = blob.time_created.isoformat() if blob.time_created else None
                file_item['updated_date'] = blob.updated.isoformat() if blob.updated else None
            
            paginated_files.append(file_item)
        
        # Pagination metadata (total_pages computed and page already clamped above)
        has_next = page < total_pages
        has_prev = page > 1

        debug_log(f"Found folders: {len(paginated_folders)} of {total_folders}")
        debug_log(f"Found files: {len(paginated_files)} of {total_files}")
        debug_log(f"Total folders before pagination: {total_folders}")
        debug_log(f"Total files before pagination: {total_files}")
        debug_log(f"Page {page} of {total_pages}")
        
        return {
            "folders": paginated_folders,
            "files": paginated_files,
            "pagination": {
                "current_page": page,
                "total_pages": total_pages,
                "page_size": page_size,
                "total_items": total_items,
                "total_folders": total_folders,
                "total_files": total_files,
                "has_next": has_next,
                "has_prev": has_prev
            }
        }, None
    except exceptions.GoogleAPICallError as e:
        logger.error(f"GCS API error listing directory {prefix}: {e}")
        debug_error(f"Error in list_gcs_directory: {e}")
        return None, str(e)

def list_gcs_directory_recursive(bucket_name, prefix='', search_term=None):
    """Recursively lists all files and folders in a GCS directory including subfolders as a tree structure.
    
    Args:
        bucket_name: GCS bucket name
        prefix: Path prefix to search within
        search_term: Optional search term to filter files and folders by name (case-insensitive substring match)
    """
    try:
        from app.models import file_management_model
        
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        
        # Normalize prefix - strip leading/trailing slashes
        prefix = prefix.strip('/')
        search_prefix = prefix if prefix else ''
        
        # First, list all blobs (including folder markers) to get files and folder markers
        debug_log(f" Listing all blobs with prefix '{search_prefix}'")
        blobs = storage_client.list_blobs(bucket, prefix=search_prefix)
        blob_list = list(blobs)
        
        debug_log(f" Found {len(blob_list)} total blobs with prefix '{search_prefix}'")
        
        # Collect all folders from blob list (both from folder markers and file paths)
        # Also extract all parent folder paths from each blob path
        all_folders = set()
        
        def get_relative_folder_path(absolute_path):
            """Helper to get relative folder path from absolute path."""
            if not absolute_path:
                return None
            
            # Get relative folder path (remove prefix if present)
            if prefix:
                if absolute_path.startswith(prefix):
                    relative_folder = absolute_path[len(prefix):].lstrip('/')
                    return relative_folder.rstrip('/') if relative_folder else None
                else:
                    return None
            else:
                return absolute_path.rstrip('/')
        
        # Collect folders from folder markers and file paths, and extract all parent folders
        for blob in blob_list:
            blob_name = blob.name
            
            if blob.name.endswith('/') and blob.size == 0:
                # This is a folder marker
                folder_path = blob_name.rstrip('/')
                relative_folder = get_relative_folder_path(folder_path)
                if relative_folder:
                    all_folders.add(relative_folder)
            else:
                # Extract folder path from file blob name and all parent folders
                if not blob_name.endswith('/'):
                    # Get the folder containing this file
                    folder_path = '/'.join(blob_name.split('/')[:-1])
                    if folder_path:
                        # Extract all parent folder paths
                        path_parts = folder_path.split('/')
                        for i in range(1, len(path_parts) + 1):
                            parent_path = '/'.join(path_parts[:i])
                            if parent_path:
                                # Get relative path and add to folders set
                                relative_folder = get_relative_folder_path(parent_path)
                                if relative_folder:
                                    all_folders.add(relative_folder)
        
        # Use a single delimiter call to catch any remaining empty folders at the top level
        # This is much faster than recursive calls
        try:
            # Check if we need delimiter call - if we have many blobs, we likely have all folders already
            if len(blob_list) < 1000:  # For smaller datasets, check for empty folders
                debug_log(f"Checking for additional empty folders with delimiter at '{search_prefix}'")
                folder_blobs = storage_client.list_blobs(bucket, prefix=search_prefix, delimiter='/')
                for page_result in folder_blobs.pages:  # Renamed from 'page' to avoid shadowing
                    for folder_prefix in page_result.prefixes:
                        # Get relative folder path
                        relative_folder = get_relative_folder_path(folder_prefix)
                        if relative_folder:
                            all_folders.add(relative_folder)
            else:
                debug_log(f"Skipping delimiter check - comprehensive folder coverage from blob list ({len(blob_list)} blobs)")
        except Exception as e:
            logger.warning(f"Error checking for empty folders with delimiter: {e}")
        
        debug_log(f"Found {len(all_folders)} total folders (including empty ones)")
        logger.info(f"Found {len(blob_list)} blobs with prefix '{search_prefix}', {len(all_folders)} folders")
        
        # Build a simple nested dictionary structure: path_parts -> {files: [], folders: {}}
        tree_dict = {}  # Will be a nested dict structure
        
        # First, add all folders to the tree structure (even empty ones)
        for folder_path in all_folders:
            folder_parts = [p for p in folder_path.split('/') if p]
            
            # Navigate/create the tree structure
            current = tree_dict
            for part in folder_parts:
                if part:  # Skip empty parts
                    if part not in current:
                        current[part] = {"_type": "folder", "_children": {}}
                    current = current[part]["_children"]
        
        # First pass: Collect all file paths for batch metadata fetching
        # Optimize by processing in a single loop
        file_paths_to_fetch = []
        file_blob_map = {}  # Map absolute_path -> blob info for later processing
        
        # Pre-compute prefix_parts to avoid repeated splitting
        prefix_parts = prefix.split('/') if prefix else []
        
        for blob in blob_list:
            # Skip folder markers
            if blob.name.endswith('/') and blob.size == 0:
                continue
            
            # Only process files
            if not blob.name.endswith('/'):
                blob_name = blob.name
                
                # Check if this file is under the prefix
                if prefix and not blob_name.startswith(prefix):
                    continue
                
                # Get the path parts
                path_parts = blob_name.split('/')
                file_name = path_parts[-1]
                folder_parts = path_parts[:-1] if len(path_parts) > 1 else []
                
                # Get relative folder parts (remove prefix if present)
                if prefix:
                    if folder_parts[:len(prefix_parts)] == prefix_parts:
                        relative_folder_parts = folder_parts[len(prefix_parts):]
                    else:
                        relative_folder_parts = folder_parts
                else:
                    relative_folder_parts = folder_parts
                
                # Build absolute path efficiently
                if prefix:
                    absolute_path = f"{prefix}/{'/'.join(relative_folder_parts)}/{file_name}" if relative_folder_parts else f"{prefix}/{file_name}"
                else:
                    absolute_path = '/'.join(relative_folder_parts + [file_name]) if relative_folder_parts else file_name
                
                file_paths_to_fetch.append(absolute_path)
                file_blob_map[absolute_path] = {
                    'blob': blob,
                    'file_name': file_name,
                    'relative_folder_parts': relative_folder_parts
                }
        
        # Batch fetch all file metadata at once
        debug_log(f"Batch fetching metadata for {len(file_paths_to_fetch)} files")
        file_metadata_map = file_management_model.get_files_metadata_batch(file_paths_to_fetch)
        debug_log(f"Retrieved metadata for {len(file_metadata_map)} files from Firestore")
        
        # Second pass: Process files and build tree structure with metadata
        for absolute_path, file_info in file_blob_map.items():
            blob = file_info['blob']
            file_name = file_info['file_name']
            relative_folder_parts = file_info['relative_folder_parts']
            
            # Navigate/create the tree structure
            current = tree_dict
            for part in relative_folder_parts:
                if part:  # Skip empty parts
                    if part not in current:
                        current[part] = {"_type": "folder", "_children": {}}
                    current = current[part]["_children"]
            
            # Add file to current location
            if "_files" not in current:
                current["_files"] = []
            
            # Get metadata from batch fetch
            file_metadata = file_metadata_map.get(absolute_path)
            
            # Build file item efficiently
            file_item = {
                "name": file_name,
                "path": absolute_path,
                "type": "file",
                "size": blob.size,
                "updatedAt": blob.updated.isoformat() if blob.updated else None,
                "contentType": blob.content_type
            }
            
            # Add created_date, updated_date, and size from metadata if available
            if file_metadata:
                if 'created_at' in file_metadata:
                    file_item['created_date'] = file_metadata['created_at']
                if 'updated_at' in file_metadata:
                    file_item['updated_date'] = file_metadata['updated_at']
                # Use file_size_bytes from metadata if available
                if 'file_size_bytes' in file_metadata:
                    file_item['size'] = file_metadata['file_size_bytes']
            else:
                # If no metadata, use blob data
                file_item['created_date'] = blob.time_created.isoformat() if blob.time_created else None
                file_item['updated_date'] = blob.updated.isoformat() if blob.updated else None
            
            current["_files"].append(file_item)
        
        debug_log(f"Tree dict structure created with {len(tree_dict)} top-level items")
        
        # Collect all folder paths for batch metadata fetching
        folder_paths_to_fetch = []
        folder_path_parts_map = {}  # Map folder_path -> path_parts for tree building
        
        def get_absolute_path(relative_parts):
            """Get absolute path from relative parts."""
            if prefix:
                if relative_parts:
                    return f"{prefix}/{'/'.join(relative_parts)}"
                else:
                    return prefix
            else:
                return '/'.join(relative_parts) if relative_parts else ""
        
        def collect_folder_paths(node_dict, current_path_parts=[]):
            """Collect all folder paths from tree structure for batch metadata fetching."""
            for key, value in node_dict.items():
                if key == "_files":
                    continue
                if isinstance(value, dict) and "_type" in value:
                    # This is a folder
                    folder_path_parts = current_path_parts + [key]
                    folder_absolute_path = get_absolute_path(folder_path_parts)
                    folder_path_for_metadata = folder_absolute_path + '/' if not folder_absolute_path.endswith('/') else folder_absolute_path
                    
                    folder_paths_to_fetch.append(folder_path_for_metadata)
                    folder_path_parts_map[folder_path_for_metadata] = {
                        'path_parts': folder_path_parts,
                        'key': key,
                        'value': value
                    }
                    
                    # Recursively collect subfolder paths
                    collect_folder_paths(value.get("_children", {}), folder_path_parts)
        
        # Collect all folder paths
        collect_folder_paths(tree_dict, [])
        
        # Batch fetch all folder metadata at once
        debug_log(f"Batch fetching metadata for {len(folder_paths_to_fetch)} folders")
        folder_metadata_map = file_management_model.get_folders_metadata_batch(folder_paths_to_fetch)
        debug_log(f"Retrieved metadata for {len(folder_metadata_map)} folders from Firestore")
        
        # Convert nested dictionary to tree node format
        def build_tree_from_dict(node_dict, current_path_parts=[]):
            """Convert dictionary structure to tree node format."""
            children = []
            
            # Process folders first
            for key, value in sorted(node_dict.items()):
                if key == "_files":
                    continue
                if isinstance(value, dict) and "_type" in value:
                    # This is a folder
                    folder_path_parts = current_path_parts + [key]
                    
                    # Build absolute path
                    folder_absolute_path = get_absolute_path(folder_path_parts)
                    
                    # Ensure folder path ends with / for metadata lookup
                    folder_path_for_metadata = folder_absolute_path + '/' if not folder_absolute_path.endswith('/') else folder_absolute_path
                    
                    # Get folder metadata from batch fetch
                    folder_metadata = folder_metadata_map.get(folder_path_for_metadata)
                    
                    # Recursively build children
                    folder_children = build_tree_from_dict(value.get("_children", {}), folder_path_parts)
                    
                    # Add files in this folder
                    folder_files = value.get("_files", [])
                    folder_children.extend(folder_files)
                    
                    # Sort: folders first, then files
                    folder_children.sort(key=lambda x: (x.get("type") != "folder", x.get("name", "").lower()))
                    
                    folder_item = {
                        "name": key,
                        "path": folder_absolute_path + '/',
                        "type": "folder",
                        "children": folder_children
                    }
                    
                    # Add created_date and updated_date from metadata if available
                    if folder_metadata:
                        if 'created_at' in folder_metadata:
                            folder_item['created_date'] = folder_metadata['created_at']
                        if 'updated_at' in folder_metadata:
                            folder_item['updated_date'] = folder_metadata['updated_at']
                    else:
                        # If no metadata, set to None
                        folder_item['created_date'] = None
                        folder_item['updated_date'] = None
                    
                    children.append(folder_item)
            
            # Add files at current level
            if "_files" in node_dict:
                children.extend(node_dict["_files"])
            
            # Sort: folders first, then files
            children.sort(key=lambda x: (x.get("type") != "folder", x.get("name", "").lower()))
            
            return children
        
        # Build the tree
        root_children = build_tree_from_dict(tree_dict, [])
        
        debug_log(f"Root children count: {len(root_children)}")
        
        # Create root node
        if prefix:
            root_name = prefix.split('/')[-1] if prefix else "root"
            root_path = prefix
        else:
            root_name = "root"
            root_path = ""
        
        root_node = {
            "name": root_name,
            "path": root_path + '/' if root_path else "",
            "type": "folder",
            "children": root_children
        }
        
        # Apply search filter if provided
        if search_term:
            search_lower = search_term.lower()
            
            def filter_tree(node):
                """Recursively filter tree by search term. Keeps nodes that match or have matching children."""
                if node.get("type") == "file":
                    # For files, check if name matches
                    file_name = node.get("name", "").lower()
                    return search_lower in file_name
                elif node.get("type") == "folder":
                    # For folders, check if name matches
                    folder_name = node.get("name", "").lower()
                    folder_matches = search_lower in folder_name
                    
                    # Filter children recursively
                    filtered_children = []
                    for child in node.get("children", []):
                        if filter_tree(child):
                            filtered_children.append(child)
                    
                    # Keep folder if it matches or has matching children
                    if folder_matches or filtered_children:
                        node["children"] = filtered_children
                        return True
                    return False
                return False
            
            # Filter the root node's children
            if root_node.get("children"):
                filtered_children = [child for child in root_node["children"] if filter_tree(child)]
                root_node["children"] = filtered_children
        
        # Calculate totals
        def count_items(node):
            """Recursively count folders and files in tree."""
            folders = 1 if node.get("type") == "folder" else 0
            files = 1 if node.get("type") == "file" else 0
            
            for child in node.get("children", []):
                child_folders, child_files = count_items(child)
                folders += child_folders
                files += child_files
            
            return folders, files
        
        total_folders, total_files = count_items(root_node)
        
        debug_log(f"Total folders: {total_folders}, Total files: {total_files}")
        
        return {
            "tree": root_node,
            "total_folders": total_folders,
            "total_files": total_files,
            "total_items": total_folders + total_files
        }, None
        
    except exceptions.GoogleAPICallError as e:
        logger.error(f"GCS API error listing directory recursively {prefix}: {e}")
        debug_error(f"Error in list_gcs_directory_recursive: {e}")
        return None, str(e)
    except Exception as e:
        logger.error(f"Unexpected error in list_gcs_directory_recursive {prefix}: {e}")
        debug_error(f"Unexpected error in list_gcs_directory_recursive: {e}")
        return None, str(e)
    
def get_destination_roots(bucket_name, source_root):
    """Lists folders starting from 'Georgia 14/', excluding the source path."""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        
        # Extract the root directory from source_root
        # If source_root is "Georgia 14/Source Files/", we want to start from "Georgia 14/"
        source_parts = source_root.rstrip('/').split('/')
        if len(source_parts) >= 1:
            destination_root = f"{source_parts[0]}/"  # "Georgia 14/"
        else:
            destination_root = ""  # Fallback to bucket root if source_root is empty
        
        debug_log(f"Getting destination roots starting from: {destination_root}")
        debug_log(f"Excluding source root: {source_root}")
        
        # List folders under the destination root
        blobs = storage_client.list_blobs(bucket, prefix=destination_root, delimiter='/')
        
        # Get all folders under "Georgia 14/" and exclude the source path
        destinations = []
        for page in blobs.pages:
            for prefix in page.prefixes:
                # Exclude the exact source root path
                if prefix != source_root:
                    # Remove the destination_root prefix to show relative paths
                    relative_name = prefix[len(destination_root):] if prefix.startswith(destination_root) else prefix
                    destinations.append({
                        "name": prefix,  # Full path for navigation
                        "display_name": relative_name  # Relative name for display
                    })
        
        debug_log(f"Found {len(destinations)} destination folders: {[d['display_name'] for d in destinations]}")
        return {"destinations": destinations}, None
    except exceptions.GoogleAPICallError as e:
        logger.error(f"GCS API error getting destination roots: {e}")
        return None, str(e)

def list_destinations_recursive(bucket_name, source_root, path='', page=1, page_size=50, search_term=None):
    """Recursively lists all destination folders and files in a tree structure, excluding source root.
    
    Args:
        bucket_name: GCS bucket name
        source_root: Source root path to exclude
        path: Path to list (relative to destination root)
        page: Page number for pagination
        page_size: Items per page
        search_term: Optional search term to filter files and folders by name (case-insensitive substring match)
    """
    try:
        from app.models import file_management_model
        
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        
        # Normalize source_root to ensure it ends with / for proper comparison
        source_root_normalized = source_root.rstrip('/') + '/'
        
        # Extract the root directory from source_root
        source_parts = source_root.rstrip('/').split('/')
        if len(source_parts) >= 1:
            destination_root = f"{source_parts[0]}/"  # "Georgia 14/"
        else:
            destination_root = ""  # Fallback to bucket root if source_root is empty
        
        # Normalize the path parameter
        path = path.strip('/')
        
        # Determine the search prefix
        if path:
            # If path is provided, use it (should be relative to destination_root or absolute)
            if path.startswith(destination_root):
                search_prefix = path
            else:
                search_prefix = f"{destination_root}{path}" if destination_root else path
        else:
            # If no path, start from destination root
            search_prefix = destination_root
        
        # Ensure search_prefix ends with / if not empty
        if search_prefix and not search_prefix.endswith('/'):
            search_prefix += '/'
        
        debug_log(f" Listing destinations recursively")
        debug_log(f" Source root (normalized): {source_root_normalized}")
        debug_log(f" Destination root: {destination_root}")
        debug_log(f" Search prefix: {search_prefix}")
        
        # First, list all blobs (including folder markers) to get files and folder markers
        debug_log(f" Listing all blobs with prefix '{search_prefix}'")
        blobs = storage_client.list_blobs(bucket, prefix=search_prefix)
        blob_list = list(blobs)
        
        debug_log(f" Found {len(blob_list)} total blobs with prefix '{search_prefix}'")
        
        # Collect all folders from blob list (both from folder markers and file paths)
        # Also extract all parent folder paths from each blob path
        all_folders = set()
        all_absolute_folder_paths = set()  # Track absolute paths for parent extraction
        
        def get_relative_folder_path(absolute_path):
            """Helper to get relative folder path from absolute path."""
            if not absolute_path:
                return None
            
            # Normalize for comparison
            path_normalized = absolute_path if absolute_path.endswith('/') else absolute_path + '/'
            
            # Exclude source root and anything under it
            if path_normalized == source_root_normalized or path_normalized.startswith(source_root_normalized):
                return None
            
            # Get relative folder path from destination root
            if destination_root:
                if absolute_path.startswith(destination_root):
                    relative_folder = absolute_path[len(destination_root):].lstrip('/')
                    return relative_folder.rstrip('/') if relative_folder else None
                else:
                    return None
            else:
                return absolute_path.rstrip('/')
        
        # Collect folders from folder markers and file paths, and extract all parent folders
        for blob in blob_list:
            blob_name = blob.name
            
            if blob.name.endswith('/') and blob.size == 0:
                # This is a folder marker
                folder_path = blob_name.rstrip('/')
                relative_folder = get_relative_folder_path(folder_path)
                if relative_folder:
                    all_folders.add(relative_folder)
                    all_absolute_folder_paths.add(folder_path)
            else:
                # Extract folder path from file blob name and all parent folders
                if not blob_name.endswith('/'):
                    # Get the folder containing this file
                    folder_path = '/'.join(blob_name.split('/')[:-1])
                    if folder_path:
                        # Extract all parent folder paths
                        path_parts = folder_path.split('/')
                        for i in range(1, len(path_parts) + 1):
                            parent_path = '/'.join(path_parts[:i])
                            if parent_path:
                                all_absolute_folder_paths.add(parent_path)
                                
                                # Get relative path and add to folders set
                                relative_folder = get_relative_folder_path(parent_path)
                                if relative_folder:
                                    all_folders.add(relative_folder)
        
        # Use a single delimiter call to catch any remaining empty folders at the top level
        # This is much faster than recursive calls
        # Only do this if we haven't found many folders yet (optimization: skip if we already have comprehensive coverage)
        try:
            # Check if we need delimiter call - if we have many blobs, we likely have all folders already
            # Only use delimiter for edge cases with truly empty folders
            if len(blob_list) < 1000:  # For smaller datasets, check for empty folders
                debug_log(f" Checking for additional empty folders with delimiter at '{search_prefix}'")
                folder_blobs = storage_client.list_blobs(bucket, prefix=search_prefix, delimiter='/')
                for page_result in folder_blobs.pages:  # Renamed from 'page' to avoid shadowing the 'page' parameter
                    for folder_prefix in page_result.prefixes:
                        # Normalize folder_prefix for comparison
                        folder_prefix_normalized = folder_prefix if folder_prefix.endswith('/') else folder_prefix + '/'
                        
                        # Exclude the source root path and anything under it
                        if folder_prefix_normalized == source_root_normalized or folder_prefix_normalized.startswith(source_root_normalized):
                            continue
                        
                        # Get relative folder path from destination root
                        relative_folder = get_relative_folder_path(folder_prefix)
                        if relative_folder:
                            all_folders.add(relative_folder)
            else:
                debug_log(f" Skipping delimiter check - comprehensive folder coverage from blob list ({len(blob_list)} blobs)")
        except Exception as e:
            logger.warning(f"Error checking for empty folders with delimiter: {e}")
        
        debug_log(f" Found {len(all_folders)} total folders (including empty ones)")
        
        debug_log(f" Found {len(all_folders)} total folders (including from markers)")
        
        # Build a simple nested dictionary structure
        tree_dict = {}
        
        # First, add all folders to the tree structure (even empty ones)
        for folder_path in all_folders:
            folder_parts = [p for p in folder_path.split('/') if p]
            
            # Navigate/create the tree structure
            current = tree_dict
            for part in folder_parts:
                if part:  # Skip empty parts
                    if part not in current:
                        current[part] = {"_type": "folder", "_children": {}}
                    current = current[part]["_children"]
        
        # First pass: Collect all file paths for batch metadata fetching
        # Optimize by processing in a single loop
        file_paths_to_fetch = []
        file_blob_map = {}  # Map absolute_path -> blob info for later processing
        
        # Pre-compute destination_parts to avoid repeated splitting
        destination_parts = [p for p in destination_root.rstrip('/').split('/') if p] if destination_root else []
        
        for blob in blob_list:
            # Skip folder markers
            if blob.name.endswith('/') and blob.size == 0:
                continue
            
            # Exclude files in source root and anything under it
            if blob.name.startswith(source_root_normalized):
                continue
            
            # Only process files
            if not blob.name.endswith('/'):
                blob_name = blob.name
                
                # Get the path parts
                path_parts = blob_name.split('/')
                file_name = path_parts[-1]
                folder_parts = path_parts[:-1] if len(path_parts) > 1 else []
                
                # Get relative folder parts from destination root
                if destination_root:
                    if blob_name.startswith(destination_root):
                        if folder_parts[:len(destination_parts)] == destination_parts:
                            relative_folder_parts = folder_parts[len(destination_parts):]
                        else:
                            relative_folder_parts = folder_parts
                    else:
                        continue
                else:
                    relative_folder_parts = folder_parts
                
                # Build absolute path efficiently
                if destination_root:
                    if relative_folder_parts:
                        absolute_path = f"{destination_root}{'/'.join(relative_folder_parts)}/{file_name}"
                    else:
                        absolute_path = f"{destination_root}{file_name}"
                else:
                    absolute_path = '/'.join(relative_folder_parts + [file_name]) if relative_folder_parts else file_name
                
                file_paths_to_fetch.append(absolute_path)
                file_blob_map[absolute_path] = {
                    'blob': blob,
                    'file_name': file_name,
                    'relative_folder_parts': relative_folder_parts
                }
        
        # Batch fetch all file metadata at once
        debug_log(f"Batch fetching metadata for {len(file_paths_to_fetch)} files")
        file_metadata_map = file_management_model.get_files_metadata_batch(file_paths_to_fetch)
        debug_log(f"Retrieved metadata for {len(file_metadata_map)} files from Firestore")
        
        # Second pass: Process files and build tree structure with metadata
        # Optimize by pre-computing datetime conversions
        for absolute_path, file_info in file_blob_map.items():
            blob = file_info['blob']
            file_name = file_info['file_name']
            relative_folder_parts = file_info['relative_folder_parts']
            
            # Navigate/create the tree structure
            current = tree_dict
            for part in relative_folder_parts:
                if part:  # Skip empty parts
                    if part not in current:
                        current[part] = {"_type": "folder", "_children": {}}
                    current = current[part]["_children"]
            
            # Add file to current location
            if "_files" not in current:
                current["_files"] = []
            
            # Get metadata from batch fetch
            file_metadata = file_metadata_map.get(absolute_path)
            
            # Build file item efficiently
            file_item = {
                "name": file_name,
                "path": absolute_path,
                "type": "file",
                "size": blob.size,
                "updatedAt": blob.updated.isoformat() if blob.updated else None,
                "contentType": blob.content_type
            }
            
            # Add created_date, updated_date, and size from metadata if available
            if file_metadata:
                if 'created_at' in file_metadata:
                    file_item['created_date'] = file_metadata['created_at']
                if 'updated_at' in file_metadata:
                    file_item['updated_date'] = file_metadata['updated_at']
                # Use file_size_bytes from metadata if available
                if 'file_size_bytes' in file_metadata:
                    file_item['size'] = file_metadata['file_size_bytes']
            else:
                # If no metadata, use blob data
                file_item['created_date'] = blob.time_created.isoformat() if blob.time_created else None
                file_item['updated_date'] = blob.updated.isoformat() if blob.updated else None
            
            current["_files"].append(file_item)
        
        debug_log(f" Tree dict structure created with {len(tree_dict)} top-level items")
        
        # Convert nested dictionary to tree node format
        def get_absolute_path(relative_parts):
            """Get absolute path from relative parts."""
            if destination_root:
                if relative_parts:
                    return f"{destination_root}{'/'.join(relative_parts)}"
                else:
                    return destination_root.rstrip('/')
            else:
                return '/'.join(relative_parts) if relative_parts else ""
        
        # Collect all folder paths for batch metadata fetching
        folder_paths_to_fetch = []
        folder_path_parts_map = {}  # Map folder_path -> path_parts for tree building
        
        def collect_folder_paths(node_dict, current_path_parts=[]):
            """Collect all folder paths from tree structure for batch metadata fetching."""
            for key, value in node_dict.items():
                if key == "_files":
                    continue
                if isinstance(value, dict) and "_type" in value:
                    # This is a folder
                    folder_path_parts = current_path_parts + [key]
                    folder_absolute_path = get_absolute_path(folder_path_parts)
                    folder_path_for_metadata = folder_absolute_path + '/' if not folder_absolute_path.endswith('/') else folder_absolute_path
                    
                    folder_paths_to_fetch.append(folder_path_for_metadata)
                    folder_path_parts_map[folder_path_for_metadata] = {
                        'path_parts': folder_path_parts,
                        'key': key,
                        'value': value
                    }
                    
                    # Recursively collect subfolder paths
                    collect_folder_paths(value.get("_children", {}), folder_path_parts)
        
        # Collect all folder paths
        collect_folder_paths(tree_dict, [])
        
        # Batch fetch all folder metadata at once
        debug_log(f" Batch fetching metadata for {len(folder_paths_to_fetch)} folders")
        folder_metadata_map = file_management_model.get_folders_metadata_batch(folder_paths_to_fetch)
        debug_log(f" Retrieved metadata for {len(folder_metadata_map)} folders from Firestore")
        
        def build_tree_from_dict(node_dict, current_path_parts=[]):
            """Convert dictionary structure to tree node format."""
            children = []
            
            # Process folders first
            for key, value in sorted(node_dict.items()):
                if key == "_files":
                    continue
                if isinstance(value, dict) and "_type" in value:
                    # This is a folder
                    folder_path_parts = current_path_parts + [key]
                    folder_absolute_path = get_absolute_path(folder_path_parts)
                    
                    # Ensure folder path ends with / for metadata lookup
                    folder_path_for_metadata = folder_absolute_path + '/' if not folder_absolute_path.endswith('/') else folder_absolute_path
                    
                    # Get folder metadata from batch fetch
                    folder_metadata = folder_metadata_map.get(folder_path_for_metadata)
                    
                    # Recursively build children
                    folder_children = build_tree_from_dict(value.get("_children", {}), folder_path_parts)
                    
                    # Add files in this folder
                    folder_files = value.get("_files", [])
                    folder_children.extend(folder_files)
                    
                    # Sort: folders first, then files
                    folder_children.sort(key=lambda x: (x.get("type") != "folder", x.get("name", "").lower()))
                    
                    folder_item = {
                        "name": key,
                        "path": folder_absolute_path + '/',
                        "type": "folder",
                        "children": folder_children
                    }
                    
                    # Add created_date and updated_date from metadata if available
                    if folder_metadata:
                        if 'created_at' in folder_metadata:
                            folder_item['created_date'] = folder_metadata['created_at']
                        if 'updated_at' in folder_metadata:
                            folder_item['updated_date'] = folder_metadata['updated_at']
                    else:
                        # If no metadata, set to None
                        folder_item['created_date'] = None
                        folder_item['updated_date'] = None
                    
                    children.append(folder_item)
            
            # Add files at current level
            if "_files" in node_dict:
                children.extend(node_dict["_files"])
            
            # Sort: folders first, then files
            children.sort(key=lambda x: (x.get("type") != "folder", x.get("name", "").lower()))
            
            return children
        
        # Build the tree
        root_children = build_tree_from_dict(tree_dict, [])
        
        debug_log(f" Root children count: {len(root_children)}")
        
        # Calculate totals before pagination
        def count_items(node):
            """Recursively count folders and files in tree."""
            folders = 1 if node.get("type") == "folder" else 0
            files = 1 if node.get("type") == "file" else 0
            
            for child in node.get("children", []):
                child_folders, child_files = count_items(child)
                folders += child_folders
                files += child_files
            
            return folders, files
        
        # Count folders and files in root children
        total_root_folders = sum(1 for child in root_children if child.get("type") == "folder")
        total_root_files = sum(1 for child in root_children if child.get("type") == "file")
        total_root_items = len(root_children)
        
        # Apply pagination to root children
        start_index = (page - 1) * page_size
        end_index = start_index + page_size
        paginated_root_children = root_children[start_index:end_index]
        
        # Calculate pagination metadata
        total_pages = (total_root_items + page_size - 1) // page_size if total_root_items > 0 else 1
        has_next = page < total_pages
        has_prev = page > 1
        
        # Apply search filter if provided (before pagination)
        if search_term:
            search_lower = search_term.lower()
            
            def filter_tree(node):
                """Recursively filter tree by search term. Keeps nodes that match or have matching children."""
                if node.get("type") == "file":
                    # For files, check if name matches
                    file_name = node.get("name", "").lower()
                    return search_lower in file_name
                elif node.get("type") == "folder":
                    # For folders, check if name matches
                    folder_name = node.get("name", "").lower()
                    folder_matches = search_lower in folder_name
                    
                    # Filter children recursively
                    filtered_children = []
                    for child in node.get("children", []):
                        if filter_tree(child):
                            filtered_children.append(child)
                    
                    # Keep folder if it matches or has matching children
                    if folder_matches or filtered_children:
                        node["children"] = filtered_children
                        return True
                    return False
                return False
            
            # Filter the root children before pagination
            if root_children:
                filtered_root_children = [child for child in root_children if filter_tree(child)]
                root_children = filtered_root_children
                # Recalculate counts after filtering
                total_root_folders = sum(1 for child in root_children if child.get("type") == "folder")
                total_root_files = sum(1 for child in root_children if child.get("type") == "file")
                total_root_items = len(root_children)
                # Reapply pagination after filtering
                start_index = (page - 1) * page_size
                end_index = start_index + page_size
                paginated_root_children = root_children[start_index:end_index]
                # Recalculate pagination metadata
                total_pages = (total_root_items + page_size - 1) // page_size if total_root_items > 0 else 1
                has_next = page < total_pages
                has_prev = page > 1
        
        # Create root node with paginated children
        if path:
            root_name = path.split('/')[-1] if path else "destinations"
            root_path = path
        elif destination_root:
            root_name = destination_root.rstrip('/').split('/')[-1] if destination_root.rstrip('/') else "destinations"
            root_path = destination_root.rstrip('/')
        else:
            root_name = "destinations"
            root_path = ""
        
        root_node = {
            "name": root_name,
            "path": root_path + '/' if root_path else "",
            "type": "folder",
            "children": paginated_root_children
        }
        
        # Calculate total folders and files in the entire tree (including nested)
        total_folders = 0
        total_files = 0
        for child in root_children:  # Use all children for counting, not paginated
            child_folders, child_files = count_items(child)
            total_folders += child_folders
            total_files += child_files
        
        # Add root node itself (it's a folder)
        total_folders += 1
        
        debug_log(f" Total folders: {total_folders}, Total files: {total_files}")
        debug_log(f" Pagination - Page {page} of {total_pages}, Showing {len(paginated_root_children)} of {total_root_items} root items")
        
        return {
            "tree": root_node,
            "total_folders": total_folders,
            "total_files": total_files,
            "total_items": total_folders + total_files,
            "pagination": {
                "current_page": page,
                "total_pages": total_pages,
                "page_size": page_size,
                "total_items": total_root_items,
                "total_folders": total_root_folders,
                "total_files": total_root_files,
                "has_next": has_next,
                "has_prev": has_prev
            }
        }, None
        
    except exceptions.GoogleAPICallError as e:
        logger.error(f"GCS API error listing destinations recursively: {e}")
        debug_error(f"Error in list_destinations_recursive: {e}")
        return None, str(e)
    except Exception as e:
        logger.error(f"Unexpected error in list_destinations_recursive: {e}")
        debug_error(f"Unexpected error in list_destinations_recursive: {e}")
        return None, str(e)


def transfer_gcs_object(bucket_name, source_blob_name, destination_blob_name, operation='copy'):
    """Transfers a GCS object."""
    debug_log(f"--- transfer_gcs_object ---")
    debug_log(f"Bucket: {bucket_name}, Source: {source_blob_name}, Destination: {destination_blob_name}, Operation: {operation}")
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        source_blob = bucket.blob(source_blob_name)
        
        if not source_blob.exists():
            debug_log("Source file not found.")
            return None, "Source file not found."

        destination_blob = bucket.blob(destination_blob_name)
        
        # Rewrite for large files, copy for smaller ones
        destination_blob.rewrite(source_blob)
        debug_log("File transferred.")

        if operation == 'move':
            source_blob.delete()
            debug_log("Source file deleted.")
            
        transfer_id = log_transfer_operation({
            "fileName": source_blob_name.split('/')[-1],
            "source": source_blob_name,
            "destination": destination_blob_name,
            "operation": operation,
            "status": "succeeded",
            "transfer_type": None  # Explicitly set to None for file transfers (not folder)
        })
        return {"transferId": transfer_id, "status": "succeeded"}, None

    except exceptions.GoogleAPICallError as e:
        logger.error(f"GCS API error transferring object: {e}")
        debug_error(f"Error in transfer_gcs_object: {e}")
        return None, str(e)


def transfer_gcs_folder(bucket_name, source_folder_path, destination_folder_path, operation='move', progress_callback=None, user_id=None):
    """Recursively transfers an entire folder and its contents with progress tracking."""
    debug_log(f"--- transfer_gcs_folder ---")
    debug_log(f"Bucket: {bucket_name}, Source Folder: {source_folder_path}, Destination Folder: {destination_folder_path}, Operation: {operation}")
    
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        
        # Ensure paths end with /
        if not source_folder_path.endswith('/'):
            source_folder_path += '/'
        if not destination_folder_path.endswith('/'):
            destination_folder_path += '/'
        
        # List all blobs including empty folder markers
        all_blobs = list(storage_client.list_blobs(bucket, prefix=source_folder_path))
        
        # Separate files and folder markers (empty folders are represented as blobs ending with '/' and size 0)
        files_to_transfer = [blob for blob in all_blobs if not (blob.name.endswith('/') and blob.size == 0)]
        folder_markers = [blob for blob in all_blobs if blob.name.endswith('/') and blob.size == 0]
        
        total_files = len(files_to_transfer)
        total_folders = len(folder_markers)
        total_items = total_files + total_folders
        
        debug_log(f"Total files to transfer: {total_files}")
        debug_log(f"Total folder markers to transfer: {total_folders}")
        
        # Call progress callback with initial state
        if progress_callback:
            progress_callback(0, total_items, "Starting transfer...")
        
        transferred_files = []
        transferred_folders = []
        errors = []
        
        # First, transfer all folder markers (including empty folders)
        for i, folder_blob in enumerate(folder_markers):
            try:
                # Calculate the relative path within the folder
                relative_path = folder_blob.name[len(source_folder_path):]
                destination_folder_marker = destination_folder_path + relative_path
                
                folder_name = folder_blob.name.rstrip('/').split('/')[-1]
                debug_log(f"Transferring folder marker ({i+1}/{total_folders}): {folder_blob.name} -> {destination_folder_marker}")
                
                # Call progress callback
                if progress_callback:
                    progress_callback(i, total_items, f"Transferring folder {folder_name}")
                
                # Create destination folder marker
                dest_folder_marker = bucket.blob(destination_folder_marker)
                dest_folder_marker.upload_from_string('', content_type='application/x-directory')
                
                if operation == 'move':
                    # Delete source folder marker after successful copy
                    folder_blob.delete()
                
                # Update or create folder metadata based on operation
                try:
                    from app.models import file_management_model
                    
                    # Extract folder name from destination path
                    folder_name = destination_folder_marker.rstrip('/').split('/')[-1]
                    
                    # Calculate parent path
                    parent_parts = destination_folder_marker.rstrip('/').split('/')[:-1]
                    parent_path = '/'.join(parent_parts) + '/' if parent_parts else ''
                    
                    if operation == 'move':
                        # Update folder path in metadata
                        metadata_success, metadata_error = file_management_model.update_folder_path(
                            old_path=folder_blob.name,
                            new_path=destination_folder_marker
                        )
                        if not metadata_success:
                            # Try to create new folder metadata if update failed
                            file_management_model.create_folder_metadata(
                                folder_path=destination_folder_marker,
                                folder_name=folder_name,
                                parent_path=parent_path,
                                user_email=user_id or 'system'
                            )
                    elif operation == 'copy':
                        # Create new folder metadata for copied folder
                        file_management_model.create_folder_metadata(
                            folder_path=destination_folder_marker,
                            folder_name=folder_name,
                            parent_path=parent_path,
                            user_email=user_id or 'system'
                        )
                except Exception as meta_error:
                    debug_log(f"WARNING: Could not update/create folder metadata for {folder_blob.name}: {meta_error}")
                
                transferred_folders.append({
                    "source": folder_blob.name,
                    "destination": destination_folder_marker,
                    "operation": operation
                })
                
                # Call progress callback after each folder
                if progress_callback:
                    progress_callback(i + 1, total_items, f"Completed folder {folder_name}")
                
            except Exception as folder_error:
                error_msg = f"Error transferring folder {folder_blob.name}: {str(folder_error)}"
                debug_log(error_msg)
                errors.append(error_msg)
                continue
        
        # Then, transfer all files
        for i, blob in enumerate(files_to_transfer):
            try:
                # Calculate the relative path within the folder
                relative_path = blob.name[len(source_folder_path):]
                destination_blob_name = destination_folder_path + relative_path
                
                current_file = blob.name.split('/')[-1]
                file_index = total_folders + i
                debug_log(f"Transferring file ({file_index+1}/{total_items}): {blob.name} -> {destination_blob_name}")
                
                # Call progress callback with current file
                if progress_callback:
                    progress_callback(file_index, total_items, f"Transferring {current_file}")
                
                # Use rewrite for the transfer
                source_blob = bucket.blob(blob.name)
                destination_blob = bucket.blob(destination_blob_name)
                destination_blob.rewrite(source_blob)
                
                if operation == 'move':
                    # Delete source after successful copy
                    source_blob.delete()
                
                # Update or create file metadata based on operation
                if not blob.name.endswith('/'):
                    # Determine destination folder path
                    dest_folder_path = '/'.join(destination_blob_name.split('/')[:-1])
                    if dest_folder_path:
                        dest_folder_path += '/'
                    
                    # Import here to avoid circular imports
                    try:
                        from app.models import file_management_model
                        
                        if operation == 'move':
                            # Update existing file metadata (file moved)
                            metadata_success, metadata_error = file_management_model.update_file_path(
                                old_path=blob.name,
                                new_path=destination_blob_name,
                                new_folder_path=dest_folder_path
                            )
                            if not metadata_success:
                                debug_log(f"WARNING: File {blob.name} transferred but metadata update failed: {metadata_error}")
                        elif operation == 'copy':
                            # Create new metadata for copied file
                            file_name = blob.name.split('/')[-1]
                            file_id, metadata_error = file_management_model.create_file_metadata(
                                file_path=destination_blob_name,
                                file_name=file_name,
                                folder_path=dest_folder_path,
                                gcs_uri=f"gs://{bucket_name}/{destination_blob_name}",
                                file_size=blob.size,
                                content_type=blob.content_type or 'application/pdf',
                                user_email=user_id or 'system'
                            )
                            if metadata_error:
                                debug_log(f"WARNING: File {blob.name} copied but metadata creation failed: {metadata_error}")
                    except Exception as meta_error:
                        debug_log(f"WARNING: Could not update/create file metadata for {blob.name}: {meta_error}")
                
                # Log individual file transfer to Firestore for history with "Folder" tag
                file_name = blob.name.split('/')[-1]
                debug_log(f"Logging successful file transfer: {file_name}")
                log_transfer_operation({
                    "fileName": file_name,
                    "source": blob.name,
                    "destination": destination_blob_name,
                    "operation": operation,
                    "status": "succeeded",
                    "transfer_type": "Folder"  # Simple tag to indicate this was part of folder transfer
                }, user_id)
                
                transferred_files.append({
                    "source": blob.name,
                    "destination": destination_blob_name,
                    "operation": operation
                })
                
                # Call progress callback after each file
                if progress_callback:
                    progress_callback(file_index + 1, total_items, f"Completed {current_file}")
                
            except Exception as file_error:
                error_msg = f"Error transferring {blob.name}: {str(file_error)}"
                debug_log(error_msg)
                errors.append(error_msg)
                
                # Log failed individual file transfer with "Folder" tag
                file_name = blob.name.split('/')[-1]
                debug_log(f"Logging failed file transfer: {file_name}")
                log_transfer_operation({
                    "fileName": file_name,
                    "source": blob.name,
                    "destination": destination_blob_name,
                    "operation": operation,
                    "status": "failed",
                    "error": str(file_error),
                    "transfer_type": "Folder"  # Simple tag to indicate this was part of folder transfer
                }, user_id)
                continue
        
        # If moving, delete all remaining folder markers and empty folders
        if operation == 'move':
            try:
                # Delete the main folder marker
                folder_marker = bucket.blob(source_folder_path)
                if folder_marker.exists():
                    folder_marker.delete()
                    debug_log(f"Deleted source folder marker: {source_folder_path}")
                
                # Also delete any remaining empty folder markers within the source path
                remaining_blobs = storage_client.list_blobs(bucket, prefix=source_folder_path)
                for remaining_blob in remaining_blobs:
                    if remaining_blob.name.endswith('/') and remaining_blob.size == 0:
                        remaining_blob.delete()
                        debug_log(f"Deleted remaining folder marker: {remaining_blob.name}")
                        
            except Exception as e:
                debug_log(f"Note: Could not delete folder markers for {source_folder_path}: {e}")
        
        # Create destination folder marker if it doesn't exist
        try:
            dest_folder_marker = bucket.blob(destination_folder_path)
            if not dest_folder_marker.exists():
                dest_folder_marker.upload_from_string('', content_type='application/x-directory')
                debug_log(f"Created destination folder marker: {destination_folder_path}")
        except Exception as e:
            debug_log(f"Note: Could not create destination folder marker {destination_folder_path}: {e}")
        
        # Update or create folder metadata based on operation
        try:
            from app.models import file_management_model
            
            # Extract folder name from destination path
            folder_name = destination_folder_path.rstrip('/').split('/')[-1]
            
            if operation == 'move':
                # Update folder path in metadata (folder moved)
                path_success, path_error = file_management_model.update_folder_path(
                    old_path=source_folder_path,
                    new_path=destination_folder_path
                )
                
                if not path_success:
                    debug_log(f"WARNING: Folder transferred but path update failed: {path_error}")
                    # Try to create new folder metadata if old one doesn't exist
                    # This handles cases where folder metadata wasn't created initially
                    try:
                        # Calculate parent path
                        parent_parts = destination_folder_path.rstrip('/').split('/')[:-1]
                        parent_path = '/'.join(parent_parts) + '/' if parent_parts else ''
                        
                        file_management_model.create_folder_metadata(
                            folder_path=destination_folder_path,
                            folder_name=folder_name,
                            parent_path=parent_path,
                            user_email=user_id or 'system'
                        )
                    except Exception as create_error:
                        debug_log(f"Note: Could not create folder metadata: {create_error}")
            elif operation == 'copy':
                # Create new folder metadata for copied folder
                try:
                    # Calculate parent path
                    parent_parts = destination_folder_path.rstrip('/').split('/')[:-1]
                    parent_path = '/'.join(parent_parts) + '/' if parent_parts else ''
                    
                    file_management_model.create_folder_metadata(
                        folder_path=destination_folder_path,
                        folder_name=folder_name,
                        parent_path=parent_path,
                        user_email=user_id or 'system'
                    )
                except Exception as create_error:
                    debug_log(f"Note: Could not create folder metadata for copied folder: {create_error}")
                
            # Also update folder name if needed (in case folder was renamed during transfer)
            # This is optional and may not always be needed
            
        except Exception as meta_error:
            debug_log(f"WARNING: Could not update/create folder metadata: {meta_error}")
        
        # Note: Folder transfer logging is now handled in gcs_routes.py
        # to ensure proper format with type="folder" and total_files fields
        
        result = {
            "message": f"Successfully {operation}d folder {source_folder_path} to {destination_folder_path}",
            "transferred_files": transferred_files,
            "transferred_folders": transferred_folders,
            "total_files": len(transferred_files),
            "total_folders": len(transferred_folders),
            "errors": errors,
            "status": "succeeded" if len(errors) == 0 else "partial"
        }
        
        debug_log(f"Folder transfer completed. Files transferred: {len(transferred_files)}, Folders transferred: {len(transferred_folders)}, Errors: {len(errors)}")
        return result, None if len(errors) == 0 else f"Completed with {len(errors)} errors"
        
    except exceptions.GoogleAPICallError as e:
        logger.error(f"GCS API error transferring folder: {e}")
        debug_error(f"Error in transfer_gcs_folder: {e}")
        return None, str(e)


def create_gcs_folder(bucket_name, folder_path):
    """Creates a new folder in GCS by creating an empty blob with trailing slash."""
    debug_log(f"--- create_gcs_folder ---")
    debug_log(f"Bucket: {bucket_name}, Folder Path: {folder_path}")
    
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        
        # Ensure folder path ends with /
        if not folder_path.endswith('/'):
            folder_path += '/'
        
        # Create an empty blob to represent the folder
        blob = bucket.blob(folder_path)
        
        # Check if folder already exists
        if blob.exists():
            return None, f"Folder '{folder_path}' already exists"
        
        # Upload empty content to create the folder marker
        blob.upload_from_string('', content_type='application/x-directory')
        
        debug_log(f"Created folder: {folder_path}")
        return {"message": f"Successfully created folder '{folder_path}'", "folderPath": folder_path}, None
        
    except exceptions.GoogleAPICallError as e:
        logger.error(f"GCS API error creating folder: {e}")
        debug_error(f"Error in create_gcs_folder: {e}")
        return None, str(e)


def get_transfer_status(transfer_id):
    """Retrieves the status of a transfer."""
    debug_log(f"--- get_transfer_status ---")
    debug_log(f"Transfer ID: {transfer_id}")
    try:
        transfer_ref = db.collection('gcs_transfers').document(transfer_id)
        transfer = transfer_ref.get()
        if transfer.exists:
            return transfer.to_dict(), None
        else:
            return None, "Transfer not found."
    except Exception as e:
        logger.error(f"Error getting transfer status: {e}")
        return None, str(e)

def get_user_transfers():
    """Retrieves all transfers for the current user."""
    try:
        user_id = get_jwt_identity()
        transfers_ref = db.collection('gcs_transfers').where('user_id', '==', user_id).order_by('timestamp', direction=firestore.Query.DESCENDING)
        transfers = [doc.to_dict() for doc in transfers_ref.stream()]
        return transfers, None
    except Exception as e:
        logger.error(f"Error getting user transfers: {e}")
        return None, str(e)

def get_user_transfers_paginated(user_id=None, limit=50, page=1, status=None, operation=None, transfer_type=None, date=None, start_date=None, end_date=None):
    """
    Retrieves transfers for a specific user with pagination and filtering.
    
    Args:
        user_id: The user ID to filter transfers by
        limit: Maximum number of transfers to return per page (default: 50, max: 200)
        page: Page number (1-based, default: 1)
        status: Filter by transfer status (e.g., 'succeeded', 'failed', 'running')
        operation: Filter by operation type ('move' or 'copy')
        transfer_type: Filter by transfer type ('Folder' for folder transfers, 'File' for file transfers)
        date: Filter by specific date (ISO format: YYYY-MM-DD). Filters transfers on this exact date
        start_date: Start date for date range filter (ISO format: YYYY-MM-DD). Inclusive.
        end_date: End date for date range filter (ISO format: YYYY-MM-DD). Inclusive.
    
    Returns:
        Tuple of (transfers list, pagination_info dict, error) or (None, None, error)
        pagination_info contains: current_page, total_pages, total_items, page_size, has_next, has_prev
    """
    try:
        # Validate and clamp limit
        limit = max(1, min(limit, 200))
        page = max(1, page)
        
        # Build base query
        transfers_ref = db.collection('gcs_transfers')
        
        # Only filter by user_id if provided
        if user_id:
            transfers_ref = transfers_ref.where('user_id', '==', user_id)
        
        # Track if we need to filter for File transfers (which have missing transfer_type field)
        filter_file_transfers_in_memory = False
        if transfer_type and transfer_type.lower() == 'file':
            # File transfers don't have the transfer_type field (it's missing, not None)
            # Firestore can't query for missing fields, so we'll filter in memory
            filter_file_transfers_in_memory = True
        else:
            # Apply transfer_type filter for Folder or other types
            if transfer_type:
                transfers_ref = transfers_ref.where('transfer_type', '==', transfer_type)
        
        # Apply other filters
        if status:
            transfers_ref = transfers_ref.where('status', '==', status)
        if operation:
            transfers_ref = transfers_ref.where('operation', '==', operation)
        
        # Handle date filters
        if date:
            # Filter by specific date (entire day)
            try:
                date_obj = datetime.datetime.strptime(date, '%Y-%m-%d')
                # Start of day (00:00:00)
                start_of_day = datetime.datetime.combine(date_obj.date(), datetime.time.min)
                start_of_day = start_of_day.replace(tzinfo=datetime.timezone.utc)
                # End of day (23:59:59.999999)
                end_of_day = datetime.datetime.combine(date_obj.date(), datetime.time.max)
                end_of_day = end_of_day.replace(tzinfo=datetime.timezone.utc)
                
                transfers_ref = transfers_ref.where('timestamp', '>=', start_of_day).where('timestamp', '<=', end_of_day)
            except ValueError as e:
                logger.warning(f"Invalid date format: {date}, error: {e}")
                return None, None, f"Invalid date format. Expected YYYY-MM-DD, got: {date}"
        elif start_date or end_date:
            # Date range filter
            if start_date:
                try:
                    start_date_obj = datetime.datetime.strptime(start_date, '%Y-%m-%d')
                    start_datetime = datetime.datetime.combine(start_date_obj.date(), datetime.time.min)
                    start_datetime = start_datetime.replace(tzinfo=datetime.timezone.utc)
                    transfers_ref = transfers_ref.where('timestamp', '>=', start_datetime)
                except ValueError as e:
                    logger.warning(f"Invalid start_date format: {start_date}, error: {e}")
                    return None, None, f"Invalid start_date format. Expected YYYY-MM-DD, got: {start_date}"
            
            if end_date:
                try:
                    end_date_obj = datetime.datetime.strptime(end_date, '%Y-%m-%d')
                    end_datetime = datetime.datetime.combine(end_date_obj.date(), datetime.time.max)
                    end_datetime = end_datetime.replace(tzinfo=datetime.timezone.utc)
                    transfers_ref = transfers_ref.where('timestamp', '<=', end_datetime)
                except ValueError as e:
                    logger.warning(f"Invalid end_date format: {end_date}, error: {e}")
                    return None, None, f"Invalid end_date format. Expected YYYY-MM-DD, got: {end_date}"
        
        # Build count query with same filters (without ordering/pagination)
        count_query = db.collection('gcs_transfers')
        if user_id:
            count_query = count_query.where('user_id', '==', user_id)
        
        # Apply same filters to count query (except transfer_type for File, which we'll filter in memory)
        if status:
            count_query = count_query.where('status', '==', status)
        if operation:
            count_query = count_query.where('operation', '==', operation)
        if transfer_type and not filter_file_transfers_in_memory:
            # Only apply transfer_type filter if not filtering for File (which requires in-memory filtering)
            count_query = count_query.where('transfer_type', '==', transfer_type)
        
        # Handle date filters for count query
        if date:
            try:
                date_obj = datetime.datetime.strptime(date, '%Y-%m-%d')
                start_of_day = datetime.datetime.combine(date_obj.date(), datetime.time.min)
                start_of_day = start_of_day.replace(tzinfo=datetime.timezone.utc)
                end_of_day = datetime.datetime.combine(date_obj.date(), datetime.time.max)
                end_of_day = end_of_day.replace(tzinfo=datetime.timezone.utc)
                count_query = count_query.where('timestamp', '>=', start_of_day).where('timestamp', '<=', end_of_day)
            except ValueError:
                pass  # Already validated above
        elif start_date or end_date:
            if start_date:
                try:
                    start_date_obj = datetime.datetime.strptime(start_date, '%Y-%m-%d')
                    start_datetime = datetime.datetime.combine(start_date_obj.date(), datetime.time.min)
                    start_datetime = start_datetime.replace(tzinfo=datetime.timezone.utc)
                    count_query = count_query.where('timestamp', '>=', start_datetime)
                except ValueError:
                    pass  # Already validated above
            if end_date:
                try:
                    end_date_obj = datetime.datetime.strptime(end_date, '%Y-%m-%d')
                    end_datetime = datetime.datetime.combine(end_date_obj.date(), datetime.time.max)
                    end_datetime = end_datetime.replace(tzinfo=datetime.timezone.utc)
                    count_query = count_query.where('timestamp', '<=', end_datetime)
                except ValueError:
                    pass  # Already validated above
        
        # Order by timestamp descending (apply to both queries)
        transfers_ref = transfers_ref.order_by('timestamp', direction=firestore.Query.DESCENDING)
        count_query_ordered = count_query.order_by('timestamp', direction=firestore.Query.DESCENDING)
        
        # Get total count and fetch documents for File transfers
        # If filtering for File transfers, we need to count and fetch in memory
        filtered_docs = None  # Will store filtered documents for File transfers
        if filter_file_transfers_in_memory:
            # For File transfers, we need to fetch and filter in memory for accurate count
            # This is less efficient but necessary since Firestore can't query for missing fields
            # Note: We fetch all matching documents which could be memory-intensive for very large datasets
            # For production with millions of documents, consider adding a migration to set transfer_type: None
            # for all file transfers, which would allow using Firestore's native count queries
            try:
                # Fetch all matching documents (without pagination) to count and use for results
                all_docs = list(count_query_ordered.stream())
                # Filter out Folder transfers (keep documents where transfer_type is missing or not "Folder")
                filtered_docs = [
                    doc for doc in all_docs 
                    if doc.to_dict().get('transfer_type') != 'Folder'
                ]
                total_items = len(filtered_docs)
            except Exception as e:
                logger.warning(f"Error getting total count for File transfers: {e}, falling back to None")
                total_items = None
                filtered_docs = None
        else:
            try:
                total_items_snapshot = count_query.count().get()
                total_items = total_items_snapshot[0][0].value if total_items_snapshot else 0
            except Exception as e:
                logger.warning(f"Error getting total count: {e}, falling back to None")
                total_items = None
        
        # For File transfers, use the pre-filtered documents; otherwise fetch normally
        if filter_file_transfers_in_memory and filtered_docs is not None:
            # Apply pagination after filtering
            offset = (page - 1) * limit
            paginated_docs = filtered_docs[offset:offset + limit + 1]
            
            # Check if there are more results
            has_more = len(paginated_docs) > limit
            if has_more:
                paginated_docs = paginated_docs[:limit]
            elif total_items is not None:
                # Use total_items to determine has_more accurately
                has_more = (offset + len(paginated_docs)) < total_items
            
            docs = paginated_docs
        else:
            # Normal pagination for non-File transfers
            offset = (page - 1) * limit
            
            # Apply pagination
            if offset > 0:
                transfers_ref = transfers_ref.offset(offset)
            
            # Limit results
            transfers_ref = transfers_ref.limit(limit + 1)  # Fetch one extra to check if there's more
            
            # Execute query
            docs = list(transfers_ref.stream())
            
            # Check if there are more results
            has_more = len(docs) > limit
            if has_more:
                docs = docs[:limit]
        
        # Convert to dictionaries
        transfers = []
        for doc in docs:
            transfer_data = doc.to_dict()
            # Ensure id is included
            if 'id' not in transfer_data:
                transfer_data['id'] = doc.id
            
            # Check if can_undo is already set in the database (e.g., set to false after undo)
            # If it exists in DB, use that value; otherwise calculate it dynamically
            if 'can_undo' in transfer_data:
                # Use the stored value from database (was explicitly set, e.g., after undo)
                can_undo = transfer_data['can_undo']
            else:
                # Calculate can_undo dynamically for transfers that don't have it stored
                # can_undo = True only if:
                # 1. Status is 'succeeded'
                # 2. Operation is 'move' (can't undo copy)
                # 3. File has NOT been batch processed
                can_undo = False
                if (transfer_data.get('status') == 'succeeded' and 
                    transfer_data.get('operation') == 'move'):
                    destination_path = transfer_data.get('destination')
                    if destination_path:
                        # Check if file has been batch processed
                        try:
                            is_processed = is_file_batch_processed(
                                config.FILE_MANAGEMENT_BUCKET_NAME, 
                                destination_path
                            )
                            can_undo = not is_processed
                        except Exception as e:
                            logger.warning(f"Error checking batch status for {destination_path}: {e}")
                            can_undo = False
                
                transfer_data['can_undo'] = can_undo
            
            transfers.append(transfer_data)
        
        # Calculate total_pages from total_items
        total_pages = None
        if total_items is not None:
            total_pages = (total_items + limit - 1) // limit  # Ceiling division
        
        # Build pagination info
        has_prev = page > 1
        
        pagination_info = {
            'current_page': page,
            'page_size': limit,
            'has_next': has_more,
            'has_prev': has_prev,
            'total_items': total_items,
            'total_pages': total_pages
        }
        
        return transfers, pagination_info, None
        
    except Exception as e:
        logger.error(f"Error getting user transfers with pagination: {e}")
        return None, None, str(e)

def log_transfer_operation(details, user_id=None):
    """Logs the details of a transfer to Firestore."""
    try:
        # Use provided user_id or get from JWT context
        if user_id is None:
            user_id = get_jwt_identity()
        
        # If still no user_id, log the issue and return None
        if user_id is None:
            logger.error("No user_id available for transfer logging (JWT context lost)")
            debug_error(f" Cannot log transfer - no user_id available: {details}")
            return None
            
        transfer_id = str(uuid.uuid4())
        details.update({
            "id": transfer_id,
            "user_id": user_id,
            "timestamp": firestore.SERVER_TIMESTAMP,
        })
        db.collection('gcs_transfers').document(transfer_id).set(details)
        logger.info(f"Logged transfer: {details}")
        debug_log(f"SUCCESS: Logged transfer with ID {transfer_id}: {details['fileName']}")
        return transfer_id
    except Exception as e:
        logger.error(f"Error logging transfer: {e}")
        debug_error(f" Failed to log transfer: {e}")
        return None

    


@retry(stop=stop_after_attempt(config.GCS_DOWNLOAD_MAX_RETRY), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
def download_blob_to_bytes(gcs_uri: str) -> bytes: # Return bytes or raise Exception
    """Downloads a file from GCS URI to bytes with retry logic."""
    logger.info(f"Attempting to download blob: {gcs_uri}")
    try:
        if not gcs_uri.startswith("gs://"):
            # This is a config/logic error, not a transient network issue.
            # Raising ValueError directly without retry for this specific case.
            logger.error(f"Invalid GCS URI format: {gcs_uri}")
            raise ValueError(f"Invalid GCS URI format: {gcs_uri}")
            
        path_parts = gcs_uri[5:].split("/", 1)
        if len(path_parts) < 2: # Ensure blob_name part exists
            logger.error(f"Invalid GCS URI, missing blob name: {gcs_uri}")
            raise ValueError(f"Invalid GCS URI, missing blob name: {gcs_uri}")

        bucket_name = path_parts[0]
        blob_name = path_parts[1]

        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        if not blob.exists():
            logger.error(f"Blob not found at {gcs_uri}. Cannot download.")
            raise FileNotFoundError(f"Blob not found: {gcs_uri}")

        content = blob.download_as_bytes()
        logger.info(f"Successfully downloaded blob {gcs_uri}")
        return content
    except (api_core_exceptions.GoogleAPICallError, api_core_exceptions.RetryError, TimeoutError) as e:
        logger.warning(f"Retrying download for {gcs_uri} due to potentially transient error: {e}")
        raise 
    except FileNotFoundError as e: # Catch specific FileNotFoundError
         logger.error(f"Download failed permanently for {gcs_uri}: {e}")
         raise 
    except ValueError as e: # Catch specific ValueError for URI format
        logger.error(f"Download failed due to invalid GCS URI: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error downloading blob {gcs_uri}: {e}", exc_info=True)
        raise


# ============================================================================
# Undo and Delete Functions for File Management
# ============================================================================

def is_file_batch_processed(bucket_name, file_path):
    """
    Check if a file has been processed by batch processing.
    Returns True if file exists in batch_processed_files collection.
    
    Args:
        bucket_name: Name of the GCS bucket
        file_path: GCS path of the file (e.g., "folder/subfolder/file.pdf")
    
    Returns:
        True if file has been batch processed, False otherwise
    """
    try:
        file_gcs_path = f"gs://{bucket_name}/{file_path}"
        query = db.collection('batch_processed_files').where('file_gcs_path', '==', file_gcs_path).limit(1)
        docs = list(query.stream())
        is_processed = len(docs) > 0
        logger.info(f"File {file_path} batch processed status: {is_processed}")
        return is_processed
    except Exception as e:
        logger.error(f"Error checking batch processing status for {file_path}: {e}", exc_info=True)
        # On error, assume processed to be safe (prevent accidental undo/delete)
        return True


def get_transfer_record_by_destination(destination_path, user_id=None):
    """
    Find the most recent transfer record for a destination file.
    
    Args:
        destination_path: GCS path of the destination file
        user_id: Optional user ID to filter transfers (if None, uses current JWT user)
    
    Returns:
        Transfer record dictionary or None if not found
    """
def get_transfer_record_by_id(transfer_id: str, user_id: Optional[str] = None):
    """
    Get a specific transfer record by its unique ID.
    
    Args:
        transfer_id: Unique transfer ID
        user_id: Optional user ID to restrict search
        
    Returns:
        Transfer record dictionary or None if not found
    """
    try:
        # First try to find by the 'id' field within the document
        query = db.collection('gcs_transfers').where('id', '==', transfer_id)
        if user_id:
            query = query.where('user_id', '==', user_id)
        
        docs = list(query.limit(1).stream())
        if docs:
            return docs[0].to_dict()
        
        # If not found, try by document ID itself
        doc_ref = db.collection('gcs_transfers').document(transfer_id)
        doc = doc_ref.get()
        if doc.exists:
            record = doc.to_dict()
            # If user_id is provided, verify ownership
            if user_id and record.get('user_id') != user_id:
                logger.warning(f"Unauthorized access attempt to transfer {transfer_id} by user {user_id}")
                return None
            return record
            
        logger.warning(f"No transfer record found for ID: {transfer_id}")
        return None
    except Exception as e:
        logger.error(f"Error getting transfer record {transfer_id}: {e}", exc_info=True)
        return None


def get_transfer_record_by_destination(destination_path, user_id=None):
    """
    Find the most recent transfer record for a specific destination path.
    Useful for backward compatibility or when ID is not available.
    """
    try:
        query = db.collection('gcs_transfers').where('destination', '==', destination_path)
        if user_id:
            query = query.where('user_id', '==', user_id)
        query = query.order_by('timestamp', direction=firestore.Query.DESCENDING).limit(1)
        
        docs = list(query.stream())
        if docs:
            return docs[0].to_dict()
        return None
    except Exception as e:
        logger.error(f"Error getting transfer record by destination: {e}")
        return None


def update_transfer_can_undo(transfer_id: str, can_undo: bool = False) -> bool:
    """
    Update the can_undo field for a transfer record.
    
    Args:
        transfer_id: Unique transfer ID
        can_undo: Boolean value to set for can_undo field (default: False after undo)
    
    Returns:
        True if update successful, False otherwise
    """
    try:
        logger.info(f"Updating can_undo={can_undo} for transfer {transfer_id}")
        
        # Try updating by document ID first
        doc_ref = db.collection('gcs_transfers').document(transfer_id)
        doc = doc_ref.get()
        
        if doc.exists:
            # Update the document directly by ID
            doc_ref.update({'can_undo': can_undo, 'updated_at': firestore.SERVER_TIMESTAMP})
            logger.info(f"Successfully updated can_undo field for transfer {transfer_id}")
            debug_log(f"Transfer {transfer_id} can_undo updated to {can_undo}")
            return True
        else:
            # If not found by document ID, try to find by 'id' field
            query = db.collection('gcs_transfers').where('id', '==', transfer_id).limit(1)
            docs = list(query.stream())
            
            if docs:
                # Update the document found by query
                docs[0].reference.update({'can_undo': can_undo, 'updated_at': firestore.SERVER_TIMESTAMP})
                logger.info(f"Successfully updated can_undo field for transfer {transfer_id}")
                debug_log(f"Transfer {transfer_id} can_undo updated to {can_undo}")
                return True
            else:
                logger.warning(f"Transfer record not found for ID: {transfer_id}")
                return False
                
    except Exception as e:
        logger.error(f"Error updating can_undo for transfer {transfer_id}: {e}", exc_info=True)
        debug_error(f"Failed to update can_undo for transfer {transfer_id}: {e}")
        return False


def undo_file_transfer(bucket_name, transfer_id, user_id=None):
    """
    Undo a file transfer by moving it back to source.
    Only works if file hasn't been batch processed.
    
    Args:
        bucket_name: Name of the GCS bucket
        transfer_id: Unique ID of the transfer to undo
        user_id: Optional user ID (if None, uses current JWT user)
    
    Returns:
        Tuple of (result_dict, error_message). result_dict is None if error occurred.
    """
    try:
        logger.info(f"Attempting to undo transfer with ID: {transfer_id}")
        
        # Get the transfer record
        transfer_record = get_transfer_record_by_id(transfer_id, user_id)
        if not transfer_record:
            error_msg = "No transfer record found for this ID"
            logger.warning(f"Cannot undo transfer {transfer_id}: {error_msg}")
            return None, error_msg
            
        destination_path = transfer_record.get('destination')
        if not destination_path:
            error_msg = "Destination path not found in transfer record"
            logger.warning(f"Cannot undo transfer {transfer_id}: {error_msg}")
            return None, error_msg

        # Check if file has been batch processed
        if is_file_batch_processed(bucket_name, destination_path):
            error_msg = "File has already been processed by batch processing and cannot be undone"
            logger.warning(f"Cannot undo {destination_path}: {error_msg}")
            return None, error_msg
        
        source_path = transfer_record.get('source')
        if not source_path:
            error_msg = "Source path not found in transfer record"
            logger.warning(f"Cannot undo {destination_path}: {error_msg}")
            return None, error_msg
        
        # Check if destination file exists
        storage_client_instance = storage.Client()
        bucket = storage_client_instance.bucket(bucket_name)
        dest_blob = bucket.blob(destination_path)
        
        if not dest_blob.exists():
            error_msg = "Destination file not found"
            logger.warning(f"Cannot undo {destination_path}: {error_msg}")
            return None, error_msg
        
        # Move file back to source
        source_blob = bucket.blob(source_path)
        source_blob.rewrite(dest_blob)
        dest_blob.delete()
        logger.info(f"File moved from {destination_path} back to {source_path}")
        
        # Update file metadata
        from app.models import file_management_model
        
        source_folder_path = '/'.join(source_path.split('/')[:-1])
        if source_folder_path:
            source_folder_path += '/'
        
        # Update metadata to reflect file is back at source
        metadata_success, metadata_error = file_management_model.update_file_path(
            old_path=destination_path,
            new_path=source_path,
            new_folder_path=source_folder_path
        )
        
        if not metadata_success:
            logger.warning(f"File moved back but metadata update failed: {metadata_error}")
        
        # Log the undo operation
        if user_id is None:
            user_id = get_jwt_identity()
        
        log_transfer_operation({
            "fileName": source_path.split('/')[-1],
            "source": destination_path,
            "destination": source_path,
            "operation": "undo",
            "status": "succeeded",
            "original_transfer_id": transfer_record.get('id'),
            "transfer_type": None
        }, user_id)
        
        # Update the original transfer record to mark can_undo as False
        undo_success = update_transfer_can_undo(transfer_id, can_undo=False)
        if not undo_success:
            logger.warning(f"Failed to update can_undo field for transfer {transfer_id}, but file was successfully moved")
        else:
            logger.info(f"Successfully updated can_undo field to False for transfer {transfer_id}")
        
        result = {
            "message": "File successfully moved back to source",
            "source_path": source_path,
            "destination_path": destination_path
        }
        logger.info(f"Successfully undone transfer for {destination_path}")
        return result, None
        
    except Exception as e:
        logger.error(f"Error undoing file transfer for {destination_path}: {e}", exc_info=True)
        return None, str(e)


def delete_destination_file(bucket_name, destination_path, user_id=None):
    """
    Delete a file from destination.
    Only works if file hasn't been batch processed.
    
    Args:
        bucket_name: Name of the GCS bucket
        destination_path: GCS path of the file to delete
        user_id: Optional user ID (if None, uses current JWT user)
    
    Returns:
        Tuple of (result_dict, error_message). result_dict is None if error occurred.
    """
    try:
        logger.info(f"Attempting to delete destination file: {destination_path}")
        
        # Check if file has been batch processed
        if is_file_batch_processed(bucket_name, destination_path):
            error_msg = "File has already been processed by batch processing and cannot be deleted"
            logger.warning(f"Cannot delete {destination_path}: {error_msg}")
            return None, error_msg
        
        storage_client_instance = storage.Client()
        bucket = storage_client_instance.bucket(bucket_name)
        dest_blob = bucket.blob(destination_path)
        
        if not dest_blob.exists():
            error_msg = "File not found"
            logger.warning(f"Cannot delete {destination_path}: {error_msg}")
            return None, error_msg
        
        # Delete from GCS
        dest_blob.delete()
        logger.info(f"File deleted from GCS: {destination_path}")
        
        # Delete metadata
        from app.models import file_management_model
        try:
            metadata_success, metadata_error = file_management_model.delete_file_metadata(destination_path)
            if not metadata_success:
                logger.warning(f"File deleted from GCS but metadata deletion failed: {metadata_error}")
        except Exception as e:
            logger.warning(f"Error deleting file metadata: {e}")
        
        # Log the delete operation
        if user_id is None:
            user_id = get_jwt_identity()
        
        log_transfer_operation({
            "fileName": destination_path.split('/')[-1],
            "source": None,
            "destination": destination_path,
            "operation": "delete",
            "status": "succeeded",
            "transfer_type": None
        }, user_id)
        
        result = {
            "message": "File successfully deleted",
            "file_path": destination_path
        }
        logger.info(f"Successfully deleted file {destination_path}")
        return result, None
        
    except Exception as e:
        logger.error(f"Error deleting file {destination_path}: {e}", exc_info=True)
        return None, str(e)


def check_transfer_conflicts(bucket_name, source_paths, dest_root, dest_path):
    """
    Checks if any files in source_paths already exist in the destination.
    Returns a list of conflicting filenames.
    """
    conflicts = []
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)

        # Ensure destination path ends with / if it's a folder path (and not empty)
        full_dest_path = f"{dest_root}{dest_path}"
        # Typically dest_root ends with /, dest_path might or might not.
        # If transferring files, we are putting them INTO a folder.
        # So destination must be a folder.
        if full_dest_path and not full_dest_path.endswith('/'):
            full_dest_path += '/'

        for source_path in source_paths:
            # Check if source is a file or folder
            if source_path.endswith('/'):
                # Skip folders for conflict check for now
                continue
            
            filename = source_path.split('/')[-1]
            destination_blob_name = f"{full_dest_path}{filename}"
            
            blob = bucket.blob(destination_blob_name)
            if blob.exists():
                conflicts.append(filename)
                
        return conflicts, None
    except Exception as e:
        logger.error(f"Error checking transfer conflicts: {e}", exc_info=True)
        return None, str(e)
