# backend/app/models/file_management_model.py
import logging
from typing import Optional, Tuple, List, Dict
from app import db
import datetime
from concurrent.futures import ThreadPoolExecutor

# Reference to the 'file_management_folders' collection in Firestore
folders_ref = db.collection('file_management_folders')
# Reference to the 'file_management_files' collection in Firestore
files_ref = db.collection('file_management_files')

logger = logging.getLogger(__name__)


def create_folder_metadata(folder_path: str, folder_name: str, parent_path: str, user_email: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Creates a new folder metadata record in Firestore.
    
    Args:
        folder_path: Full GCS path of the folder (e.g., "Pending Files/MyFolder/")
        folder_name: Name of the folder
        parent_path: Path of the parent folder (empty string for root)
        user_email: Email of the user creating the folder
    
    Returns:
        Tuple of (folder_id, error_message). folder_id is None if creation failed.
    """
    try:
        folder_data = {
            'folder_path': folder_path,
            'folder_name': folder_name,
            'folder_name_lower': folder_name.lower(),  # Add lowercase field for case-insensitive search
            'parent_path': parent_path,
            'created_by': user_email,
            'created_at': datetime.datetime.now(tz=datetime.timezone.utc),
            'updated_at': datetime.datetime.now(tz=datetime.timezone.utc),
        }
        
        # Use folder_path as document ID (sanitized) or let Firestore generate one
        # Sanitize folder_path to create a valid document ID
        doc_id = folder_path.replace('/', '_').replace(' ', '_').strip('_')
        if not doc_id:
            doc_id = None  # Let Firestore generate ID
        
        if doc_id:
            doc_ref = folders_ref.document(doc_id)
            # Check if folder already exists
            if doc_ref.get().exists:
                return None, f"Folder metadata already exists for path: {folder_path}"
            doc_ref.set(folder_data)
            logger.info(f"Folder metadata created for {folder_name} with ID: {doc_id}")
            return doc_id, None
        else:
            doc_ref = folders_ref.document()
            doc_ref.set(folder_data)
            logger.info(f"Folder metadata created for {folder_name} with ID: {doc_ref.id}")
            return doc_ref.id, None
            
    except Exception as e:
        logger.error(f"ERROR: Failed to create folder metadata for {folder_name}: {e}", exc_info=True)
        return None, f"Failed to create folder metadata: {e}"


def update_folder_name(folder_path: str, new_folder_name: str, user_email: str, new_folder_path: Optional[str] = None) -> Tuple[bool, Optional[str]]:
    """
    Updates the name of a folder in Firestore, optionally updating the path as well.
    
    Args:
        folder_path: Current GCS path of the folder
        new_folder_name: New name for the folder
        user_email: Email of the user renaming the folder
        new_folder_path: Optional new path for the folder (if provided, both name and path are updated)
    
    Returns:
        Tuple of (success, error_message)
    """
    try:
        # Find folder by path
        query = folders_ref.where('folder_path', '==', folder_path).limit(1)
        docs = list(query.stream())
        
        if not docs:
            return False, f"Folder not found: {folder_path}"
        
        doc_ref = folders_ref.document(docs[0].id)
        update_data = {
            'folder_name': new_folder_name,
            'folder_name_lower': new_folder_name.lower(),  # Update lowercase field
            'updated_at': datetime.datetime.now(tz=datetime.timezone.utc),
            'updated_by': user_email
        }
        
        # If new path is provided, update it as well
        if new_folder_path:
            update_data['folder_path'] = new_folder_path
        
        doc_ref.update(update_data)
        
        logger.info(f"Folder renamed: {folder_path} -> {new_folder_name}" + (f" (path: {new_folder_path})" if new_folder_path else ""))
        return True, None
        
    except Exception as e:
        logger.error(f"ERROR: Failed to rename folder {folder_path}: {e}", exc_info=True)
        return False, f"Failed to rename folder: {e}"


def update_folder_path(old_path: str, new_path: str) -> Tuple[bool, Optional[str]]:
    """
    Updates the path of a folder in Firestore (used when folder is moved/renamed in GCS).
    
    Args:
        old_path: Old GCS path of the folder
        new_path: New GCS path of the folder
    
    Returns:
        Tuple of (success, error_message)
    """
    try:
        # Find folder by old path
        query = folders_ref.where('folder_path', '==', old_path).limit(1)
        docs = list(query.stream())
        
        if not docs:
            return False, f"Folder not found: {old_path}"
        
        doc_ref = folders_ref.document(docs[0].id)
        doc_ref.update({
            'folder_path': new_path,
            'updated_at': datetime.datetime.now(tz=datetime.timezone.utc)
        })
        
        logger.info(f"Folder path updated: {old_path} -> {new_path}")
        return True, None
        
    except Exception as e:
        logger.error(f"ERROR: Failed to update folder path {old_path}: {e}", exc_info=True)
        return False, f"Failed to update folder path: {e}"


def create_file_metadata(
    file_path: str,
    file_name: str,
    folder_path: str,
    gcs_uri: str,
    file_size: int,
    content_type: str,
    user_email: str
) -> Tuple[Optional[str], Optional[str]]:
    """
    Creates a new file metadata record in Firestore.
    
    Args:
        file_path: Full GCS path of the file
        file_name: Name of the file
        folder_path: Path of the folder containing the file
        gcs_uri: GCS URI of the file
        file_size: Size of the file in bytes
        content_type: MIME type of the file
        user_email: Email of the user uploading the file
    
    Returns:
        Tuple of (file_id, error_message). file_id is None if creation failed.
    """
    try:
        file_data = {
            'file_path': file_path,
            'file_name': file_name,
            'file_name_lower': file_name.lower(),  # Add lowercase field for case-insensitive search
            'folder_path': folder_path,
            'gcs_uri': gcs_uri,
            'file_size_bytes': file_size,
            'content_type': content_type,
            'uploaded_by': user_email,
            'created_at': datetime.datetime.now(tz=datetime.timezone.utc),
            'updated_at': datetime.datetime.now(tz=datetime.timezone.utc),
        }
        
        # Use file_path as document ID (sanitized) or let Firestore generate one
        doc_id = file_path.replace('/', '_').replace(' ', '_').strip('_')
        if not doc_id:
            doc_id = None
        
        if doc_id:
            doc_ref = files_ref.document(doc_id)
            # Check if file already exists
            if doc_ref.get().exists:
                # Update existing file metadata
                file_data['updated_at'] = datetime.datetime.now(tz=datetime.timezone.utc)
                doc_ref.update(file_data)
                logger.info(f"File metadata updated for {file_name} with ID: {doc_id}")
                return doc_id, None
            doc_ref.set(file_data)
            logger.info(f"File metadata created for {file_name} with ID: {doc_id}")
            return doc_id, None
        else:
            doc_ref = files_ref.document()
            doc_ref.set(file_data)
            logger.info(f"File metadata created for {file_name} with ID: {doc_ref.id}")
            return doc_ref.id, None
            
    except Exception as e:
        logger.error(f"ERROR: Failed to create file metadata for {file_name}: {e}", exc_info=True)
        return None, f"Failed to create file metadata: {e}"


def update_file_path(old_path: str, new_path: str, new_folder_path: str) -> Tuple[bool, Optional[str]]:
    """
    Updates the path of a file in Firestore (used when file is moved).
    
    Args:
        old_path: Old GCS path of the file
        new_path: New GCS path of the file
        new_folder_path: New folder path containing the file
    
    Returns:
        Tuple of (success, error_message)
    """
    try:
        # Find file by old path
        query = files_ref.where('file_path', '==', old_path).limit(1)
        docs = list(query.stream())
        
        if not docs:
            return False, f"File not found: {old_path}"
        
        doc_ref = files_ref.document(docs[0].id)
        doc_ref.update({
            'file_path': new_path,
            'folder_path': new_folder_path,
            'updated_at': datetime.datetime.now(tz=datetime.timezone.utc)
        })
        
        logger.info(f"File path updated: {old_path} -> {new_path}")
        return True, None
        
    except Exception as e:
        logger.error(f"ERROR: Failed to update file path {old_path}: {e}", exc_info=True)
        return False, f"Failed to update file path: {e}"


def update_file_name_and_path(old_path: str, new_path: str, new_file_name: str, new_folder_path: str) -> Tuple[bool, Optional[str]]:
    """
    Updates the name and path of a file in Firestore (used when file is renamed).
    
    Args:
        old_path: Old GCS path of the file
        new_path: New GCS path of the file
        new_file_name: New name of the file
        new_folder_path: New folder path containing the file
    
    Returns:
        Tuple of (success, error_message)
    """
    try:
        # Find file by old path
        query = files_ref.where('file_path', '==', old_path).limit(1)
        docs = list(query.stream())
        
        if not docs:
            return False, f"File not found: {old_path}"
        
        doc_ref = files_ref.document(docs[0].id)
        doc_ref.update({
            'file_path': new_path,
            'file_name': new_file_name,
            'file_name_lower': new_file_name.lower(),  # Update lowercase field
            'folder_path': new_folder_path,
            'updated_at': datetime.datetime.now(tz=datetime.timezone.utc)
        })
        
        logger.info(f"File name and path updated: {old_path} -> {new_path} (name: {new_file_name})")
        return True, None
        
    except Exception as e:
        logger.error(f"ERROR: Failed to update file name and path {old_path}: {e}", exc_info=True)
        return False, f"Failed to update file name and path: {e}"


def get_folder_metadata(folder_path: str) -> Tuple[Optional[dict], Optional[str]]:
    """
    Retrieves folder metadata by path.
    
    Args:
        folder_path: Full GCS path of the folder
    
    Returns:
        Tuple of (folder_data, error_message)
    """
    try:
        query = folders_ref.where('folder_path', '==', folder_path).limit(1)
        docs = list(query.stream())
        
        if not docs:
            return None, f"Folder not found: {folder_path}"
        
        folder_data = docs[0].to_dict()
        folder_data['id'] = docs[0].id
        
        # Convert datetime to ISO format
        if 'created_at' in folder_data and isinstance(folder_data['created_at'], datetime.datetime):
            folder_data['created_at'] = folder_data['created_at'].isoformat()
        if 'updated_at' in folder_data and isinstance(folder_data['updated_at'], datetime.datetime):
            folder_data['updated_at'] = folder_data['updated_at'].isoformat()
        
        # Add aliases for API consistency: created_date, updated_date
        folder_data['created_date'] = folder_data.get('created_at')
        folder_data['updated_date'] = folder_data.get('updated_at')
        
        return folder_data, None
        
    except Exception as e:
        logger.error(f"ERROR: Failed to get folder metadata for {folder_path}: {e}", exc_info=True)
        return None, f"Failed to get folder metadata: {e}"


def get_file_metadata(file_path: str) -> Tuple[Optional[dict], Optional[str]]:
    """
    Retrieves file metadata by path.
    
    Args:
        file_path: Full GCS path of the file
    
    Returns:
        Tuple of (file_data, error_message)
    """
    try:
        query = files_ref.where('file_path', '==', file_path).limit(1)
        docs = list(query.stream())
        
        if not docs:
            return None, f"File not found: {file_path}"
        
        file_data = docs[0].to_dict()
        file_data['id'] = docs[0].id
        
        # Convert datetime to ISO format
        if 'created_at' in file_data and isinstance(file_data['created_at'], datetime.datetime):
            file_data['created_at'] = file_data['created_at'].isoformat()
        if 'updated_at' in file_data and isinstance(file_data['updated_at'], datetime.datetime):
            file_data['updated_at'] = file_data['updated_at'].isoformat()
        
        # Add aliases for API consistency: created_date, updated_date, size
        file_data['created_date'] = file_data.get('created_at')
        file_data['updated_date'] = file_data.get('updated_at')
        # Use file_size_bytes if available, otherwise use size if it exists
        if 'file_size_bytes' in file_data:
            file_data['size'] = file_data['file_size_bytes']
        elif 'size' not in file_data:
            file_data['size'] = None
        
        return file_data, None
        
    except Exception as e:
        logger.error(f"ERROR: Failed to get file metadata for {file_path}: {e}", exc_info=True)
        return None, f"Failed to get file metadata: {e}"


def get_files_metadata_batch(file_paths: List[str]) -> Dict[str, dict]:
    """
    Batch fetch file metadata for multiple paths.
    Firestore 'in' query supports up to 10 items per query, so we batch them.
    Batches are executed concurrently via ThreadPoolExecutor.

    Args:
        file_paths: List of file paths to fetch

    Returns:
        Dictionary mapping file_path -> metadata dict
    """
    metadata_map = {}
    if not file_paths:
        return metadata_map

    def _fetch_batch(batch_paths):
        result = {}
        query = files_ref.where('file_path', 'in', batch_paths)
        docs = list(query.stream())
        for doc in docs:
            file_data = doc.to_dict()
            file_path = file_data.get('file_path')
            if file_path:
                if 'created_at' in file_data and isinstance(file_data['created_at'], datetime.datetime):
                    file_data['created_at'] = file_data['created_at'].isoformat()
                if 'updated_at' in file_data and isinstance(file_data['updated_at'], datetime.datetime):
                    file_data['updated_at'] = file_data['updated_at'].isoformat()
                file_data['created_date'] = file_data.get('created_at')
                file_data['updated_date'] = file_data.get('updated_at')
                if 'file_size_bytes' in file_data:
                    file_data['size'] = file_data['file_size_bytes']
                result[file_path] = file_data
        return result

    try:
        batch_size = 10
        batches = [file_paths[i:i + batch_size] for i in range(0, len(file_paths), batch_size)]
        with ThreadPoolExecutor(max_workers=min(len(batches), 5)) as executor:
            for partial_map in executor.map(_fetch_batch, batches):
                metadata_map.update(partial_map)
    except Exception as e:
        logger.error(f"ERROR: Failed to batch fetch file metadata: {e}", exc_info=True)

    return metadata_map


def get_folders_metadata_batch(folder_paths: List[str]) -> Dict[str, dict]:
    """
    Batch fetch folder metadata for multiple paths.
    Firestore 'in' query supports up to 10 items per query, so we batch them.
    Batches are executed concurrently via ThreadPoolExecutor.

    Args:
        folder_paths: List of folder paths to fetch

    Returns:
        Dictionary mapping folder_path -> metadata dict
    """
    metadata_map = {}
    if not folder_paths:
        return metadata_map

    def _fetch_batch(batch_paths):
        result = {}
        query = folders_ref.where('folder_path', 'in', batch_paths)
        docs = list(query.stream())
        for doc in docs:
            folder_data = doc.to_dict()
            folder_path = folder_data.get('folder_path')
            if folder_path:
                if 'created_at' in folder_data and isinstance(folder_data['created_at'], datetime.datetime):
                    folder_data['created_at'] = folder_data['created_at'].isoformat()
                if 'updated_at' in folder_data and isinstance(folder_data['updated_at'], datetime.datetime):
                    folder_data['updated_at'] = folder_data['updated_at'].isoformat()
                folder_data['created_date'] = folder_data.get('created_at')
                folder_data['updated_date'] = folder_data.get('updated_at')
                result[folder_path] = folder_data
        return result

    try:
        batch_size = 10
        batches = [folder_paths[i:i + batch_size] for i in range(0, len(folder_paths), batch_size)]
        with ThreadPoolExecutor(max_workers=min(len(batches), 5)) as executor:
            for partial_map in executor.map(_fetch_batch, batches):
                metadata_map.update(partial_map)
    except Exception as e:
        logger.error(f"ERROR: Failed to batch fetch folder metadata: {e}", exc_info=True)

    return metadata_map


def get_files_in_folder(folder_path: str) -> Tuple[list, Optional[str]]:
    """
    Retrieves all files in a specific folder.
    
    Args:
        folder_path: Path of the folder
    
    Returns:
        Tuple of (files_list, error_message)
    """
    try:
        query = files_ref.where('folder_path', '==', folder_path)
        docs = list(query.stream())
        
        files_list = []
        for doc in docs:
            file_data = doc.to_dict()
            file_data['id'] = doc.id
            
            # Convert datetime to ISO format
            if 'created_at' in file_data and isinstance(file_data['created_at'], datetime.datetime):
                file_data['created_at'] = file_data['created_at'].isoformat()
            if 'updated_at' in file_data and isinstance(file_data['updated_at'], datetime.datetime):
                file_data['updated_at'] = file_data['updated_at'].isoformat()
            
            # Add aliases for API consistency: created_date, updated_date, size
            file_data['created_date'] = file_data.get('created_at')
            file_data['updated_date'] = file_data.get('updated_at')
            # Use file_size_bytes if available, otherwise use size if it exists
            if 'file_size_bytes' in file_data:
                file_data['size'] = file_data['file_size_bytes']
            elif 'size' not in file_data:
                file_data['size'] = None
            
            files_list.append(file_data)
        
        return files_list, None
        
    except Exception as e:
        logger.error(f"ERROR: Failed to get files in folder {folder_path}: {e}", exc_info=True)
        return [], f"Failed to get files in folder: {e}"


def delete_folder_metadata(folder_path: str) -> Tuple[bool, Optional[str]]:
    """
    Deletes folder metadata from Firestore.
    
    Args:
        folder_path: Full GCS path of the folder
    
    Returns:
        Tuple of (success, error_message)
    """
    try:
        # Find folder by path
        query = folders_ref.where('folder_path', '==', folder_path).limit(1)
        docs = list(query.stream())
        
        if not docs:
            return False, f"Folder metadata not found: {folder_path}"
        
        doc_ref = folders_ref.document(docs[0].id)
        doc_ref.delete()
        
        logger.info(f"Folder metadata deleted: {folder_path}")
        return True, None
        
    except Exception as e:
        logger.error(f"ERROR: Failed to delete folder metadata for {folder_path}: {e}", exc_info=True)
        return False, f"Failed to delete folder metadata: {e}"


def delete_file_metadata(file_path: str) -> Tuple[bool, Optional[str]]:
    """
    Deletes file metadata from Firestore.
    
    Args:
        file_path: Full GCS path of the file
    
    Returns:
        Tuple of (success, error_message)
    """
    try:
        # Find file by path
        query = files_ref.where('file_path', '==', file_path).limit(1)
        docs = list(query.stream())
        
        if not docs:
            return False, f"File metadata not found: {file_path}"
        
        doc_ref = files_ref.document(docs[0].id)
        doc_ref.delete()
        
        logger.info(f"File metadata deleted: {file_path}")
        return True, None
        
    except Exception as e:
        logger.error(f"ERROR: Failed to delete file metadata for {file_path}: {e}", exc_info=True)
        return False, f"Failed to delete file metadata: {e}"


def delete_all_files_metadata_in_folder(folder_path: str) -> Tuple[int, Optional[str]]:
    """
    Deletes metadata for all files in a specific folder.
    
    Args:
        folder_path: Path of the folder
    
    Returns:
        Tuple of (deleted_count, error_message)
    """
    try:
        query = files_ref.where('folder_path', '==', folder_path)
        docs = list(query.stream())
        
        deleted_count = 0
        for doc in docs:
            try:
                doc.reference.delete()
                deleted_count += 1
            except Exception as e:
                logger.warning(f"Failed to delete file metadata {doc.id}: {e}")
        
        logger.info(f"Deleted {deleted_count} file metadata records for folder: {folder_path}")
        return deleted_count, None
        
    except Exception as e:
        logger.error(f"ERROR: Failed to delete files metadata in folder {folder_path}: {e}", exc_info=True)
        return 0, f"Failed to delete files metadata: {e}"


def search_files_by_name(
    search_term: str,
    path_filter: Optional[str] = None,
    page: int = 1,
    page_size: int = 50
) -> Tuple[list, dict, Optional[str]]:
    """
    Search files by name using Firestore queries with case-insensitive substring matching.
    
    Args:
        search_term: Search term (substring match, case-insensitive)
        path_filter: Optional path prefix to filter results within (e.g., "Georgia 14/Pending Files/")
        page: Page number (1-based)
        page_size: Items per page
    
    Returns:
        Tuple of (files_list, pagination_info, error_message)
    """
    try:
        # Normalize search term for case-insensitive search
        search_lower = search_term.lower()
        
        # Query all files (since some may not have file_name_lower field yet)
        # We'll filter in Python for substring matching
        # For better performance with large datasets, we can optimize this later
        # by ensuring all documents have the lowercase field
        query = files_ref
        
        # Add path filter if provided
        if path_filter:
            # Normalize path filter
            if not path_filter.endswith('/'):
                path_filter += '/'
        
        # Get all matching documents
        docs = list(query.stream())
        
        # Apply substring filtering and path filtering
        matching_files = []
        for doc in docs:
            file_data = doc.to_dict()
            file_name = file_data.get('file_name', '')
            # Use lowercase field if available, otherwise compute it
            file_name_lower = file_data.get('file_name_lower', file_name.lower())
            folder_path = file_data.get('folder_path', '')
            
            # Check substring match (case-insensitive)
            if search_lower not in file_name_lower:
                continue
            
            # Apply path filter if provided
            if path_filter:
                if not folder_path.startswith(path_filter):
                    continue
            
            # Format the file data
            file_data['id'] = doc.id
            
            # Convert datetime to ISO format
            if 'created_at' in file_data and isinstance(file_data['created_at'], datetime.datetime):
                file_data['created_at'] = file_data['created_at'].isoformat()
            if 'updated_at' in file_data and isinstance(file_data['updated_at'], datetime.datetime):
                file_data['updated_at'] = file_data['updated_at'].isoformat()
            
            # Add aliases for API consistency
            file_data['created_date'] = file_data.get('created_at')
            file_data['updated_date'] = file_data.get('updated_at')
            if 'file_size_bytes' in file_data:
                file_data['size'] = file_data['file_size_bytes']
            elif 'size' not in file_data:
                file_data['size'] = None
            
            matching_files.append(file_data)
        
        # Sort by name
        matching_files.sort(key=lambda x: x.get('file_name', '').lower())
        
        # Pagination
        total = len(matching_files)
        start_index = (page - 1) * page_size
        end_index = start_index + page_size
        paginated_files = matching_files[start_index:end_index]
        
        total_pages = (total + page_size - 1) // page_size if total > 0 else 1
        
        pagination_info = {
            "current_page": page,
            "total_pages": total_pages,
            "page_size": page_size,
            "total_items": total,
            "has_next": page < total_pages,
            "has_prev": page > 1
        }
        
        return paginated_files, pagination_info, None
        
    except Exception as e:
        logger.error(f"ERROR: Failed to search files by name: {e}", exc_info=True)
        return [], {}, f"Failed to search files: {e}"


def search_folders_by_name(
    search_term: str,
    path_filter: Optional[str] = None,
    page: int = 1,
    page_size: int = 50
) -> Tuple[list, dict, Optional[str]]:
    """
    Search folders by name using Firestore queries with case-insensitive substring matching.
    
    Args:
        search_term: Search term (substring match, case-insensitive)
        path_filter: Optional path prefix to filter results within
        page: Page number (1-based)
        page_size: Items per page
    
    Returns:
        Tuple of (folders_list, pagination_info, error_message)
    """
    try:
        # Normalize search term for case-insensitive search
        search_lower = search_term.lower()
        
        # Query all folders (since some may not have folder_name_lower field yet)
        # We'll filter in Python for substring matching
        # For better performance with large datasets, we can optimize this later
        # by ensuring all documents have the lowercase field
        query = folders_ref
        
        # Get all matching documents
        docs = list(query.stream())
        
        # Apply substring filtering and path filtering
        matching_folders = []
        for doc in docs:
            folder_data = doc.to_dict()
            folder_name = folder_data.get('folder_name', '')
            # Use lowercase field if available, otherwise compute it
            folder_name_lower = folder_data.get('folder_name_lower', folder_name.lower())
            folder_path = folder_data.get('folder_path', '')
            
            # Check substring match (case-insensitive)
            if search_lower not in folder_name_lower:
                continue
            
            # Apply path filter if provided
            if path_filter:
                if not path_filter.endswith('/'):
                    path_filter += '/'
                if not folder_path.startswith(path_filter):
                    continue
            
            # Format the folder data
            folder_data['id'] = doc.id
            
            # Convert datetime to ISO format
            if 'created_at' in folder_data and isinstance(folder_data['created_at'], datetime.datetime):
                folder_data['created_at'] = folder_data['created_at'].isoformat()
            if 'updated_at' in folder_data and isinstance(folder_data['updated_at'], datetime.datetime):
                folder_data['updated_at'] = folder_data['updated_at'].isoformat()
            
            # Add aliases for API consistency
            folder_data['created_date'] = folder_data.get('created_at')
            folder_data['updated_date'] = folder_data.get('updated_at')
            
            matching_folders.append(folder_data)
        
        # Sort by name
        matching_folders.sort(key=lambda x: x.get('folder_name', '').lower())
        
        # Pagination
        total = len(matching_folders)
        start_index = (page - 1) * page_size
        end_index = start_index + page_size
        paginated_folders = matching_folders[start_index:end_index]
        
        total_pages = (total + page_size - 1) // page_size if total > 0 else 1
        
        pagination_info = {
            "current_page": page,
            "total_pages": total_pages,
            "page_size": page_size,
            "total_items": total,
            "has_next": page < total_pages,
            "has_prev": page > 1
        }
        
        return paginated_folders, pagination_info, None
        
    except Exception as e:
        logger.error(f"ERROR: Failed to search folders by name: {e}", exc_info=True)
        return [], {}, f"Failed to search folders: {e}"
