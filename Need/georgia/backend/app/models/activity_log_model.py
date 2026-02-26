# backend/app/models/activity_log_model.py
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List
from app import db
from google.cloud import firestore

logger = logging.getLogger(__name__)

# Reference to the 'user_activity_logs' collection in Firestore
activity_logs_ref = db.collection('user_activity_logs')

def create_activity_log(
    user_email: str,
    activity_type: str,
    activity_description: str,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    success: bool = True,
    error_message: Optional[str] = None
) -> Optional[str]:
    """
    Creates a new activity log entry in Firestore.
    Returns the document ID if successful, None if failed.
    Never raises exceptions.
    """
    try:
        activity_data = {
            'user_email': user_email,
            'activity_type': activity_type,
            'activity_description': activity_description,
            'timestamp': datetime.now(timezone.utc),
            'ip_address': ip_address,
            'user_agent': user_agent,
            'metadata': metadata or {},
            'success': success,
            'error_message': error_message
        }
        
        # Add to Firestore and get the document reference
        doc_ref = activity_logs_ref.add(activity_data)
        return doc_ref[1].id  # Return the document ID
        
    except Exception as e:
        # Silent failure - log the error but don't raise
        logger.warning(f"Failed to create activity log for {user_email}: {e}")
        return None


def cleanup_old_activity_logs(days_to_keep: int = 90) -> int:
    """
    Removes activity logs older than specified days.
    Returns the number of logs deleted.
    This function can be called periodically for maintenance.
    """
    try:
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_to_keep)
        
        # Query for old logs
        old_logs_query = activity_logs_ref.where('timestamp', '<', cutoff_date)
        old_logs = old_logs_query.stream()
        
        deleted_count = 0
        batch = db.batch()
        batch_size = 0
        
        for log_doc in old_logs:
            batch.delete(log_doc.reference)
            batch_size += 1
            deleted_count += 1
            
            # Firestore batch limit is 500 operations
            if batch_size >= 500:
                batch.commit()
                batch = db.batch()
                batch_size = 0
        
        # Commit any remaining operations
        if batch_size > 0:
            batch.commit()
        
        logger.info(f"Cleaned up {deleted_count} old activity logs (older than {days_to_keep} days)")
        return deleted_count
        
    except Exception as e:
        logger.error(f"Failed to cleanup old activity logs: {e}")
        return 0


# Activity type constants for consistency
class ActivityTypes:
    """Constants for activity types to ensure consistency across the application."""
    
    # Authentication activities
    AUTH_LOGIN = 'AUTH_LOGIN'
    AUTH_LOGIN_FAILED = 'AUTH_LOGIN_FAILED'
    AUTH_LOGOUT = 'AUTH_LOGOUT'
    AUTH_SSO_LOGIN = 'AUTH_SSO_LOGIN'
    AUTH_TOKEN_REFRESH = 'AUTH_TOKEN_REFRESH'
    
    # File operations
    FILE_UPLOAD = 'FILE_UPLOAD'
    FILE_DOWNLOAD = 'FILE_DOWNLOAD'
    FILE_PREVIEW = 'FILE_PREVIEW'
    FILE_TRANSFER = 'FILE_TRANSFER'
    FILE_DELETE = 'FILE_DELETE'
    
    # Folder operations
    FOLDER_CREATE = 'FOLDER_CREATE'
    FOLDER_TRANSFER = 'FOLDER_TRANSFER'
    FOLDER_DELETE = 'FOLDER_DELETE'
    
    # Search and chat activities
    SEARCH_QUERY = 'SEARCH_QUERY'
    CHAT_SESSION_START = 'CHAT_SESSION_START'
    CHAT_SESSION_DELETE = 'CHAT_SESSION_DELETE'
    CHAT_SESSION_RENAME = 'CHAT_SESSION_RENAME'
    CHAT_MESSAGE = 'CHAT_MESSAGE'
    
    # Data access activities
    HISTORY_VIEW = 'HISTORY_VIEW'
    DOCUMENT_DETAILS = 'DOCUMENT_DETAILS'
    DOCUMENT_CHUNKS = 'DOCUMENT_CHUNKS'
    DOCUMENT_LOGS = 'DOCUMENT_LOGS'
    DOCUMENT_REPROCESS = 'DOCUMENT_REPROCESS'
    
    # Admin activities
    USER_MANAGEMENT = 'USER_MANAGEMENT'
    USER_CREATE = 'USER_CREATE'
    USER_UPDATE = 'USER_UPDATE'
    USER_DELETE = 'USER_DELETE'
    BATCH_PROCESS_START = 'BATCH_PROCESS_START'
    SYSTEM_DIAGNOSIS = 'SYSTEM_DIAGNOSIS'
    PROCESSOR_RULES = 'PROCESSOR_RULES'
    
    # Page access activities
    PAGE_ACCESS = 'PAGE_ACCESS'
    FEATURE_ACCESS = 'FEATURE_ACCESS'


def get_activity_logs_count(
    user_email: Optional[str] = None,
    activity_type: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    success: Optional[bool] = None
) -> tuple[Optional[int], Optional[str]]:
    """
    Gets the total count of activity logs matching the filters.
    
    Args:
        user_email: Filter by user email
        activity_type: Filter by activity type
        start_date: Filter logs after this date
        end_date: Filter logs before this date
        success: Filter by success status
    
    Returns:
        Tuple of (total_count, error)
    """
    try:
        # Build query
        query = activity_logs_ref
        
        # Apply filters
        if user_email:
            query = query.where('user_email', '==', user_email)
        
        if activity_type:
            query = query.where('activity_type', '==', activity_type)
        
        if success is not None:
            query = query.where('success', '==', success)
        
        if start_date:
            query = query.where('timestamp', '>=', start_date)
        
        if end_date:
            query = query.where('timestamp', '<=', end_date)
        
        # Get count (this can be expensive for large collections)
        # Note: Firestore doesn't have a direct count() method, so we need to stream
        # For better performance, we limit the count query to a reasonable number
        # In practice, you might want to cache counts or use a different approach
        try:
            # Stream documents to count (this can be slow for very large collections)
            # Consider implementing caching or approximate counts for production
            count = sum(1 for _ in query.stream())
            return count, None
        except Exception as e:
            logger.warning(f"Error counting documents: {e}")
            return None, str(e)
        
    except Exception as e:
        logger.error(f"Failed to get activity logs count: {e}", exc_info=True)
        return None, f"Failed to get activity logs count: {e}"


def get_activity_logs(
    limit: int = 50,
    start_after_log_id: Optional[str] = None,
    page: Optional[int] = None,
    user_email: Optional[str] = None,
    activity_type: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    success: Optional[bool] = None
) -> tuple[List[Dict[str, Any]], Optional[str], Optional[int], Optional[str]]:
    """
    Retrieves paginated activity logs with optional filtering.
    Supports both cursor-based and page-based pagination.
    
    Args:
        limit: Maximum number of logs to return (1-200)
        start_after_log_id: Document ID for pagination cursor (cursor-based pagination)
        page: Page number (1-based, page-based pagination). Max page: 100
        user_email: Filter by user email
        activity_type: Filter by activity type (e.g., 'FILE_UPLOAD', 'AUTH_LOGIN')
        start_date: Filter logs after this date (datetime with timezone)
        end_date: Filter logs before this date (datetime with timezone)
        success: Filter by success status (True/False)
    
    Returns:
        Tuple of (logs, next_cursor, total_items, error)
        - logs: List of activity log dictionaries
        - next_cursor: Document ID for next page (None if no more pages or using page-based)
        - total_items: Total count of matching logs (None if count unavailable)
        - error: Error message if query failed (None if successful)
    """
    try:
        # Validate and clamp limit
        limit = max(1, min(limit, 200))
        
        # Validate page number if provided
        if page is not None:
            if page < 1:
                return [], None, None, "Page number must be >= 1"
            # Limit max page to 100 to avoid performance issues with large offsets
            if page > 100:
                return [], None, None, "Page number cannot exceed 100. Use cursor-based pagination for deeper pages."
        
        # Build query
        query = activity_logs_ref
        
        # Apply filters
        if user_email:
            query = query.where('user_email', '==', user_email)
        
        if activity_type:
            query = query.where('activity_type', '==', activity_type)
        
        if success is not None:
            query = query.where('success', '==', success)
        
        if start_date:
            query = query.where('timestamp', '>=', start_date)
        
        if end_date:
            query = query.where('timestamp', '<=', end_date)
        
        # Order by timestamp descending (most recent first)
        query = query.order_by('timestamp', direction=firestore.Query.DESCENDING)
        
        # Apply pagination method
        if page is not None:
            # Page-based pagination using offset
            offset = (page - 1) * limit
            query = query.offset(offset)
        elif start_after_log_id:
            # Cursor-based pagination
            try:
                start_after_doc = activity_logs_ref.document(start_after_log_id).get()
                if start_after_doc.exists:
                    query = query.start_after(start_after_doc)
                else:
                    logger.warning(f"Pagination cursor document ID '{start_after_log_id}' not found.")
                    return [], None, None, f"Pagination cursor document ID '{start_after_log_id}' not found."
            except Exception as e:
                logger.error(f"Error with pagination cursor: {e}")
                return [], None, None, f"Error with pagination cursor: {e}"
        
        # Limit results
        query = query.limit(limit)
        
        # Execute query
        docs = list(query.stream())
        
        # Process results
        logs = []
        last_doc_id = None
        
        for doc in docs:
            log_data = doc.to_dict()
            log_data['id'] = doc.id
            
            # Convert timestamp to ISO format if it's a datetime object
            if 'timestamp' in log_data and isinstance(log_data['timestamp'], datetime):
                log_data['timestamp'] = log_data['timestamp'].isoformat()
            
            logs.append(log_data)
            last_doc_id = doc.id
        
        # For page-based pagination, we can determine if there's a next page
        # by checking if we got exactly 'limit' items
        if page is not None:
            # If we got fewer items than limit, we're on the last page
            has_more = len(logs) == limit
            next_cursor = last_doc_id if has_more else None
        else:
            # For cursor-based, use the last doc id as next cursor
            next_cursor = last_doc_id if len(logs) == limit else None
        
        # Get total count (optional, can be expensive for large collections)
        # We'll get it if page-based pagination is used
        total_items = None
        if page is not None:
            total_items, count_error = get_activity_logs_count(
                user_email=user_email,
                activity_type=activity_type,
                start_date=start_date,
                end_date=end_date,
                success=success
            )
            if count_error:
                logger.warning(f"Could not get total count: {count_error}")
        
        return logs, next_cursor, total_items, None
        
    except Exception as e:
        logger.error(f"Failed to retrieve activity logs: {e}", exc_info=True)
        return [], None, None, f"Failed to retrieve activity logs: {e}"


# Helper function to sanitize metadata for logging
def sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitizes metadata to remove sensitive information before logging.
    """
    if not metadata:
        return {}
    
    # Create a copy to avoid modifying original
    sanitized = metadata.copy()
    
    # Remove sensitive fields
    sensitive_fields = ['password', 'token', 'secret', 'key', 'credential']
    for field in sensitive_fields:
        if field in sanitized:
            sanitized[field] = '[REDACTED]'
    
    # Truncate long strings
    for key, value in sanitized.items():
        if isinstance(value, str) and len(value) > 1000:
            sanitized[key] = value[:1000] + '...[TRUNCATED]'
    
    return sanitized
