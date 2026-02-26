# backend/app/services/activity_log_service.py
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from app import db
from flask import Request
from app.utils.debug_logger import debug_log, debug_warn, debug_error

logger = logging.getLogger(__name__)

# Activity log collection reference
activity_logs_ref = db.collection('user_activity_logs')

def log_user_activity(
    user_email: str,
    activity_type: str,
    activity_description: str,
    metadata: Optional[Dict[str, Any]] = None,
    request_obj: Optional[Request] = None,
    success: bool = True,
    error_message: Optional[str] = None
) -> None:
    """
    Silently logs user activity to Firestore.
    Never raises exceptions to protect main functionality.
    
    Args:
        user_email: Email of the user performing the activity
        activity_type: Type of activity (e.g., 'AUTH_LOGIN', 'FILE_UPLOAD')
        activity_description: Human-readable description of the activity
        metadata: Optional dictionary with additional activity data
        request_obj: Flask request object to extract IP and User-Agent
        success: Whether the activity was successful
        error_message: Error message if success=False
    """
    try:
        # Extract request information if available
        ip_address = None
        user_agent = None
        if request_obj:
            ip_address = request_obj.remote_addr
            user_agent = request_obj.headers.get('User-Agent', '')[:500]  # Truncate long user agents
        
        # Prepare activity log data
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
        
        # Add to Firestore collection
        debug_log(f"Attempting to write activity log to Firestore: {activity_type} for {user_email}")
        doc_ref = activity_logs_ref.add(activity_data)
        debug_log(f"Successfully wrote activity log to Firestore with ID: {doc_ref[1].id}")
        
        # Log successful activity logging (for debugging)
        logger.debug(f"Activity logged: {user_email} - {activity_type}")
        logger.info(f"Activity log created with ID: {doc_ref[1].id} for {user_email} - {activity_type}")
        
    except Exception as e:
        # CRITICAL: Never raise exceptions - log the error and continue silently
        debug_error(f"Activity logging failed for {user_email} - {activity_type}: {e}")
        logger.error(f"Activity logging failed for {user_email} - {activity_type}: {e}", exc_info=True)
        # Main application functionality continues unaffected


def log_authentication_activity(user_email: str, activity_type: str, success: bool, 
                               additional_info: Optional[Dict[str, Any]] = None, 
                               request_obj: Optional[Request] = None) -> None:
    """Helper function for authentication-related activities."""
    description = f"Authentication: {activity_type.replace('AUTH_', '').lower()}"
    if not success:
        description += " (failed)"
    
    metadata = {
        'auth_type': activity_type,
        'success': success,
        **(additional_info or {})
    }
    
    log_user_activity(
        user_email=user_email,
        activity_type=activity_type,
        activity_description=description,
        metadata=metadata,
        request_obj=request_obj,
        success=success
    )


def log_file_activity(user_email: str, activity_type: str, filename: str, 
                     file_info: Optional[Dict[str, Any]] = None,
                     request_obj: Optional[Request] = None) -> None:
    """Helper function for file-related activities."""
    description = f"File operation: {filename}"
    
    metadata = {
        'filename': filename,
        'file_operation': activity_type,
        **(file_info or {})
    }
    
    log_user_activity(
        user_email=user_email,
        activity_type=activity_type,
        activity_description=description,
        metadata=metadata,
        request_obj=request_obj
    )


def log_search_activity(user_email: str, query: str, session_id: Optional[str] = None,
                       results_count: Optional[int] = None,
                       request_obj: Optional[Request] = None) -> None:
    """Helper function for search/chat activities."""
    # Truncate long queries for description
    query_preview = query[:100] + "..." if len(query) > 100 else query
    description = f"Search query: {query_preview}"
    
    metadata = {
        'query': query,
        'query_length': len(query),
        'session_id': session_id,
        'results_count': results_count
    }
    
    log_user_activity(
        user_email=user_email,
        activity_type='SEARCH_QUERY',
        activity_description=description,
        metadata=metadata,
        request_obj=request_obj
    )


def log_navigation_activity(user_email: str, page_name: str, page_path: str,
                          request_obj: Optional[Request] = None) -> None:
    """Helper function for page navigation activities."""
    description = f"Accessed: {page_name}"
    
    metadata = {
        'page_name': page_name,
        'page_path': page_path
    }
    
    log_user_activity(
        user_email=user_email,
        activity_type='PAGE_ACCESS',
        activity_description=description,
        metadata=metadata,
        request_obj=request_obj
    )


def log_admin_activity(user_email: str, admin_action: str, target_info: Optional[Dict[str, Any]] = None,
                      request_obj: Optional[Request] = None) -> None:
    """Helper function for admin-related activities."""
    description = f"Admin action: {admin_action}"
    
    metadata = {
        'admin_action': admin_action,
        'user_role': 'admin',
        **(target_info or {})
    }
    
    log_user_activity(
        user_email=user_email,
        activity_type='ADMIN_ACTION',
        activity_description=description,
        metadata=metadata,
        request_obj=request_obj
    )
