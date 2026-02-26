# backend/app/routes/activity_log_routes.py
from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity, get_jwt
from datetime import datetime, timezone
import logging
from app.models.activity_log_model import get_activity_logs, ActivityTypes

logger = logging.getLogger(__name__)

activity_log_bp = Blueprint('activity_log_bp', __name__)


@activity_log_bp.route('/history', methods=['GET'])
@jwt_required()
def get_user_activity_history():
    """
    Retrieves paginated activity logs for the current user or all users (if admin).
    
    Query Parameters:
    - limit: Number of records to return (1-200, default: 50)
    - start_after: Document ID for pagination cursor
    - user_email: Filter by specific user email (admin only)
    - activity_type: Filter by activity type (e.g., 'FILE_UPLOAD', 'AUTH_LOGIN')
    - start_date: Filter logs after this date (ISO format: YYYY-MM-DDTHH:MM:SS)
    - end_date: Filter logs before this date (ISO format: YYYY-MM-DDTHH:MM:SS)
    - success: Filter by success status (true/false)
    
    Returns:
    - JSON response with activity logs, pagination info, and filters applied
    """
    try:
        current_user_email = get_jwt_identity()
        claims = get_jwt()
        user_role = claims.get("role")
        
        # Get query parameters
        limit = request.args.get('limit', 50, type=int)
        page = request.args.get('page', None, type=int)
        start_after = request.args.get('start_after', None, type=str)
        activity_type = request.args.get('activity_type', None, type=str)
        start_date_str = request.args.get('start_date', None, type=str)
        end_date_str = request.args.get('end_date', None, type=str)
        success_param = request.args.get('success', None, type=str)
        user_email_filter = request.args.get('user_email', None, type=str)
        
        # Validate pagination parameters
        # Cannot use both page and start_after
        if page is not None and start_after is not None:
            return jsonify({
                "message": "Cannot use both 'page' and 'start_after' parameters. Use one pagination method."
            }), 400
        
        # Validate and clamp limit
        limit = max(1, min(limit, 200))
        
        # Validate page number
        if page is not None and page < 1:
            return jsonify({
                "message": "Page number must be >= 1"
            }), 400
        
        # Parse date strings if provided
        start_date = None
        end_date = None
        
        if start_date_str:
            try:
                start_date = datetime.fromisoformat(start_date_str.replace('Z', '+00:00'))
                if start_date.tzinfo is None:
                    start_date = start_date.replace(tzinfo=timezone.utc)
            except ValueError as e:
                return jsonify({
                    "message": "Invalid start_date format. Use ISO format: YYYY-MM-DDTHH:MM:SS",
                    "error": str(e)
                }), 400
        
        if end_date_str:
            try:
                end_date = datetime.fromisoformat(end_date_str.replace('Z', '+00:00'))
                if end_date.tzinfo is None:
                    end_date = end_date.replace(tzinfo=timezone.utc)
            except ValueError as e:
                return jsonify({
                    "message": "Invalid end_date format. Use ISO format: YYYY-MM-DDTHH:MM:SS",
                    "error": str(e)
                }), 400
        
        # Parse success filter
        success = None
        if success_param is not None:
            if success_param.lower() == 'true':
                success = True
            elif success_param.lower() == 'false':
                success = False
            else:
                return jsonify({
                    "message": "Invalid success parameter. Use 'true' or 'false'"
                }), 400
        
        # Authorization: Only admins can view other users' activity logs
        if user_email_filter:
            if user_role not in ["admin", "super-user"]:
                return jsonify({
                    "message": "Only admins can filter by other users' email addresses"
                }), 403
        else:
            # Non-admin users can only see their own logs
            if user_role not in ["admin", "super-user"]:
                user_email_filter = current_user_email
        
        # Retrieve activity logs
        logs, next_cursor, total_items, error = get_activity_logs(
            limit=limit,
            page=page,
            start_after_log_id=start_after,
            user_email=user_email_filter,
            activity_type=activity_type,
            start_date=start_date,
            end_date=end_date,
            success=success
        )
        
        if error:
            logger.error(f"Error retrieving activity logs: {error}")
            return jsonify({
                "message": "Error retrieving activity logs",
                "error": error
            }), 500
        
        # Calculate current count
        current_count = len(logs)
        has_more = next_cursor is not None
        
        # Build pagination object
        pagination = {
            "limit": limit,
            "current_count": current_count,
            "pagination_type": "page" if page is not None else "cursor"
        }
        
        # Add page-based pagination info
        if page is not None:
            pagination["page"] = page
            pagination["has_previous"] = page > 1
            pagination["has_next"] = has_more
            
            # Calculate total pages if total_items is available
            if total_items is not None:
                pagination["total_items"] = total_items
                total_pages = (total_items + limit - 1) // limit if total_items > 0 else 0
                pagination["total_pages"] = total_pages
                pagination["has_next"] = page < total_pages
        else:
            # Cursor-based pagination info
            pagination["has_previous"] = start_after is not None
            pagination["has_next"] = has_more
            pagination["next_cursor"] = next_cursor
            pagination["previous_cursor"] = start_after  # The cursor used for this page can be used as previous
            
            # Add total items if available (might be available from other sources)
            if total_items is not None:
                pagination["total_items"] = total_items
                pagination["estimated_total_pages"] = (total_items + limit - 1) // limit if total_items > 0 else 0
        
        # Build response
        response_data = {
            "logs": logs,
            "pagination": pagination
        }
        
        # Include applied filters in response
        filters_applied = {}
        if user_email_filter:
            filters_applied["user_email"] = user_email_filter
        if activity_type:
            filters_applied["activity_type"] = activity_type
        if start_date:
            filters_applied["start_date"] = start_date.isoformat()
        if end_date:
            filters_applied["end_date"] = end_date.isoformat()
        if success is not None:
            filters_applied["success"] = success
        
        if filters_applied:
            response_data["filters_applied"] = filters_applied
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"Unexpected error in get_user_activity_history: {e}", exc_info=True)
        return jsonify({
            "message": "An unexpected error occurred while retrieving activity logs",
            "error": str(e)
        }), 500


@activity_log_bp.route('/activity-types', methods=['GET'])
@jwt_required()
def get_activity_types():
    """
    Returns a list of all available activity types.
    Useful for frontend dropdowns and filtering.
    """
    try:
        # Get all activity type constants from ActivityTypes class
        activity_types = {
            "authentication": [
                ActivityTypes.AUTH_LOGIN,
                ActivityTypes.AUTH_LOGIN_FAILED,
                ActivityTypes.AUTH_LOGOUT,
                ActivityTypes.AUTH_SSO_LOGIN,
                ActivityTypes.AUTH_TOKEN_REFRESH
            ],
            "file_operations": [
                ActivityTypes.FILE_UPLOAD,
                ActivityTypes.FILE_DOWNLOAD,
                ActivityTypes.FILE_PREVIEW,
                ActivityTypes.FILE_TRANSFER,
                ActivityTypes.FILE_DELETE
            ],
            "folder_operations": [
                ActivityTypes.FOLDER_CREATE,
                ActivityTypes.FOLDER_TRANSFER,
                ActivityTypes.FOLDER_DELETE
            ],
            "search_and_chat": [
                ActivityTypes.SEARCH_QUERY,
                ActivityTypes.CHAT_SESSION_START,
                ActivityTypes.CHAT_SESSION_DELETE,
                ActivityTypes.CHAT_SESSION_RENAME,
                ActivityTypes.CHAT_MESSAGE
            ],
            "data_access": [
                ActivityTypes.HISTORY_VIEW,
                ActivityTypes.DOCUMENT_DETAILS,
                ActivityTypes.DOCUMENT_CHUNKS,
                ActivityTypes.DOCUMENT_LOGS,
                ActivityTypes.DOCUMENT_REPROCESS
            ],
            "admin_activities": [
                ActivityTypes.USER_MANAGEMENT,
                ActivityTypes.USER_CREATE,
                ActivityTypes.USER_UPDATE,
                ActivityTypes.USER_DELETE,
                ActivityTypes.BATCH_PROCESS_START,
                ActivityTypes.SYSTEM_DIAGNOSIS,
                ActivityTypes.PROCESSOR_RULES
            ],
            "navigation": [
                ActivityTypes.PAGE_ACCESS,
                ActivityTypes.FEATURE_ACCESS
            ]
        }
        
        return jsonify({
            "activity_types": activity_types,
            "all_types": [
                ActivityTypes.AUTH_LOGIN,
                ActivityTypes.AUTH_LOGIN_FAILED,
                ActivityTypes.AUTH_LOGOUT,
                ActivityTypes.AUTH_SSO_LOGIN,
                ActivityTypes.AUTH_TOKEN_REFRESH,
                ActivityTypes.FILE_UPLOAD,
                ActivityTypes.FILE_DOWNLOAD,
                ActivityTypes.FILE_PREVIEW,
                ActivityTypes.FILE_TRANSFER,
                ActivityTypes.FILE_DELETE,
                ActivityTypes.FOLDER_CREATE,
                ActivityTypes.FOLDER_TRANSFER,
                ActivityTypes.FOLDER_DELETE,
                ActivityTypes.SEARCH_QUERY,
                ActivityTypes.CHAT_SESSION_START,
                ActivityTypes.CHAT_SESSION_DELETE,
                ActivityTypes.CHAT_SESSION_RENAME,
                ActivityTypes.CHAT_MESSAGE,
                ActivityTypes.HISTORY_VIEW,
                ActivityTypes.DOCUMENT_DETAILS,
                ActivityTypes.DOCUMENT_CHUNKS,
                ActivityTypes.DOCUMENT_LOGS,
                ActivityTypes.DOCUMENT_REPROCESS,
                ActivityTypes.USER_MANAGEMENT,
                ActivityTypes.USER_CREATE,
                ActivityTypes.USER_UPDATE,
                ActivityTypes.USER_DELETE,
                ActivityTypes.BATCH_PROCESS_START,
                ActivityTypes.SYSTEM_DIAGNOSIS,
                ActivityTypes.PROCESSOR_RULES,
                ActivityTypes.PAGE_ACCESS,
                ActivityTypes.FEATURE_ACCESS
            ]
        }), 200
        
    except Exception as e:
        logger.error(f"Error retrieving activity types: {e}", exc_info=True)
        return jsonify({
            "message": "Error retrieving activity types",
            "error": str(e)
        }), 500

