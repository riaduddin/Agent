# backend/app/routes/log_routes.py
import io
import csv
from flask import Blueprint, jsonify, request, current_app, Response # Import Response, io, csv
from flask_jwt_extended import jwt_required, get_jwt_identity
from app.models import log_model, SystemLogModel # Import the log model and SystemLogModel
from app import db # Import the initialized Firestore client

log_bp = Blueprint('log_bp', __name__) # Define blueprint once


# Reference to the 'chat_sessions' collection
sessions_ref = db.collection('chat_sessions')


@log_bp.route('', methods=['GET']) # Route is at the blueprint prefix
@jwt_required(optional=True) # Allow OPTIONS preflight, check GET for token
def get_paginated_logs(): # Function definition immediately follows decorators
    """Retrieves paginated and optionally filtered processing logs."""
    try:
        # Pagination parameters
        limit = request.args.get('limit', default=50, type=int)
        start_after = request.args.get('start_after', default=None, type=str)

        # Filter parameters
        level = request.args.get('level', default=None, type=str)
        step = request.args.get('step', default=None, type=str)
        # Read the original_filename parameter sent by the frontend
        original_filename = request.args.get('original_filename', default=None, type=str)
        document_id = request.args.get('document_id', default=None, type=str)
        worker_id = request.args.get('worker_id', default=None, type=str)

        # Ensure limit is within reasonable bounds
        limit = max(1, min(limit, 200)) # Example: Min 1, Max 200

        # Call the updated model function with filters
        logs, next_cursor, error = log_model.get_logs(
            limit=limit,
            start_after_doc_id=start_after,
            level=level,
            step=step,
            original_filename=original_filename, # Pass the filename filter
            document_id=document_id,
            worker_id=worker_id
        )

        if error:
            return jsonify({"msg": "Error retrieving logs", "error": error}), 500

        return jsonify({
            "logs": logs,
            "next_cursor": next_cursor,
            "limit": limit
        }), 200

    except Exception as e:
        # Log the exception for debugging
        current_app.logger.error(f"Error in get_paginated_logs: {e}", exc_info=True)
        return jsonify({"msg": "An unexpected error occurred while fetching logs."}), 500


@log_bp.route('/export', methods=['GET'])
@jwt_required(optional=True) # Allow OPTIONS preflight, check GET for token
def export_logs_as_csv():
    """Exports filtered logs as a CSV file."""
    try:
        # Filter parameters (same as get_paginated_logs)
        level = request.args.get('level', default=None, type=str)
        step = request.args.get('step', default=None, type=str)
        document_id = request.args.get('document_id', default=None, type=str)
        worker_id = request.args.get('worker_id', default=None, type=str)

        # Fetch all logs matching filters using the new model function
        logs, error = log_model.get_all_logs_for_export(
            level=level,
            step=step,
            document_id=document_id,
            worker_id=worker_id
        )

        if error:
            return jsonify({"msg": "Error retrieving logs for export", "error": error}), 500

        if not logs:
            return jsonify({"msg": "No logs found matching the specified filters."}), 404

        # Define CSV headers (adjust based on actual fields in log_data)
        # Use a consistent order
        fieldnames = [
            'timestamp', 'level', 'step', 'worker_id', 'document_id',
            'chunk_id', 'message', 'details', 'id'
        ]

        # Create CSV in memory
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction='ignore') # ignore extra fields in dict

        writer.writeheader()
        for log_entry in logs:
            # Ensure details is a string for CSV compatibility
            if 'details' in log_entry and isinstance(log_entry['details'], dict):
                log_entry['details'] = str(log_entry['details'])
            writer.writerow(log_entry)

        # Prepare response
        csv_data = output.getvalue()
        output.close()

        return Response(
            csv_data,
            mimetype="text/csv",
            headers={"Content-disposition": "attachment; filename=processing_logs.csv"}
        )

    except Exception as e:
        current_app.logger.error(f"Error in export_logs_as_csv: {e}", exc_info=True)
        return jsonify({"msg": "An unexpected error occurred while exporting logs."}), 500


@log_bp.route('/sessions/<session_id>/messages/<message_id>/system_logs', methods=['GET'])
@jwt_required()
def get_text_message_system_logs(session_id: str, message_id: str):
    """
    Retrieves system logs for a specific message within a chat session.
    """
    try:
        # Optional: Add authorization check if needed, e.g., ensure user owns the session
        # current_user_id = get_jwt_identity()
        # if not SessionModel.is_user_session(current_user_id, session_id):
        #     return jsonify({"msg": "Unauthorized to view these logs"}), 403
        

        # Find message details with message_id
        
        # Fetch session document
        session_ref = sessions_ref.document(session_id)
        session_doc = session_ref.get()

        session_data = session_doc.to_dict()

                # Find the message in the messages list
        messages = session_data.get('messages', [])
        message = next((msg for msg in messages if msg.get('id') == message_id), None)

        if not message:
            return jsonify({"msg": "Message not found in session."}), 404
        

        logs = SystemLogModel.get_logs_for_message(session_id, message_id)
        
        if logs is None: # Should return [] on error from model, but defensive check
            return jsonify({"msg": "Error retrieving system logs"}), 500
            
        return jsonify({
            "message": message,
            "logs": logs
        }), 200


    except Exception as e:
        current_app.logger.error(f"Error in get_text_message_system_logs for session {session_id}, message {message_id}: {e}", exc_info=True)
        return jsonify({"msg": "An unexpected error occurred while fetching system logs."}), 500
