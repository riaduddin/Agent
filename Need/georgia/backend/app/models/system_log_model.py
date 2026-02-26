from google.cloud import firestore
# from flask import current_app # Removed
from app import db # Import db directly
from app.utils.debug_logger import debug_log, debug_error
from app.utils.redis_client import get_redis_client
import json

class SystemLogModel:
    def __init__(self, session_id: str, message_id: str, user_query_text: str, step_name: str, status: str, step_details: dict = None, error_message: str = None):
        self.session_id = session_id
        self.message_id = message_id
        self.user_query_text = user_query_text
        self.timestamp = firestore.SERVER_TIMESTAMP  # Let Firestore set the timestamp
        self.step_name = step_name
        self.status = status
        self.step_details = step_details if step_details is not None else {}
        self.error_message = error_message

    def to_dict(self):
        data = {
            "session_id": self.session_id,
            "message_id": self.message_id,
            "user_query_text": self.user_query_text,
            "timestamp": self.timestamp,
            "step_name": self.step_name,
            "status": self.status,
            "step_details": self.step_details
        }
        if self.error_message:
            data["error_message"] = self.error_message
        return data

    @staticmethod
    def get_db():
        # return current_app.config['FIRESTORE_CLIENT'] # Old way
        return db # New way: use the imported db instance

    @staticmethod
    def _get_redis_buffer_key(message_id: str) -> str:
        """Get the Redis key for buffering logs for a specific message."""
        return f"system_logs_buffer:{message_id}"

    @staticmethod
    def add_log_entry(log_entry_data: dict):
        """
        Adds a new system log entry.
        
        If session_id is None (first message before session creation),
        the log is buffered in Redis. Otherwise, it's written directly to Firestore.
        """
        try:
            session_id = log_entry_data.get('session_id')
            message_id = log_entry_data.get('message_id')
            
            if session_id is None or session_id == '':
                # Buffer to Redis for first message (before session is created)
                redis_client = get_redis_client()
                buffer_key = SystemLogModel._get_redis_buffer_key(message_id)
                
                # Store log entry as JSON in a Redis list
                # Use a serializable timestamp placeholder
                log_for_redis = log_entry_data.copy()
                log_for_redis['timestamp'] = '__SERVER_TIMESTAMP__'  # Placeholder
                log_for_redis['_buffer_order'] = redis_client.llen(buffer_key)  # Track order
                
                redis_client.rpush(buffer_key, json.dumps(log_for_redis))
                # Set expiry (1 hour) in case flush never happens
                redis_client.expire(buffer_key, 3600)
                
                debug_log(f"INFO: System log buffered in Redis for message_id: {message_id}, step: {log_entry_data.get('step_name')}")
            else:
                # Write directly to Firestore (existing session)
                firestore_client = SystemLogModel.get_db()
                logs_ref = firestore_client.collection("text_message_system_logs")
                logs_ref.add(log_entry_data)
                debug_log(f"INFO: System log entry added for message_id: {message_id}, step: {log_entry_data.get('step_name')}")
                
        except Exception as e:
            debug_error(f"Error adding system log entry: {e} - Data: {log_entry_data}")

    @staticmethod
    def flush_buffered_logs(message_id: str, session_id: str):
        """
        Flush all buffered logs from Redis to Firestore with the correct session_id.
        
        This should be called after a new session is created (first message).
        """
        try:
            redis_client = get_redis_client()
            buffer_key = SystemLogModel._get_redis_buffer_key(message_id)
            
            # Get all buffered logs
            buffered_logs = redis_client.lrange(buffer_key, 0, -1)
            
            if not buffered_logs:
                debug_log(f"INFO: No buffered logs to flush for message_id: {message_id}")
                return
            
            debug_log(f"INFO: Flushing {len(buffered_logs)} buffered logs for message_id: {message_id} to session_id: {session_id}")
            
            firestore_client = SystemLogModel.get_db()
            logs_ref = firestore_client.collection("text_message_system_logs")
            
            # Write each buffered log to Firestore with correct session_id
            for log_json in buffered_logs:
                log_data = json.loads(log_json)
                log_data['session_id'] = session_id  # Set the correct session_id
                log_data['timestamp'] = firestore.SERVER_TIMESTAMP  # Replace placeholder
                del log_data['_buffer_order']  # Remove internal tracking field
                
                logs_ref.add(log_data)
            
            # Delete the Redis buffer
            redis_client.delete(buffer_key)
            debug_log(f"INFO: Successfully flushed {len(buffered_logs)} logs to Firestore and cleared Redis buffer")
            
        except Exception as e:
            debug_error(f"Error flushing buffered logs for message_id {message_id}: {e}")

    @staticmethod
    def get_logs_for_message(session_id: str, message_id: str):
        """
        Retrieves all system log entries for a specific message_id within a session_id,
        ordered by timestamp.
        """
        try:
            firestore_client = SystemLogModel.get_db()
            logs_ref = firestore_client.collection("text_message_system_logs")
            query = logs_ref.where("session_id", "==", session_id) \
                            .where("message_id", "==", message_id) \
                            .order_by("timestamp", direction=firestore.Query.ASCENDING)
            
            results = query.stream()
            logs = []
            for doc in results:
                log_data = doc.to_dict()
                log_data['log_id'] = doc.id
                if 'timestamp' in log_data and hasattr(log_data['timestamp'], 'isoformat'):
                     log_data['timestamp'] = log_data['timestamp'].isoformat()
                logs.append(log_data)
            return logs
        except Exception as e:
            debug_error(f"Error retrieving system logs for message_id {message_id}: {e}")
            return []

# Example usage (for testing or within route handlers):
# log_data = SystemLogModel(
#     session_id="some_session_id",
#     message_id="some_message_id",
#     user_query_text="What is the capital of Georgia?",
#     step_name="QUERY_RECEIVED",
#     status="INFO",
#     step_details={"source_ip": "127.0.0.1"}
# ).to_dict()
# SystemLogModel.add_log_entry(log_data)
