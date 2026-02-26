import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Environment variable to control Socket.IO logging (default: disabled)
ENABLE_SOCKETIO_LOGGING = os.getenv("ENABLE_SOCKETIO_LOGGING", "false").lower() == "true"

# Create logs directory if it doesn't exist (only if logging is enabled)
if ENABLE_SOCKETIO_LOGGING:
    SOCKETIO_LOGS_DIR = Path("logs/socketio_messages")
    SOCKETIO_LOGS_DIR.mkdir(parents=True, exist_ok=True)
else:
    SOCKETIO_LOGS_DIR = None

def log_socketio_message(message: Dict[str, Any], p_id: str, direction: str = "outgoing"):
    """
    Log Socket.IO messages to a text file for debugging.
    
    This function is controlled by the ENABLE_SOCKETIO_LOGGING environment variable.
    Set ENABLE_SOCKETIO_LOGGING=true in your .env file to enable logging.
    
    Args:
        message: The message dictionary being sent/received
        p_id: Presentation ID
        direction: "outgoing" or "incoming"
    """
    # Skip logging if not enabled
    if not ENABLE_SOCKETIO_LOGGING:
        return
    
    try:
        # Create a log file per presentation
        log_file = SOCKETIO_LOGS_DIR / f"{p_id}_socketio.log"
        
        timestamp = datetime.utcnow().isoformat()
        separator = "=" * 80
        
        # Format the message for readability
        log_entry = f"""
        {separator}
        TIMESTAMP: {timestamp}
        DIRECTION: {direction.upper()}
        P_ID: {p_id}
        MESSAGE TYPE: {message.get('type', 'unknown')}
        AUTHOR: {message.get('author', 'unknown')}
        EVENT: {message.get('event', 'N/A')}
        {separator}

        MESSAGE CONTENT:
        {json.dumps(message, indent=2, default=str)}

        {separator}

        """
        
        # Append to log file
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry)
            
        logger.debug(f"📝 Logged {direction} Socket.IO message to {log_file}")
        
    except Exception as e:
        logger.error(f"❌ Failed to log Socket.IO message: {e}")
