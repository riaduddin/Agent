import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Environment variable to control Token logging (default: disabled)
ENABLE_TOKEN_LOGGING = os.getenv("ENABLE_TOKEN_LOGGING", "false").lower() == "true"

# Create logs directory if it doesn't exist (only if logging is enabled)
if ENABLE_TOKEN_LOGGING:
    TOKEN_LOGS_DIR = Path("logs/token_usage")
    TOKEN_LOGS_DIR.mkdir(parents=True, exist_ok=True)
else:
    TOKEN_LOGS_DIR = None

def log_token_usage(p_id: str, author: str, input_tokens: int = 0, output_tokens: int = 0, thoughts_tokens: int = 0, message: str = None):
    """
    Log token usage per agent/step to a text file for auditing.
    
    This function is controlled by the ENABLE_TOKEN_LOGGING environment variable.
    
    Args:
        p_id: Presentation ID
        author: The agent or step name
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        thoughts_tokens: Number of thought tokens (optional)
        message: Optional message about the tracking state (e.g. initialization)
    """
    if not ENABLE_TOKEN_LOGGING:
        return
    
    try:
        # Create a log file per presentation
        log_file = TOKEN_LOGS_DIR / f"{p_id}_tokens.log"
        
        timestamp = datetime.utcnow().isoformat()
        
        # Format the entry
        entry = {
            "timestamp": timestamp,
            "p_id": p_id,
            "author": author,
            "usage": {
                "input": input_tokens,
                "output": output_tokens,
                "thoughts": thoughts_tokens,
                "step_total": input_tokens + output_tokens
            }
        }
        
        if message:
            entry["message"] = message
        
        # Append as JSON line
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry) + "\n")
            
    except Exception as e:
        logger.error(f"❌ Failed to log token usage: {e}")
