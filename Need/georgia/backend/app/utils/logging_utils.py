import logging
import sys
import json
import datetime

class JSONFormatter(logging.Formatter):
    """
    Formatter that outputs JSON strings compatible with Google Cloud Logging.
    """
    def format(self, record):
        log_record = {
            "severity": record.levelname,
            "message": record.getMessage(),
            "timestamp": datetime.datetime.fromtimestamp(record.created, tz=datetime.timezone.utc).isoformat(),
            "component": "backend", # Default component
            "logger": record.name
        }
        
        # Add worker_id or other fields if present in extra args
        if hasattr(record, "worker_id"):
            log_record["worker_id"] = record.worker_id
            log_record["component"] = "worker"
            
        # Add exception info if present
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
            # Google Cloud Error Reporting looks for this by default
            log_record["stack_trace"] = log_record["exception"] 

        return json.dumps(log_record)

def setup_cloud_logging(root_level=logging.INFO):
    """
    Configures the root logger to output structured JSON to stdout.
    This ensures Cloud Run parses severity correct (INFO vs ERROR).
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(root_level)

    # Remove existing handlers (e.g. from basicConfig or default Flask setup)
    if root_logger.handlers:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

    # Create stdout handler
    # Cloud Run treats:
    # stdout -> INFO/DEFAULT (parsed by JSON 'severity')
    # stderr -> ERROR (regardless of content)
    # So we write EVERYTHING to stdout strings, using "severity" field to distinguish.
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    root_logger.addHandler(handler)
    
    return root_logger
