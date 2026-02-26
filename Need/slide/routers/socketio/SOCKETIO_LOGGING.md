# Socket.IO Message Logging

## Overview
A debugging utility to log all Socket.IO messages to text files for inspection. **This feature is disabled by default** and can be enabled via environment variable.

## Enabling/Disabling
Add to your `.env` file:
```bash
# Enable Socket.IO message logging (default: false)
ENABLE_SOCKETIO_LOGGING=true
```

To disable, either remove the variable or set it to `false`:
```bash
ENABLE_SOCKETIO_LOGGING=false
```

## Location
- **Utility**: `routers/socketio/socketio_logger.py`
- **Log Directory**: `logs/socketio_messages/` (created only when logging is enabled)
- **Log Files**: `{p_id}_socketio.log` (one file per presentation)

## Log Format
Each message is logged with:
- Timestamp (UTC ISO format)
- Direction (OUTGOING/INCOMING)
- Presentation ID
- Message Type
- Author
- Event Name
- Full JSON content (pretty-printed)

## Example Log Entry
```
================================================================================
TIMESTAMP: 2026-01-11T10:19:12.123456
DIRECTION: OUTGOING
P_ID: 696364e498accc507a7479c2
MESSAGE TYPE: chunk
AUTHOR: presentation_spec_extractor_agent
EVENT: N/A
================================================================================

MESSAGE CONTENT:
{
  "type": "chunk",
  "author": "presentation_spec_extractor_agent",
  "p_id": "696364e498accc507a7479c2",
  "text": "Creating presentation...",
  "timestamp": "2026-01-11T10:19:12.123456+00:00",
  "event_id": "abc123"
}

================================================================================
```

## Usage
1. Set `ENABLE_SOCKETIO_LOGGING=true` in your `.env` file
2. Restart the server
3. Run your presentation
4. Check `logs/socketio_messages/` for the log files

The logging is automatic when enabled and captures all messages broadcast via `manager.broadcast_to_presentation()`.

## Performance Note
When disabled (default), the logging function returns immediately with no overhead. Log files are only created when the feature is explicitly enabled.
