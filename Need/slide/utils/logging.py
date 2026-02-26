# utils_logging.py
"""
Utility functions for logging events to database.
This module avoids circular imports by being independent.
"""

from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def log_event_to_db(event, db, type=None, session_id=None, user_id=None, output=None, reference=False, final_content_summary=None):
    """
    Log agent events to database.
    This is a standalone utility function to avoid circular imports.
    """
    try:
        if reference:
            db.references.insert_one({
                "event_id": getattr(event, "id", None),
                "name": getattr(event, "name", None),
                "author": getattr(event, "author", None),
                "timestamp": str(getattr(event, "timestamp", datetime.utcnow())),
                "reference": final_content_summary,
                "session_id": session_id,
                "user_id": user_id
            })
        else:
            # Safely extract content from event
            content = getattr(event, "content", None)
            content_str = None
            
            if content:
                try:
                    # Handle Google GenAI Content objects
                    if hasattr(content, 'parts'):
                        # Extract text from parts
                        text_parts = []
                        for part in content.parts:
                            if hasattr(part, 'text') and part.text:
                                text_parts.append(part.text)
                            elif hasattr(part, 'function_call'):
                                # Handle function calls
                                func_call = part.function_call
                                if hasattr(func_call, 'name') and hasattr(func_call, 'args'):
                                    text_parts.append(f"Function: {func_call.name}({func_call.args})")
                            elif hasattr(part, 'function_response'):
                                # Handle function responses
                                func_resp = part.function_response
                                if hasattr(func_resp, 'response'):
                                    text_parts.append(f"Response: {func_resp.response}")
                        
                        content_str = "\n".join(text_parts) if text_parts else str(content)
                    else:
                        content_str = str(content)
                except Exception as content_error:
                    logger.warning(f"⚠️ Failed to extract content: {content_error}")
                    content_str = str(content)
            
            # 1. Log to agent_outputs_2 collection (User Progress) - Exclude technical generator events
            author = getattr(event, "author", "") or ""
            # Exclude internal/technical agents from user-facing logs
            excluded_authors = ["enhanced_slide_generator", "template_selector"]
            if not any(excluded in author for excluded in excluded_authors):
                db["agent_outputs_2"].insert_one({
                    "event_id": getattr(event, "id", None),
                    "name": getattr(event, "name", None),
                    "author": author,
                    "timestamp": datetime.utcnow(),
                    "session_id": session_id,
                    "user_id": user_id,
                    "type": type,
                    "content": content_str or output
                })

            # 2. Log to agent_logs collection (Developer Audit)
            db["agent_logs"].insert_one({
                "event_id": getattr(event, "id", None),
                "name": getattr(event, "name", None),
                "author": getattr(event, "author", None),
                "timestamp": datetime.utcnow(), # MongoDB handles ISODate
                "session_id": session_id,
                "user_id": user_id,
                "type": type,
                "content": content_str or output
            })
            
            # Also emit a structured log to stdout (Auditor's recommendation)
            import json
            log_payload = {
                "event": "agent_event",
                "p_id": session_id,
                "user_id": user_id,
                "author": getattr(event, "author", None),
                "type": type
            }
            logger.info(f"📊 Agent Log: {json.dumps(log_payload)}")
            
    except Exception as e:
        logger.error(f"❌ Failed to log event: {type(e).__name__}: {e}", exc_info=True)
