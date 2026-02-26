import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Optional
from collections import defaultdict
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse

logger = logging.getLogger(__name__)

APP_NAME = "Slide_creator"

class SlideOperationContextManager:
    """
    Manages shared state for slide operations per presentation.
    This replaces the global defaultdict to provide better isolation.
    """
    def __init__(self):
        self._contexts = defaultdict(lambda: {
            "slide_number": None,
            "slide_index": None,
            "html_content": None,
            "from_validate_insertion": False
        })
    
    def get_context(self, p_id: str, user_id: str) -> dict:
        key = f"{user_id}:{p_id}"
        return self._contexts[key]
    
    def clear_context(self, p_id: str, user_id: str):
        key = f"{user_id}:{p_id}"
        if key in self._contexts:
            del self._contexts[key]

slide_op_manager = SlideOperationContextManager()

def format_db_url_with_ssl(db_url: str) -> str:
    """
    Format database URL with SSL for Supabase/PostgreSQL using asyncpg.
    Forces postgresql+asyncpg:// scheme and uses ssl=require.
    Removes incompatible parameters like sslmode and connect_timeout.
    """
    if not db_url:
        return db_url
    
    # Ensure we use asyncpg driver
    if db_url.startswith("postgresql://"):
        db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    
    parts = urlparse(db_url)
    query = dict(parse_qsl(parts.query))

    # asyncpg uses 'ssl=require' instead of 'sslmode'
    # connect_timeout in query string is not supported by asyncpg driver in this context
    query.pop("sslmode", None)
    query.pop("connect_timeout", None)
    query["ssl"] = "require"

    new_query = urlencode(query)
    formatted = urlunparse(parts._replace(query=new_query))
    return formatted

def get_utc_timestamp_iso() -> str:
    """
    Get current UTC timestamp in ISO 8601 format with timezone info.
    Returns format like: '2024-01-15T10:30:45.123456+00:00'
    This allows frontend to properly convert to user's local timezone.
    """
    return datetime.now(timezone.utc).isoformat()

def convert_datetime_to_iso(dt) -> Optional[str]:
    """
    Convert a datetime object or ISO string (from database) to ISO 8601 format with timezone info.
    Assumes the datetime is UTC if it doesn't have timezone info.
    Returns format like: '2024-01-15T10:30:45.123456+00:00'
    If already a string, returns as-is (assuming it's already in ISO format).
    """
    if dt is None:
        return None
    if isinstance(dt, str):
        # Already an ISO string, return as-is
        return dt
    if isinstance(dt, datetime):
        # If datetime has no timezone info, assume it's UTC
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()
    return str(dt)  # Fallback for other types

def _get_context_key(p_id: str, user_id: str) -> str:
    return f"{user_id}:{p_id}"

def _normalize_tool_args(args) -> dict:
    if isinstance(args, dict):
        return args
    if hasattr(args, "to_dict"):
        try:
            return args.to_dict()
        except Exception:
            pass
    if hasattr(args, "items"):
        try:
            return dict(args.items())
        except Exception:
            pass
    if isinstance(args, str):
        try:
            return json.loads(args)
        except Exception:
            return {}
    try:
        return json.loads(json.dumps(args))
    except Exception:
        return {}

async def safe_append_event(session_service, p_id: str, user_id: str, event, max_retries: int = 3) -> bool:
    """
    Safely append an event to a session, handling stale session errors by re-fetching the session.
    
    Returns True if successful, False otherwise.
    """
    for attempt in range(max_retries):
        try:
            # Re-fetch the session to ensure we have the latest version
            session = await asyncio.wait_for(
                session_service.get_session(app_name=APP_NAME, session_id=p_id, user_id=user_id),
                timeout=30.0  # Increased from 10s to 30s
            )
            if not session:
                logger.warning(f"⚠️ Session not found for p_id={p_id} (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff
                    continue
                return False
            
            # Try to append the event
            await asyncio.wait_for(
                session_service.append_event(session, event),
                timeout=30.0  # Increased from 10s to 30s
            )
            return True
            
        except ValueError as e:
            error_msg = str(e)
            if "stale session" in error_msg.lower() or "last_update_time" in error_msg.lower():
                # Stale session error - retry with fresh session
                logger.debug(f"🔄 Stale session detected (attempt {attempt + 1}/{max_retries}), re-fetching...")
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff
                    continue
                else:
                    logger.warning(f"⚠️ Failed to append event after {max_retries} attempts due to stale session: {e}")
                    return False
            else:
                # Other ValueError - don't retry
                logger.warning(f"⚠️ ValueError while appending event: {e}")
                return False
                
        except asyncio.TimeoutError:
            logger.warning(f"⚠️ Timeout appending event (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                await asyncio.sleep(1.0 * (attempt + 1))  # Longer backoff for timeouts
                continue
            return False
                
        except Exception as e:
            logger.warning(f"⚠️ Error appending event (attempt {attempt + 1}/{max_retries}): {type(e).__name__}: {repr(e)}")
            if attempt < max_retries - 1:
                await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff
                continue
            return False
    
    return False
