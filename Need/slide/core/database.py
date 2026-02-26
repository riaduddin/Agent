from datetime import datetime, timezone
import os
from typing import Generator, Dict, List, Any, Optional
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse
from pymongo import MongoClient, DESCENDING, UpdateOne
from pymongo.errors import PyMongoError, ConnectionFailure
from google.adk.sessions import DatabaseSessionService
import logging

from .models import PresentationModel, SlideModel, AgentOutputModel

APP_NAME = "Slide_creator"


logger = logging.getLogger(__name__)

_MONGO_CLIENT: Optional[MongoClient] = None

def get_mongo_client() -> MongoClient:
    """
    Lazily create a single MongoClient per process.
    PyMongo's MongoClient is thread-safe and uses an internal pool.
    """
    global _MONGO_CLIENT
    if _MONGO_CLIENT is None:
        mongo_uri = os.getenv("MONGODB_URL")
        if not mongo_uri:
            raise ValueError("MONGODB_URL environment variable is required")
        
        # Masked URI for safe logging
        try:
            from urllib.parse import urlparse
            p = urlparse(mongo_uri)
            creds = p.netloc.split('@')[0] if '@' in p.netloc else ""
            masked_netloc = p.netloc.replace(creds, "***") if creds else p.netloc
            masked_uri = p._replace(netloc=masked_netloc).geturl()
            logger.info(f"📡 Initializing MongoDB connection with URI: {masked_uri}")
        except Exception:
            logger.info("📡 Initializing MongoDB connection")
        
        # Tune pool + timeouts as needed
        _MONGO_CLIENT = MongoClient(
            mongo_uri,
            maxPoolSize=20,         # total pooled sockets per process
            minPoolSize=8,           # keep warm sockets
            waitQueueTimeoutMS=5000, # how long to wait for a socket from pool
            serverSelectionTimeoutMS=5000,  # fail fast if cluster unreachable
            retryWrites=True,
            connect=False,           # lazy connect; set True to connect at import
        )
    return _MONGO_CLIENT

def get_db():
    """
    FastAPI dependency to get the DB handle using the pooled client.
    """
    client = get_mongo_client()
    return client["slide_creator_db"]

def get_collections(db):
    """
    Helper to fetch your collections in one place.
    """
    return {
        "agent_outputs": db["agent_outputs_2"],
        "agent_logs": db["agent_logs"],
        "presentations": db["presentations"],
        "slides": db["slides"],
        "slide_status": db["slide_status"],
        "slide_html": db["slide_html"],
        "references": db["references"],
    }

def get_presentations_by_user(db, user_id: str) -> List[Dict[str, Any]]:
    """
    Get all presentations for a specific user, sorted by completion date (newest first).
    """
    try:
        col = db["presentations"]
        cursor = col.find(
            {"user_id": user_id},
            {"_id": 0, "p_id": 1, "title": 1, "creation_date": 1, "completion_date": 1, "status": 1},
        ).sort("completion_date", DESCENDING)
        return list(cursor)
    except PyMongoError as e:
        logger.error(f"❌ MongoDB error fetching presentations for user {user_id}: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error fetching presentations for user {user_id}: {e}", exc_info=True)
        raise

def get_slides_by_p_id(db, p_id: str) -> Dict[str, Any]:
    """
    Get all slides for a presentation by presentation ID.
    """
    try:
        slide_html = db["slide_html"]
        presentations = db["presentations"]

        slides_cur = slide_html.find(
            {"p_id": p_id},
            {"_id": 0, "slide_plan": 1, "slide_index": 1, "body": 1, "thought": 1},
        ).sort("slide_index", 1)

        status_doc = presentations.find_one(
            {"p_id": p_id},
            {"_id": 0, "status": 1, "title": 1, "total_slides": 1},
        )

        slides_list = list(slides_cur)
        if status_doc:
            return {
                "slides": slides_list,
                "status": status_doc.get("status", ""),
                "title": status_doc.get("title", ""),
                "total_slides": status_doc.get("total_slides", 0),
            }
        return {"slides": slides_list, "status": "processing"}
    except PyMongoError as e:
        logger.error(f"❌ MongoDB error fetching slides for presentation {p_id}: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error fetching slides for presentation {p_id}: {e}", exc_info=True)
        raise

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

def get_agent_logs_by_p_id(db, p_id: str) -> Dict[str, Any]:
    """
    Get all agent logs for a presentation by presentation ID.
    Excludes enhanced_slide_generator outputs (they're in the slides field).
    Converts timestamps to ISO 8601 format with timezone info for frontend compatibility.
    """
    try:
        agent_outputs = db["agent_outputs_2"]
        presentations = db["presentations"]

        # Exclude enhanced_slide_generator agents from logs
        logs_cur = agent_outputs.find(
            {
                "p_id": p_id,
                "agent_name": {"$not": {"$regex": "^enhanced_slide_generator"}}
            },
            {"_id": 0, "role": 1, "message": 1,"user_message": 1, "author": 1, "timestamp": 1, "parsed_output": 1,"file_urls": 1 },
        ).sort("timestamp", 1)

        # Convert timestamps to ISO format with timezone
        logs = []
        for log in logs_cur:
            if "timestamp" in log:
                log["timestamp"] = convert_datetime_to_iso(log["timestamp"])
            logs.append(log)

        status_doc = presentations.find_one({"p_id": p_id}, {"_id": 0, "status": 1})
        
        # Get slide data from the slides collection instead of agent_outputs_2
        # This part was missing after the refactoring and is now restored
        slide_outputs = list(db["slide_html"].find(
            {"p_id": p_id},
            {
                "_id": 0,
                "slide_number": 1,
                "thinking": 1,
                "body": 1,
                "created_at": 1,
                "updated_at": 1
            }
        ).sort("slide_number", 1))
        
        # Format slides for the response
        slides = []
        for slide in slide_outputs:
            slide_info = {
                "slide_number": slide.get("slide_number"),
                "thinking": slide.get("thinking", ""),
                "html_content": slide.get("body", ""),  # Use body field as html_content
                "timestamp": slide.get("created_at") or slide.get("updated_at")
            }
            
            # Use same ISO format as logs
            if slide_info["timestamp"]:
                slide_info["timestamp"] = convert_datetime_to_iso(slide_info["timestamp"])
                
            # Only include slides that have HTML content
            if slide_info["html_content"]:
                # Remove None values for cleaner response
                cleaned_slide = {k: v for k, v in slide_info.items() if v is not None}
                slides.append(cleaned_slide)

        return {
            "logs": logs,
            "status": status_doc["status"] if status_doc else "processing",
            "slides": slides
        }
    except PyMongoError as e:
        logger.error(f"❌ MongoDB error fetching agent logs for presentation {p_id}: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error fetching agent logs for presentation {p_id}: {e}", exc_info=True)
        raise

def create_presentation(db, presentation_data: Dict[str, Any]) -> str:
    """
    Create a new presentation record.
    """
    try:
        col = db["presentations"]
        result = col.insert_one(presentation_data)
        return str(result.inserted_id)
    except PyMongoError as e:
        logger.error(f"❌ MongoDB error creating presentation: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error creating presentation: {e}", exc_info=True)
        raise

def update_presentation_status(db, p_id: str, status: str, **kwargs) -> bool:
    """
    Update presentation status and other fields.
    """
    try:
        col = db["presentations"]
        update_data = {"status": status, **kwargs}
        result = col.update_one(
            {"p_id": p_id},
            {"$set": update_data}
        )
        return result.modified_count > 0
    except PyMongoError as e:
        logger.error(f"❌ MongoDB error updating presentation {p_id} status to {status}: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error updating presentation {p_id} status to {status}: {e}", exc_info=True)
        raise

def insert_slide(db, slide_data: Dict[str, Any]) -> str:
    """
    Insert a new slide.
    """
    try:
        col = db["slide_html"]
        result = col.insert_one(slide_data)
        return str(result.inserted_id)
    except PyMongoError as e:
        logger.error(f"❌ MongoDB error inserting slide: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error inserting slide: {e}", exc_info=True)
        raise

def insert_agent_log(db, log_data: Dict[str, Any]) -> str:
    """
    Insert a new agent log entry (user-facing).
    """
    try:
        col = db["agent_outputs_2"]
        result = col.insert_one(log_data)
        return str(result.inserted_id)
    except PyMongoError as e:
        logger.error(f"❌ MongoDB error inserting agent log: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error inserting agent log: {e}", exc_info=True)
        raise

def insert_dev_log(db, log_data: Dict[str, Any]) -> str:
    """
    Insert a new developer log entry (for agent_logs collection).
    """
    try:
        col = db["agent_logs"]
        result = col.insert_one(log_data)
        return str(result.inserted_id)
    except PyMongoError as e:
        logger.error(f"❌ MongoDB error inserting dev log: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error inserting dev log: {e}", exc_info=True)
        raise

def get_presentation_by_id(db, p_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a presentation by its ID.
    """
    try:
        col = db["presentations"]
        return col.find_one({"p_id": p_id}, {"_id": 0})
    except PyMongoError as e:
        logger.error(f"❌ MongoDB error fetching presentation {p_id}: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error fetching presentation {p_id}: {e}", exc_info=True)
        raise

def delete_presentation(db, p_id: str) -> bool:
    """
    Delete a presentation and all related data.
    """
    try:
        # Delete from all related collections
        collections_to_clean = ["presentations", "slide_html", "agent_outputs_2", "slide_status", "references"]
        
        for collection_name in collections_to_clean:
            col = db[collection_name]
            col.delete_many({"p_id": p_id})
        
        return True
    except PyMongoError as e:
        logger.error(f"❌ MongoDB error deleting presentation {p_id}: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error deleting presentation {p_id}: {e}", exc_info=True)
        raise

def clone_presentation_transactional(db, original_p_id: str, new_p_id: str, new_user_id: str, new_user_email: str, new_user_verified: bool, new_user_package: str) -> bool:
    """
    Clones a presentation and all related documents atomically using a MongoDB transaction.
    """
    client = db.client
    with client.start_session() as session:
        def callback(session):
            # 1. Fetch original
            orig = db.presentations.find_one({"p_id": original_p_id}, session=session)
            if not orig:
                raise ValueError("Original presentation not found")
            
            # 2. Insert new presentation
            now_iso = datetime.now(timezone.utc).isoformat()
            
            new_pres = {
                "p_id": new_p_id,
                "session_id": new_p_id,
                "user_id": new_user_id,
                "user_email": new_user_email,
                "user_verified": new_user_verified,
                "user_package": new_user_package,
                "creation_date": now_iso,
                "updated_at": now_iso,
                "status": "completed",
                "cloned_from": original_p_id,
                "title": orig.get("title", ""),
                "total_slides": orig.get("total_slides", 0)
            }
            db.presentations.insert_one(new_pres, session=session)
            
            # 3. Clone related collections
            for coll_name in ["slide_html", "agent_outputs_2", "references", "slide_status"]:
                items = list(db[coll_name].find({"p_id": original_p_id}, session=session))
                if items:
                    for item in items:
                        item.pop("_id", None)
                        item["p_id"] = new_p_id
                        if "user_id" in item: item["user_id"] = new_user_id
                        if "session_id" in item: item["session_id"] = new_p_id
                    db[coll_name].insert_many(items, session=session)
            
            return True

        try:
            session.with_transaction(callback)
            return True
        except Exception as e:
            logger.error(f"❌ Transaction failed during presentation cloning: {e}", exc_info=True)
            raise

def get_user_by_id(db, user_id: str) -> Optional[Dict[str, Any]]:
    """
    Get user document by ID. Handles both string and ObjectId formats.
    """
    from bson import ObjectId
    try:
        # Try as ObjectId if it looks like one, else literal string
        target_id = user_id
        if len(user_id) == 24 and all(c in "0123456789abcdef" for c in user_id.lower()):
            target_id = ObjectId(user_id)
        
        return db.users.find_one({"_id": target_id})
    except Exception as e:
        logger.error(f"Error fetching user {user_id}: {e}")
        return None

def check_db_health(db) -> bool:
    """
    Check if database connection is healthy.
    """
    try:
        # Simple ping to check connection
        db.command('ping')
        return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False

def format_db_url_with_ssl(db_url: str) -> str:
    """
    Format database URL for asyncpg with SSL required.
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

    query.pop("sslmode", None)
    query.pop("connect_timeout", None)
    query["ssl"] = "require"

    new_query = urlencode(query)
    formatted = urlunparse(parts._replace(query=new_query))
    return formatted

_SESSION_SERVICE = None

def get_session_service():
    """Get the singleton session service instance."""
    global _SESSION_SERVICE
    if _SESSION_SERVICE is None:
        db_url = os.getenv("DATABASE_URL")
        formatted_url = format_db_url_with_ssl(db_url)
        logger.info("Initializing global DatabaseSessionService...")
        _SESSION_SERVICE = DatabaseSessionService(db_url=formatted_url)
    return _SESSION_SERVICE