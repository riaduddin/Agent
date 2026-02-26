import os
import time
import logging
import json
from datetime import datetime
from pymongo import MongoClient
from pymongo.errors import PyMongoError
import redis.asyncio as redis
from pathlib import Path
import sys

# Setup logging
logger = logging.getLogger(__name__)

# Environment variables
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = os.getenv("REDIS_PORT", 6379)
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
MONGO_URI = os.getenv("MONGODB_URL", "mongodb://localhost:27017")

# Clients
try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    # Force a server selection to verify connection
    client.admin.command('ping')
    db = client["slide_creator_db"]
except Exception as e:
    logger.error(f"❌ MongoDB connection initialization failed: {e}")
    client = None
    db = None

try:
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD, decode_responses=False)
except Exception as e:
    logger.error(f"❌ Redis connection initialization failed: {e}")
    redis_client = None

def get_db():
    if db is None:
        logger.warning("MongoDB client not initialized; returning None")
    return db

def get_redis_client():
    return redis_client

async def test_redis_connection():
    """Test Redis connection properly"""
    if redis_client is None:
        return False
    try:
        result = await redis_client.ping()
        return True
    except Exception as e:
        logger.error(f"❌ Redis connection failed: {e}")
        return False

def safe_db_insert(collection, document, max_retries=3):
    """Database insert with retry and proper error handling"""
    for attempt in range(max_retries):
        try:
            logger.debug(f"Attempting to insert document (attempt {attempt + 1})")
            result = collection.insert_one(document)
            logger.debug(f"✅ DB Insert successful: {result.inserted_id}")
            return result
        except PyMongoError as e:
            logger.error(f"❌ DB Insert failed (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                return None
            time.sleep(2 ** attempt)
        except Exception as e:
            logger.error(f"❌ Unexpected error during DB Insert: {e}")
            return None

def safe_db_update(collection, filter_dict, update_dict, max_retries=3):
    """Database update with retry and proper error handling"""
    for attempt in range(max_retries):
        try:
            logger.debug(f"Updating the collection (attempt {attempt + 1})")
            result = collection.update_one(filter_dict, update_dict)
            logger.debug(f"✅ DB Update successful: modified {result.modified_count}")
            return result
        except PyMongoError as e:
            logger.error(f"❌ DB Update failed (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                return None
            time.sleep(2 ** attempt)
        except Exception as e:
            logger.error(f"❌ Unexpected error during DB Update: {e}")
            return None

def log_event_to_db(event, db, type=None, session_id=None, user_id=None, reference=False, final_content_summary=None):
    if reference:
        db.references.insert_one({
            "event_id": event.id,
            "name": getattr(event, "name", None),
            "author": getattr(event, "author", None),
            "timestamp": getattr(event, "timestamp", datetime.utcnow()),
            "reference": final_content_summary,
            "session_id": session_id,
            "user_id": user_id
        })
    else:
        # Collect content from parts
        content_parts = []
        if event.content and event.content.parts:
            for part in event.content.parts:
                text = getattr(part, "text", None)
                if text:
                    content_parts.append(text)
        
        content_str = "\n".join(content_parts) if content_parts else None
        current_time = datetime.utcnow()

        # 1. Developer Log (agent_logs) - Flat schema as requested
        dev_log_entry = {
            "event_id": event.id,
            "name": getattr(event, "name", None),
            "author": getattr(event, "author", None),
            "timestamp": getattr(event, "timestamp", current_time),
            "session_id": session_id,
            "user_id": user_id,
            "type": type,
            "content": content_str
        }
        db.agent_logs.insert_one(dev_log_entry)
