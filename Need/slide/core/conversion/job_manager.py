import json
import logging
import os
import redis
from datetime import datetime, timezone
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Redis Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)

class JobManager:
    _redis_client = None

    @classmethod
    def get_redis(cls) -> redis.Redis:
        """Get or create singleton Redis client"""
        if cls._redis_client is None:
            try:
                cls._redis_client = redis.Redis(
                    host=REDIS_HOST,
                    port=REDIS_PORT,
                    db=REDIS_DB,
                    password=REDIS_PASSWORD,
                    decode_responses=True
                )
                cls._redis_client.ping()
                logger.info(f"✅ Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
            except Exception as e:
                logger.error(f"❌ Failed to connect to Redis: {e}")
                raise
        return cls._redis_client

    @classmethod
    def _key(cls, job_id: str) -> str:
        return f"job:{job_id}"

    @classmethod
    def create_job(cls, job_id: str, format: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Initialize a new job in Redis"""
        r = cls.get_redis()
        
        job_data = {
            "jobId": job_id,
            "format": format,
            "status": "queued",
            "progress": 0,
            "currentSlide": 0,
            "totalSlides": 0,
            "message": "Job queued",
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "updatedAt": datetime.now(timezone.utc).isoformat(),
            "userId": user_id or ""
        }
        
        # Store as hash
        # We store JSON for complex fields or just flat hash? 
        # Node.js 'bull' stores complex nested, but for simple status, flat hash is fine
        # BUT 'result' might be complex. Let's store the whole thing as a JSON string mostly for simplicity and compatibility with complex nested objects if needed, 
        # OR use HSET with specific fields.
        # Let's use HSET for fields we query and JSON for the rest? 
        # Simpler: Store the whole object as JSON in a single key for now, OR use Redis Hash.
        # The frontend expects specific JSON structure. Let's use Redis Hash for individual fields to allow partial updates (like progress).
        
        # Flattening slightly for Redis Hash
        # Note: Redis Hash values must be strings.
        mapping = {k: str(v) for k, v in job_data.items()}
        r.hset(cls._key(job_id), mapping=mapping)
        # Set expiry (e.g., 24 hours)
        r.expire(cls._key(job_id), 86400)
        
        return job_data

    @classmethod
    def get_job(cls, job_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve job data"""
        r = cls.get_redis()
        data = r.hgetall(cls._key(job_id))
        if not data:
            return None
            
        # Type conversion if needed
        if "progress" in data:
            data["progress"] = int(data["progress"])
        if "currentSlide" in data:
            data["currentSlide"] = int(data["currentSlide"])
        if "totalSlides" in data:
            data["totalSlides"] = int(data["totalSlides"])
            
        # Parse nested result JSON if it exists
        if "result" in data and data["result"]:
            try:
                data["result"] = json.loads(data["result"])
            except:
                pass
                
        return data

    @classmethod
    def is_cancelled(cls, job_id: str) -> bool:
        """Check if a job has been marked as cancelled"""
        r = cls.get_redis()
        status = r.hget(cls._key(job_id), "status")
        return status == "cancelled"

    @classmethod
    def update_progress(cls, job_id: str, progress: int, current_slide: int = 0, message: str = ""):
        """Update job progress"""
        r = cls.get_redis()
        updates = {
            "progress": progress,
            "currentSlide": current_slide,
            "message": message,
            "updatedAt": datetime.now(timezone.utc).isoformat()
        }
        r.hset(cls._key(job_id), mapping=updates)
        # Publish event for SSE? (Optional, can just poll Redis)
        r.publish(f"job_updates:{job_id}", json.dumps(updates))

    @classmethod
    def complete_job(cls, job_id: str, result: Dict[str, Any]):
        """Mark job as completed"""
        r = cls.get_redis()
        updates = {
            "status": "completed",
            "progress": 100,
            "message": "Conversion completed",
            "completedAt": datetime.now(timezone.utc).isoformat(),
            "updatedAt": datetime.now(timezone.utc).isoformat(),
            "result": json.dumps(result) # Store result as JSON string
        }
        r.hset(cls._key(job_id), mapping=updates)
        r.publish(f"job_updates:{job_id}", json.dumps(updates))

    @classmethod
    def fail_job(cls, job_id: str, error_message: str):
        """Mark job as failed"""
        r = cls.get_redis()
        updates = {
            "status": "failed",
            "error": error_message,
            "message": "Conversion failed",
            "completedAt": datetime.now(timezone.utc).isoformat(),
            "updatedAt": datetime.now(timezone.utc).isoformat()
        }
        r.hset(cls._key(job_id), mapping=updates)
        r.publish(f"job_updates:{job_id}", json.dumps(updates))
        
    @classmethod
    def is_cancelled(cls, job_id: str) -> bool:
        """Check if job has been cancelled"""
        r = cls.get_redis()
        status = r.hget(cls._key(job_id), "status")
        return status == "cancelled"

    @classmethod
    def cancel_job(cls, job_id: str):
        """Mark job as cancelled"""
        r = cls.get_redis()
        updates = {
            "status": "cancelled",
            "message": "Job cancelled by user",
            "completedAt": datetime.now(timezone.utc).isoformat(),
            "updatedAt": datetime.now(timezone.utc).isoformat()
        }
        r.hset(cls._key(job_id), mapping=updates)
        r.publish(f"job_updates:{job_id}", json.dumps(updates))
