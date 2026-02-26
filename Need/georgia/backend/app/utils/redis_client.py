import redis
from app import config
import logging
import os
from app.utils.debug_logger import debug_log, debug_error

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

QUEUE_NAME = "pdf_processing_queue" # Define queue name here

redis_client = None

def get_redis_client():
    """
    Initializes and returns a Redis client instance.
    Uses configuration from app.config.
    """
    global redis_client
    if redis_client is None:
        try:
            logger.info(f"Attempting to connect to Redis at {config.REDIS_HOST}:{config.REDIS_PORT}, DB: {config.REDIS_DB}")
            # Basic connection pool
            # Use username and password from config if they exist
            connection_kwargs = {
                "host": config.REDIS_HOST,
                "port": config.REDIS_PORT,
                "db": config.REDIS_DB,
                "decode_responses": True # Decode responses to UTF-8 automatically
            }
            # Check if REDIS_USERNAME and REDIS_PASSWORD exist in config and add them
            if hasattr(config, 'REDIS_USERNAME') and config.REDIS_USERNAME:
                connection_kwargs["username"] = config.REDIS_USERNAME
                logger.info("Using Redis username from config.")
            if hasattr(config, 'REDIS_PASSWORD') and config.REDIS_PASSWORD:
                connection_kwargs["password"] = config.REDIS_PASSWORD
                logger.info("Using Redis password from config.")

            pool = redis.ConnectionPool(**connection_kwargs)
            redis_client = redis.Redis(connection_pool=pool)
            # Test connection
            redis_client.ping()
            logger.info("Successfully connected to Redis.")
        except redis.exceptions.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            # Depending on requirements, you might want to raise the error
            # or handle it gracefully (e.g., return None and check elsewhere)
            raise ConnectionError(f"Could not connect to Redis: {e}") from e
        except Exception as e:
            logger.error(f"An unexpected error occurred during Redis connection: {e}")
            raise ConnectionError(f"Unexpected error connecting to Redis: {e}") from e

    return redis_client

# Example usage (optional, can be removed or kept for testing)
if __name__ == '__main__':
    try:
        client = get_redis_client()
        if client:
            debug_log("Redis client obtained successfully.")
            # Example command
            # client.set('mykey', 'hello')
            # value = client.get('mykey')
            # debug_log(f"Got value: {value}")
    except ConnectionError as e:
        debug_error(f"Main execution failed: {e}")
