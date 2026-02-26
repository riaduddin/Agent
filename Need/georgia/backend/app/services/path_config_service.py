# backend/app/services/path_config_service.py
from app import db
from datetime import datetime, timezone
import logging
import time

logger = logging.getLogger(__name__)

PATH_CONFIG_COLLECTION = "path_config"
DEFAULT_CONFIG_ID = "default"

# In-memory cache for path config. Invalidated immediately on write.
# _path_config_cache_populated distinguishes "never fetched" from "fetched, result was None".
_PATH_CONFIG_TTL = 60  # seconds
_path_config_cache = None
_path_config_cache_ts = 0.0
_path_config_cache_populated = False

def get_path_config():
    """Get the current path configuration from Firestore."""
    global _path_config_cache, _path_config_cache_ts, _path_config_cache_populated
    now = time.time()
    if _path_config_cache_populated and (now - _path_config_cache_ts) < _PATH_CONFIG_TTL:
        return _path_config_cache
    try:
        config_ref = db.collection(PATH_CONFIG_COLLECTION).document(DEFAULT_CONFIG_ID)
        config_doc = config_ref.get()

        if not config_doc.exists:
            # Cache the "no document" result so we don't keep hitting Firestore
            _path_config_cache = None
            _path_config_cache_ts = now
            _path_config_cache_populated = True
            return None

        config_data = config_doc.to_dict()
        result = {
            "base_path": config_data.get("base_path", ""),
            "source_folder": config_data.get("source_folder", ""),
            "last_updated_by": config_data.get("last_updated_by", ""),
            "last_updated_at": config_data.get("last_updated_at")
        }
        _path_config_cache = result
        _path_config_cache_ts = now
        _path_config_cache_populated = True
        return result
    except Exception as e:
        logger.error(f"Error getting path config: {e}")
        return None

def set_path_config(base_path: str, source_folder: str, user_email: str):
    """Set the path configuration in Firestore. Only superadmin can call this."""
    global _path_config_cache_populated
    try:
        # Normalize paths
        base_path = base_path.strip()
        if base_path:
            base_path = base_path.rstrip('/')
            if not base_path.endswith('/'):
                base_path += '/'

        source_folder = source_folder.strip().strip('/')

        config_ref = db.collection(PATH_CONFIG_COLLECTION).document(DEFAULT_CONFIG_ID)
        config_ref.set({
            "base_path": base_path,
            "source_folder": source_folder,
            "last_updated_by": user_email,
            "last_updated_at": datetime.now(timezone.utc)
        }, merge=True)

        _path_config_cache_populated = False  # invalidate cache so next read fetches fresh data
        logger.info(f"Path config updated by {user_email}: base_path={base_path}, source_folder={source_folder}")
        return True, None
    except Exception as e:
        logger.error(f"Error setting path config: {e}")
        return False, str(e)

def get_dynamic_source_path():
    """Get the full source path by combining base_path and source_folder."""
    config = get_path_config()
    
    if not config:
        # Fallback to environment variable if no config exists
        from app.config import GCS_SOURCE_ROOT
        return GCS_SOURCE_ROOT
    
    base_path = config.get("base_path", "").strip()
    source_folder = config.get("source_folder", "").strip()
    
    if not base_path and not source_folder:
        # Fallback to environment variable
        from app.config import GCS_SOURCE_ROOT
        return GCS_SOURCE_ROOT
    
    # Combine base_path and source_folder
    if base_path and source_folder:
        # Remove trailing slash from base_path if present, then combine with slash
        base_path = base_path.rstrip('/')
        source_folder = source_folder.strip('/')
        return f"{base_path}/{source_folder}/"
    elif base_path:
        return base_path if base_path.endswith('/') else f"{base_path}/"
    elif source_folder:
        return f"{source_folder}/" if not source_folder.endswith('/') else source_folder
    
    # Final fallback
    from app.config import GCS_SOURCE_ROOT
    return GCS_SOURCE_ROOT

def get_dynamic_base_path():
    """Get the base path from configuration."""
    config = get_path_config()
    
    if not config:
        # Extract base path from GCS_SOURCE_ROOT as fallback
        from app.config import GCS_SOURCE_ROOT
        source_parts = GCS_SOURCE_ROOT.rstrip('/').split('/')
        if len(source_parts) > 0:
            return f"{source_parts[0]}/"
        return ""
    
    base_path = config.get("base_path", "").strip()
    if base_path:
        return base_path if base_path.endswith('/') else f"{base_path}/"
    
    # Fallback: extract from source path
    source_path = get_dynamic_source_path()
    source_parts = source_path.rstrip('/').split('/')
    if len(source_parts) > 0:
        return f"{source_parts[0]}/"
    
    return ""

def get_all_path_configs():
    """Get all path configurations from Firestore. Returns list of all config documents."""
    try:
        configs_ref = db.collection(PATH_CONFIG_COLLECTION)
        configs_docs = configs_ref.stream()
        
        all_configs = []
        for doc in configs_docs:
            config_data = doc.to_dict()
            all_configs.append({
                "id": doc.id,
                "base_path": config_data.get("base_path", ""),
                "source_folder": config_data.get("source_folder", ""),
                "last_updated_by": config_data.get("last_updated_by", ""),
                "last_updated_at": config_data.get("last_updated_at")
            })
        
        return all_configs, None
    except Exception as e:
        logger.error(f"Error getting all path configs: {e}")
        return None, str(e)

def get_path_configs_for_user(user_email: str):
    """Get path configurations updated by a specific user."""
    try:
        configs_ref = db.collection(PATH_CONFIG_COLLECTION)
        # Query for configs where last_updated_by matches the user email
        configs_query = configs_ref.where("last_updated_by", "==", user_email)
        configs_docs = configs_query.stream()
        
        user_configs = []
        for doc in configs_docs:
            config_data = doc.to_dict()
            user_configs.append({
                "id": doc.id,
                "base_path": config_data.get("base_path", ""),
                "source_folder": config_data.get("source_folder", ""),
                "last_updated_by": config_data.get("last_updated_by", ""),
                "last_updated_at": config_data.get("last_updated_at")
            })
        
        return user_configs, None
    except Exception as e:
        logger.error(f"Error getting path configs for user {user_email}: {e}")
        return None, str(e)

