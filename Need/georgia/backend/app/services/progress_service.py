"""
Progress tracking service for file and folder transfers.
Uses Redis for centralized storage across multiple replicas.
"""
import time
import threading
import json
from typing import Dict, Optional, Callable
from datetime import datetime
from app.utils.redis_client import get_redis_client
import logging

logger = logging.getLogger(__name__)

class ProgressTracker:
    def __init__(self):
        self.redis_client = None
        self.redis_key_prefix = "transfer_progress:"
        self.lock = threading.Lock()
        
    def _get_redis_client(self):
        """Get Redis client with lazy initialization."""
        if self.redis_client is None:
            try:
                self.redis_client = get_redis_client()
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                raise
        return self.redis_client
    
    def _get_redis_key(self, transfer_id: str) -> str:
        """Generate Redis key for a transfer ID."""
        return f"{self.redis_key_prefix}{transfer_id}"
    
    def create_transfer(self, transfer_id: str, transfer_type: str, operation: str, total_items: int) -> None:
        """Create a new transfer progress entry."""
        try:
            redis_client = self._get_redis_client()
            transfer_data = {
                'id': transfer_id,
                'type': transfer_type,  # 'file' or 'folder'
                'operation': operation,  # 'copy' or 'move'
                'total_items': total_items,
                'completed_items': 0,
                'current_item': None,
                'status': 'pending',  # 'pending', 'in_progress', 'completed', 'failed'
                'error': None,
                'start_time': datetime.now().isoformat(),
                'end_time': None
            }
            
            redis_key = self._get_redis_key(transfer_id)
            # Store in Redis with 24-hour expiration
            redis_client.setex(redis_key, 86400, json.dumps(transfer_data))
            logger.info(f"Created transfer progress entry in Redis: {transfer_id}")
        except Exception as e:
            logger.error(f"Failed to create transfer in Redis: {e}")
            raise
    
    def start_transfer(self, transfer_id: str) -> None:
        """Mark transfer as started."""
        try:
            redis_client = self._get_redis_client()
            redis_key = self._get_redis_key(transfer_id)
            
            # Get current transfer data
            transfer_data_str = redis_client.get(redis_key)
            if transfer_data_str:
                transfer_data = json.loads(transfer_data_str)
                transfer_data['status'] = 'in_progress'
                transfer_data['start_time'] = datetime.now().isoformat()
                
                # Update in Redis with 24-hour expiration
                redis_client.setex(redis_key, 86400, json.dumps(transfer_data))
                logger.info(f"Started transfer: {transfer_id}")
            else:
                logger.warning(f"Transfer not found in Redis: {transfer_id}")
        except Exception as e:
            logger.error(f"Failed to start transfer in Redis: {e}")
            raise
    
    def update_progress(self, transfer_id: str, completed_items: int, current_item: Optional[str] = None) -> None:
        """Update transfer progress."""
        try:
            redis_client = self._get_redis_client()
            redis_key = self._get_redis_key(transfer_id)
            
            # Get current transfer data
            transfer_data_str = redis_client.get(redis_key)
            if transfer_data_str:
                transfer_data = json.loads(transfer_data_str)
                transfer_data['completed_items'] = completed_items
                if current_item:
                    transfer_data['current_item'] = current_item
                
                # Update in Redis with 24-hour expiration
                redis_client.setex(redis_key, 86400, json.dumps(transfer_data))
            else:
                logger.warning(f"Transfer not found in Redis for progress update: {transfer_id}")
        except Exception as e:
            logger.error(f"Failed to update progress in Redis: {e}")
            # Don't raise here as this is called frequently during transfers
    
    def complete_transfer(self, transfer_id: str) -> None:
        """Mark transfer as completed."""
        try:
            redis_client = self._get_redis_client()
            redis_key = self._get_redis_key(transfer_id)
            
            # Get current transfer data
            transfer_data_str = redis_client.get(redis_key)
            if transfer_data_str:
                transfer_data = json.loads(transfer_data_str)
                transfer_data['status'] = 'completed'
                transfer_data['end_time'] = datetime.now().isoformat()
                transfer_data['current_item'] = None
                
                # Update in Redis with 24-hour expiration
                redis_client.setex(redis_key, 86400, json.dumps(transfer_data))
                logger.info(f"Completed transfer: {transfer_id}")
            else:
                logger.warning(f"Transfer not found in Redis for completion: {transfer_id}")
        except Exception as e:
            logger.error(f"Failed to complete transfer in Redis: {e}")
            raise
    
    def fail_transfer(self, transfer_id: str, error: str) -> None:
        """Mark transfer as failed."""
        try:
            redis_client = self._get_redis_client()
            redis_key = self._get_redis_key(transfer_id)
            
            # Get current transfer data
            transfer_data_str = redis_client.get(redis_key)
            if transfer_data_str:
                transfer_data = json.loads(transfer_data_str)
                transfer_data['status'] = 'failed'
                transfer_data['error'] = error
                transfer_data['end_time'] = datetime.now().isoformat()
                
                # Update in Redis with 24-hour expiration
                redis_client.setex(redis_key, 86400, json.dumps(transfer_data))
                logger.info(f"Failed transfer: {transfer_id}, error: {error}")
            else:
                logger.warning(f"Transfer not found in Redis for failure: {transfer_id}")
        except Exception as e:
            logger.error(f"Failed to mark transfer as failed in Redis: {e}")
            raise
    
    def get_transfer(self, transfer_id: str) -> Optional[dict]:
        """Get transfer progress by ID."""
        try:
            redis_client = self._get_redis_client()
            redis_key = self._get_redis_key(transfer_id)
            
            transfer_data_str = redis_client.get(redis_key)
            if transfer_data_str:
                return json.loads(transfer_data_str)
            else:
                return None
        except Exception as e:
            logger.error(f"Failed to get transfer from Redis: {e}")
            return None
    
    def get_all_transfers(self) -> Dict[str, dict]:
        """Get all transfers."""
        try:
            redis_client = self._get_redis_client()
            
            # Get all keys matching our pattern
            pattern = f"{self.redis_key_prefix}*"
            keys = redis_client.keys(pattern)
            
            transfers = {}
            for key in keys:
                transfer_data_str = redis_client.get(key)
                if transfer_data_str:
                    transfer_data = json.loads(transfer_data_str)
                    transfer_id = transfer_data.get('id')
                    if transfer_id:
                        transfers[transfer_id] = transfer_data
            
            return transfers
        except Exception as e:
            logger.error(f"Failed to get all transfers from Redis: {e}")
            return {}
    
    def cleanup_old_transfers(self, max_age_hours: int = 24) -> None:
        """Remove transfers older than max_age_hours."""
        try:
            redis_client = self._get_redis_client()
            cutoff_time = time.time() - (max_age_hours * 3600)
            
            # Get all transfer keys
            pattern = f"{self.redis_key_prefix}*"
            keys = redis_client.keys(pattern)
            
            removed_count = 0
            for key in keys:
                transfer_data_str = redis_client.get(key)
                if transfer_data_str:
                    transfer_data = json.loads(transfer_data_str)
                    if transfer_data.get('status') in ['completed', 'failed']:
                        start_time_str = transfer_data.get('start_time')
                        if start_time_str:
                            try:
                                start_time = datetime.fromisoformat(start_time_str).timestamp()
                                if start_time < cutoff_time:
                                    redis_client.delete(key)
                                    removed_count += 1
                            except ValueError:
                                # Invalid datetime format, remove it
                                redis_client.delete(key)
                                removed_count += 1
            
            if removed_count > 0:
                logger.info(f"Cleaned up {removed_count} old transfers from Redis")
        except Exception as e:
            logger.error(f"Failed to cleanup old transfers from Redis: {e}")
    
    def create_progress_callback(self, transfer_id: str) -> Callable:
        """Create a progress callback function for a specific transfer."""
        def progress_callback(completed_items: int, total_items: int, current_item: str = None):
            self.update_progress(transfer_id, completed_items, current_item)
        
        return progress_callback

# Global progress tracker instance
progress_tracker = ProgressTracker()

def get_progress_tracker() -> ProgressTracker:
    """Get the global progress tracker instance."""
    return progress_tracker
