import asyncio
import logging
import os
from datetime import datetime, timezone
from core.database import get_mongo_client
from core.socketio_manager import get_manager
from routers.socketio.logic import execute_agent_for_presentation

logger = logging.getLogger(__name__)

async def recover_interrupted_tasks():
    """
    Finds presentations that were interrupted (status 'processing') and restarts them.
    This should be called during application startup.
    """
    logger.info("🔍 Checking for interrupted tasks to recover...")
    
    try:
        client = get_mongo_client()
        db = client["slide_creator_db"]
        
        # Find presentations stuck in 'processing' status
        interrupted_pres = list(db.presentations.find({"status": "processing"}))
        
        if not interrupted_pres:
            logger.info("✅ No interrupted tasks found.")
            return
            
        logger.info(f"💾 Found {len(interrupted_pres)} interrupted tasks. Attempting recovery...")
        
        manager = await get_manager()
        
        for pres in interrupted_pres:
            p_id = pres.get("p_id")
            user_id = pres.get("user_id")
            user_message = pres.get("message") or f"Recovering presentation for p_id: {p_id}"
            
            if not p_id or not user_id:
                logger.warning(f"⚠️ Skipping recovery for malformed presentation: {pres.get('_id')}")
                continue
                
            logger.info(f"🔄 Recovering task: p_id={p_id}, user_id={user_id}")
            
            # Restart the agent execution as a background task
            # Using asyncio.create_task to not block the startup sequence
            asyncio.create_task(execute_agent_for_presentation(p_id, user_id, manager, user_message))
            
        logger.info("🚀 Interrupted task recovery sequence completed.")
        
    except Exception as e:
        logger.error(f"❌ Error during task recovery: {e}", exc_info=True)
