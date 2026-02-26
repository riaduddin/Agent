import asyncio
import logging
import os
import httpx
from fastapi import APIRouter, HTTPException, Query, Depends, Header, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from typing import Optional
from bson import ObjectId

from core.socketio_manager import get_manager
from core.database import get_db, APP_NAME
from middleware.auth import get_current_user, AuthenticatedUser, decode_jwt_token

from .models import PresentationRequest, StartPresentationRequest, ClonePresentationRequest
from .utils import (
    get_utc_timestamp_iso, 
    convert_datetime_to_iso
)
from .token_tracker import token_api_base_url
from .logic import execute_agent_for_presentation

router = APIRouter(tags=["presentations"])
logger = logging.getLogger(__name__)

@router.post("/create-presentation")
async def create_presentation(
    request: PresentationRequest,
    current_user: AuthenticatedUser = Depends(get_current_user),
    db=Depends(get_db)
    ):
    """
    Create or update a presentation.
    """
    try:
        user_id = current_user.user_id
        message = request.message
        if not message:
            raise HTTPException(status_code=400, detail="Message is required")
        
        p_id = request.p_id if request.p_id else str(ObjectId())

        # Check token availability
        server_api_key = os.getenv("SERVER_API_KEY")
        feature_endpoint_id = os.getenv("FEATURE_ENDPOINT_ID")
        
        if server_api_key and feature_endpoint_id:
            try:
                token_check_url = f"{token_api_base_url}/api/token-process/start"
                headers = {"x-server-api-key": server_api_key, "Content-Type": "application/json"}
                payload = {"user_id": user_id, "feature_endpoint_id": feature_endpoint_id}
                
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.post(token_check_url, json=payload, headers=headers)
                    response.raise_for_status()
                    result = response.json()
                    
                    if result.get("success") and result.get("status") == 200:
                        data = result.get("data", {})
                        usage_key = data.get("usage_key")
                        if usage_key:
                            db.presentations.update_one({"p_id": p_id}, {"$set": {"usage_key": usage_key}})
                            
                        if data.get("status") != "accessible":
                            db.presentations.update_one({"p_id": p_id}, {"$set": {"status": "completed", "error_message": "ACCESS_DENIED", "updated_at": get_utc_timestamp_iso()}})
                            return JSONResponse(status_code=403, content={"error": "ACCESS_DENIED", "message": data.get("message", "Insufficient tokens"), "currentBalance": data.get("token", 0)})
            except Exception as e:
                logger.error(f"Token check failed: {e}")
                return JSONResponse(status_code=500, content={"error": "TOKEN_PROCESS_ERROR", "message": "Failed to verify token access."})
        
        existing_presentation = await run_in_threadpool(db.presentations.find_one, {"p_id": p_id})
        
        if existing_presentation:
            if existing_presentation.get("user_id") != user_id:
                raise HTTPException(status_code=403, detail="Access denied")
            
            await run_in_threadpool(db.presentations.update_one, {"p_id": p_id}, {"$set": {"message": message, "file_urls": request.file_urls or [], "status": "queued", "updated_at": get_utc_timestamp_iso()}})
            await run_in_threadpool(db.job_requests.update_one, {"p_id": p_id}, {"$set": {"message": message, "status": "queued", "created_at": get_utc_timestamp_iso()}}, upsert=True)
            
            return {"message": "Presentation updated successfully", "p_id": p_id, "status": "queued", "user_id": user_id}
        else:
            presentation_data = {
                "p_id": p_id, "user_id": user_id, "user_email": current_user.email,
                "user_verified": current_user.is_verified, "user_package": current_user.package,
                "message": message, "file_urls": request.file_urls or [], "status": "queued",
                "creation_date": get_utc_timestamp_iso(), "updated_at": get_utc_timestamp_iso()
            }
            await run_in_threadpool(db.presentations.insert_one, presentation_data)
            await run_in_threadpool(db.job_requests.insert_one, {"p_id": p_id, "user_id": user_id, "message": message, "status": "queued", "created_at": get_utc_timestamp_iso()})
            
            return {"message": "Presentation created successfully", "p_id": p_id, "status": "queued", "user_id": user_id}
    except HTTPException: raise
    except Exception as e:
        logger.error(f"Error creating presentation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/start-presentation/{p_id}")
async def start_presentation(
    p_id: str,
    request: Optional[StartPresentationRequest] = None,
    current_user: AuthenticatedUser = Depends(get_current_user),
    db=Depends(get_db)
    ):
    """
    Start presentation generation for a specific p_id.
    """
    try:
        user_id = current_user.user_id
        try:
            manager = await get_manager()
        except Exception as e:
            logger.error(f"❌ Failed to get Socket.IO manager: {e}")
            raise HTTPException(status_code=503, detail="Real-time service is temporarily unavailable. Please try again later.")
        
        jr = await run_in_threadpool(db.job_requests.find_one, {"p_id": p_id})
        if not jr or jr.get("user_id") != user_id:
            raise HTTPException(status_code=404 if not jr else 403, detail="Presentation not found" if not jr else "Access denied")
        
        pres = await run_in_threadpool(db.presentations.find_one, {"p_id": p_id})
        if not pres: raise HTTPException(status_code=404, detail="Presentation not found")
        
        status = pres.get("status")
        if status == "processing": raise HTTPException(status_code=409, detail="Presentation already processing")
        if status in ("completed", "failed"): raise HTTPException(status_code=409, detail=f"Presentation already {status}")
        
        user_message = pres.get("message", f"Generate presentation for p_id: {p_id}")
        
        # Normalize file_urls
        file_urls = []
        source_urls = request.file_urls if request and request.file_urls else pres.get("file_urls", [])
        for item in source_urls:
            if isinstance(item, str): file_urls.append({"name": item, "url": item})
            elif isinstance(item, dict): file_urls.append({"name": item.get("name", item.get("url", "")), "url": item.get("url", "")})
        
        if file_urls:
            await run_in_threadpool(db.presentations.update_one, {"p_id": p_id}, {"$set": {"file_urls": file_urls, "updated_at": get_utc_timestamp_iso()}})
        
        # Token check
        server_api_key = os.getenv("SERVER_API_KEY")
        feature_endpoint_id = os.getenv("FEATURE_ENDPOINT_ID")
        if server_api_key and feature_endpoint_id:
            try:
                payload = {"user_id": user_id, "feature_endpoint_id": feature_endpoint_id}
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.post(f"{token_api_base_url}/api/token-process/start", json=payload, headers={"x-server-api-key": server_api_key})
                    result = response.json()
                    if result.get("success"):
                        data = result.get("data", {})
                        usage_key = data.get("usage_key")
                        if usage_key:
                            db.presentations.update_one({"p_id": p_id}, {"$set": {"usage_key": usage_key}})
                            
                        if data.get("status") != "accessible":
                            return JSONResponse(status_code=403, content={"error": "ACCESS_DENIED", "message": result.get("data", {}).get("message")})
            except Exception as e:
                logger.error(f"Token check failed: {e}")
        
        asyncio.create_task(execute_agent_for_presentation(p_id, user_id, manager, user_message))
        return {"message": "Presentation generation started", "p_id": p_id, "status": "processing", "file_urls": file_urls}
    except HTTPException: raise
    except Exception as e:
        logger.error(f"Error starting presentation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/presentation/{p_id}/status")
async def get_presentation_status(
    p_id: str,
    current_user: AuthenticatedUser = Depends(get_current_user),
    db=Depends(get_db)
    ):
    """Get presentation status"""
    presentation = await run_in_threadpool(db.presentations.find_one, {"p_id": p_id})
    if not presentation: raise HTTPException(status_code=404, detail="Presentation not found")
    if presentation.get("user_id") != current_user.user_id: raise HTTPException(status_code=403, detail="Access denied")
    
    return {
        "p_id": p_id, "status": presentation.get("status", "unknown"),
        "title": presentation.get("title", ""), "total_slides": presentation.get("total_slides", 0),
        "created_at": presentation.get("creation_date"), "updated_at": presentation.get("updated_at"),
        "completed_at": presentation.get("completion_date"), "duration_display": presentation.get("duration_display")
    }

@router.get("/presentation/{p_id}/data")
async def get_presentation_data(
    p_id: str,
    current_user: AuthenticatedUser = Depends(get_current_user),
    db=Depends(get_db)
    ):
    """Get complete presentation data"""
    presentation = await run_in_threadpool(db.presentations.find_one, {"p_id": p_id})
    if not presentation or presentation.get("user_id") != current_user.user_id:
        raise HTTPException(status_code=404 if not presentation else 403, detail="Not found" if not presentation else "Denied")
    
    slides = await run_in_threadpool(lambda: list(db.slides.find({"p_id": p_id}).sort("slide_number", 1)))
    formatted_slides = [{
        "slide_number": s.get("slide_number"), "title": s.get("title", ""),
        "content": s.get("content", ""), "slide_html": s.get("slide_html", ""),
        "created_at": convert_datetime_to_iso(s.get("created_at")),
        "updated_at": convert_datetime_to_iso(s.get("updated_at"))
    } for s in slides]
    
    return {
        "p_id": p_id, "title": presentation.get("title", ""), "status": presentation.get("status", "unknown"),
        "total_slides": presentation.get("total_slides", 0), "slides": formatted_slides,
        "created_at": convert_datetime_to_iso(presentation.get("creation_date")),
        "updated_at": convert_datetime_to_iso(presentation.get("updated_at")),
        "completed_at": convert_datetime_to_iso(presentation.get("completion_date"))
    }

# @router.get("/debug/workers")
# async def debug_workers(current_user: AuthenticatedUser = Depends(get_current_user)):
#     manager = await get_manager()
#     return {"workers": [manager.get_worker_info()], "total_workers": 1, "timestamp": get_utc_timestamp_iso()}

# @router.get("/debug/connections")
# async def debug_connections(current_user: AuthenticatedUser = Depends(get_current_user)):
#     manager = await get_manager()
#     return {
#         "active_connections": manager.active_connections,
#         "user_presentations": {k: list(v) for k, v in manager.user_presentations.items()},
#         "session_presentations": manager.session_presentations,
#     }

from core.database import clone_presentation_transactional

# @router.post("/replica")
# async def clone_presentation(
#     request: ClonePresentationRequest,
#     current_user: AuthenticatedUser = Depends(get_current_user),
#     db=Depends(get_db)
#     ):
#     """Clone a shared presentation"""
#     original_p_id = request.original_p_id
#     new_user_id = current_user.user_id
#     new_p_id = str(ObjectId())
    
#     try:
#         # Auditor Recommendation: Use atomic transactions for cloning
#         await run_in_threadpool(
#             clone_presentation_transactional,
#             db, original_p_id, new_p_id, new_user_id, 
#             current_user.email, current_user.is_verified, current_user.package
#         )
        
#         try:
#             db_url = os.getenv("DATABASE_URL")
#             session_service = DatabaseSessionService(db_url=format_db_url_with_ssl(db_url))
#             await session_service.create_session(app_name=APP_NAME, user_id=new_user_id, session_id=new_p_id, state={"p_id": new_p_id, "user_id": new_user_id, "cloned_from": original_p_id})
#         except Exception as e: logger.error(f"Cloned session failed: {e}")
        
#         return {"status": "success", "new_p_id": new_p_id}
#     except Exception as e:
#         logger.error(f"Clone failed: {e}")
#         raise HTTPException(status_code=500, detail=str(e))
