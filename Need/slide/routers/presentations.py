import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from core.database import get_db, get_presentations_by_user, get_slides_by_p_id, get_agent_logs_by_p_id
from middleware.auth import get_current_user, AuthenticatedUser
import json
import asyncio
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)

class SaveSlideMetadata(BaseModel):
    version: Optional[int] = None
    lastEdited: Optional[str] = None
    editedBy: Optional[str] = None

class SaveSlideRequest(BaseModel):
    slideIndex: int
    presentationId: str
    htmlContent: str
    metadata: Optional[SaveSlideMetadata] = None

class SaveSlideResponse(BaseModel):
    success: bool
    p_id: str
    version: int
    savedAt: str
    conflict: Optional[bool] = None
    error: Optional[str] = None

router = APIRouter()

@router.get("/presentations")
async def list_presentations(
    current_user: AuthenticatedUser = Depends(get_current_user),
    db=Depends(get_db)
    ):
    """List presentations for the authenticated user."""
    return await run_in_threadpool(get_presentations_by_user, db, current_user.user_id)

@router.get("/presentations/{p_id}")
async def get_presentation(
    p_id: str, 
    current_user: AuthenticatedUser = Depends(get_current_user),
    db=Depends(get_db)
    ):
    """Get slides for a specific presentation."""
    # Verify user owns this presentation
    pres = await run_in_threadpool(db.presentations.find_one, {"p_id": p_id})
    if not pres:
        raise HTTPException(status_code=404, detail="Presentation not found")
    
    if pres.get("user_id") != current_user.user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return await run_in_threadpool(get_slides_by_p_id, db, p_id)

@router.get("/presentation-status/{p_id}")
async def get_presentation_status(
    p_id: str, 
    current_user: AuthenticatedUser = Depends(get_current_user),
    db=Depends(get_db)
    ):
    """Get specific status fields for a presentation."""
    # Verify user owns this presentation
    pres = await run_in_threadpool(db.presentations.find_one, {"p_id": p_id}, {"status": 1, "title": 1, "total_slides": 1, "user_id": 1})
    if not pres:
        raise HTTPException(status_code=404, detail="Presentation not found")
    
    if pres.get("user_id") != current_user.user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return {
        "p_id": p_id,
        "status": pres.get("status", "unknown")
    }

@router.get("/logs")
async def get_agent_logs(
    p_id: str, 
    current_user: AuthenticatedUser = Depends(get_current_user),
    db=Depends(get_db)
    ):
    """Get agent logs for a specific presentation."""
    # Verify user owns this presentation
    pres = await run_in_threadpool(db.presentations.find_one, {"p_id": p_id})
    if not pres:
        raise HTTPException(status_code=404, detail="Presentation not found")
    
    if pres.get("user_id") != current_user.user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return await run_in_threadpool(get_agent_logs_by_p_id, db, p_id)

# @router.get("/simulation_slides/{p_id}")
# async def stream_slides(
#     p_id: str, 
#     current_user: AuthenticatedUser = Depends(get_current_user),
#     db=Depends(get_db)
#     ):
#     """Stream slides with authentication."""
#     pres = await run_in_threadpool(db.presentations.find_one, {"p_id": p_id})
#     if not pres:
#         raise HTTPException(status_code=404, detail="Presentation not found")
    
#     if pres.get("user_id") != current_user.user_id:
#         raise HTTPException(status_code=403, detail="Access denied")
    
#     data = await run_in_threadpool(get_slides_by_p_id, db, p_id)
#     slides = data.get("slides", [])
#     title = data.get("title", "")
#     total_slides = data.get("total_slides", 0)

#     async def gen():
#         await asyncio.sleep(3) # Reduced from 30s for better DX
#         for s in slides:
#             payload = {
#                 "status": "processing",
#                 "title": title,
#                 "total_slides": total_slides,
#                 "slides": s
#             }
#             yield json.dumps(payload) + "\n"
#             await asyncio.sleep(1)
#         yield json.dumps({"status": "completed"}) + "\n"

#     return StreamingResponse(
#         gen(),
#         media_type="application/x-ndjson",
#         headers={"X-Accel-Buffering": "no"}
#     )

@router.put("/slides/save", response_model=SaveSlideResponse)
async def save_slide(
    request: SaveSlideRequest,
    current_user: AuthenticatedUser = Depends(get_current_user),
    db=Depends(get_db)
    ):
    """
    Save/update slide HTML content in the database.
    Requires authentication and verifies user owns the presentation.
    """
    try:
        p_id = request.presentationId
        slide_index = request.slideIndex
        html_content = request.htmlContent
        
        # Validate slide index (must be non-negative)
        if slide_index < 0:
            return SaveSlideResponse(
                success=False,
                p_id=p_id,
                version=0,
                savedAt=datetime.now(timezone.utc).isoformat(),
                error=f"Invalid slideIndex: {slide_index}. Slide index must be 0 or greater."
            )
        
        # Verify user owns this presentation
        pres = await run_in_threadpool(db.presentations.find_one, {"p_id": p_id})
        if not pres:
            return SaveSlideResponse(
                success=False,
                p_id=p_id,
                version=0,
                savedAt=datetime.now(timezone.utc).isoformat(),
                error="Presentation not found"
            )
        
        if pres.get("user_id") != current_user.user_id:
            return SaveSlideResponse(
                success=False,
                p_id=p_id,
                version=0,
                savedAt=datetime.now(timezone.utc).isoformat(),
                error="You don't have permission to access this presentation"
            )
        
        # Find the slide document
        slide_doc = await run_in_threadpool(db.slide_html.find_one, {"p_id": p_id, "slide_index": slide_index})
        if not slide_doc:
            return SaveSlideResponse(
                success=False,
                p_id=p_id,
                version=0,
                savedAt=datetime.now(timezone.utc).isoformat(),
                error=f"Slide with index {slide_index} not found"
            )
        
        # Verify the slide belongs to the user (additional security check)
        if slide_doc.get("user_id") != current_user.user_id:
            return SaveSlideResponse(
                success=False,
                p_id=p_id,
                version=0,
                savedAt=datetime.now(timezone.utc).isoformat(),
                error="You don't have permission to modify this slide"
            )
        
        # Get current version or start at 1
        current_version = slide_doc.get("version", 0)
        new_version = current_version + 1
        
        # Check for conflict if version is provided in metadata
        conflict = False
        if request.metadata and request.metadata.version is not None:
            if request.metadata.version != current_version:
                conflict = True
                # For now, we'll proceed with the update but flag the conflict
        
        # Update the slide document
        update_data = {
            "body": html_content,
            "updated_at": datetime.now(timezone.utc),
            "version": new_version
        }
        
        # Add metadata if provided
        if request.metadata:
            if request.metadata.lastEdited:
                update_data["last_edited"] = request.metadata.lastEdited
            if request.metadata.editedBy:
                update_data["edited_by"] = request.metadata.editedBy
        
        result = await run_in_threadpool(db.slide_html.update_one,
            {"p_id": p_id, "slide_index": slide_index},
            {"$set": update_data}
        )
        
        if result.modified_count > 0 or result.matched_count > 0:
            saved_at = datetime.now(timezone.utc).isoformat()
            return SaveSlideResponse(
                success=True,
                p_id=p_id,
                version=new_version,
                savedAt=saved_at,
                conflict=conflict if conflict else None
            )
        else:
            return SaveSlideResponse(
                success=False,
                p_id=p_id,
                version=current_version,
                savedAt=datetime.now(timezone.utc).isoformat(),
                error="Failed to update slide"
            )
            
    except Exception as e:
        logger.error(f"Error saving slide: {e}", exc_info=True)
        return SaveSlideResponse(
            success=False,
            p_id=request.presentationId if hasattr(request, 'presentationId') else "",
            version=0,
            savedAt=datetime.now(timezone.utc).isoformat(),
            error=f"Internal server error: {str(e)}"
        )
