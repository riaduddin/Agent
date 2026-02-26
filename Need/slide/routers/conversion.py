import asyncio
import json
import logging
import os
import uuid
import mimetypes
import requests
from datetime import timedelta
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Request
from fastapi.responses import StreamingResponse, Response
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, validator
from sse_starlette.sse import EventSourceResponse
from google.cloud import storage

from core.conversion.node_bridge import NodeConversionBridge
from core.conversion.job_manager import JobManager
from middleware.auth import get_current_user, AuthenticatedUser

# Setup Router (No prefix, as it will be mounted at /api)
router = APIRouter(tags=["conversion"])
logger = logging.getLogger(__name__)

# Constants
MAX_SLIDES = 100
MAX_SLIDE_SIZE = 1000000  # 1MB
MAX_TOTAL_SIZE = 10000000 # 10MB
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "shothik-ai-presentations")

# --- Pydantic Models ---

class Slide(BaseModel):
    html_content: str
    slide_number: Optional[int] = None

class ConversionRequest(BaseModel):
    slides: List[Slide]
    format: str = Field(..., pattern="^(pdf|pptx)$")

class GoogleSlidesConversionRequest(BaseModel):
    slides: List[Slide]
    google_token: str

    @validator('slides')
    def validate_slides(cls, v):
        if len(v) > MAX_SLIDES:
            raise ValueError(f"Max {MAX_SLIDES} slides allowed")
        if not v:
            raise ValueError("At least one slide is required")
            
        total_size = 0
        for i, slide in enumerate(v):
            size = len(slide.html_content.encode('utf-8'))
            if size > MAX_SLIDE_SIZE:
                 raise ValueError(f"Slide {i+1} exceeds size limit of 1MB")
            total_size += size
            
        if total_size > MAX_TOTAL_SIZE:
             raise ValueError(f"Total request size {total_size} exceeds limit of 10MB")
        return v

# --- Helper Functions ---

def _sync_upload_blob(bucket_name, object_name, content, content_type):
    """Sync GCS upload helper"""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(object_name)
        blob.upload_from_string(content, content_type=content_type)
        
        # generate_signed_url returns a string
        signed_url = blob.generate_signed_url(
            version="v4",
            expiration=timedelta(hours=24),
            method="GET",
        )
        return blob.public_url, signed_url
    except Exception as e:
        logger.error(f"GCS Upload failed: {e}")
        raise

async def _upload_to_google_drive(google_token: str, content: bytes, filename: str):
    """Upload PPTX to Google Drive and convert to Google Slides"""
    try:
        url = "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart"
        
        # Prepare the multipart/related request body
        boundary = "-------314159265358979323846"
        
        metadata = {
            "name": filename,
            "mimeType": "application/vnd.google-apps.presentation"
        }
        
        # Build the body manually to ensure multipart/related format
        body = []
        body.append(f"--{boundary}".encode('utf-8'))
        body.append(b"Content-Type: application/json; charset=UTF-8")
        body.append(b"")
        body.append(json.dumps(metadata).encode('utf-8'))
        
        body.append(f"--{boundary}".encode('utf-8'))
        body.append(b"Content-Type: application/vnd.openxmlformats-officedocument.presentationml.presentation")
        body.append(b"")
        body.append(content)
        body.append(f"--{boundary}--".encode('utf-8'))
        
        data_body = b"\r\n".join(body)
        
        headers = {
            "Authorization": f"Bearer {google_token}",
            "Content-Type": f"multipart/related; boundary={boundary}",
            "Content-Length": str(len(data_body))
        }
        
        logger.info(f"📤 Uploading {len(content)} bytes to Google Drive as '{filename}'")
        
        response = await run_in_threadpool(
            requests.post,
            url,
            headers=headers,
            data=data_body
        )
        
        if response.status_code != 200:
            logger.error(f"❌ Google Drive upload failed ({response.status_code}): {response.text}")
            response.raise_for_status()
            
        result = response.json()
        file_id = result.get("id")
        
        # Get the webViewLink for the client to open
        get_url = f"https://www.googleapis.com/drive/v3/files/{file_id}?fields=webViewLink"
        get_response = await run_in_threadpool(
            requests.get,
            get_url,
            headers={"Authorization": f"Bearer {google_token}"}
        )
        get_response.raise_for_status()
        
        return get_response.json().get("webViewLink")
        
    except Exception as e:
        logger.error(f"❌ Google Drive upload error: {e}")
        raise

async def process_conversion_task(job_id: str, slides: List[dict], format: str, user_id: str):
    """Background task to run conversion and update Redis"""
    try:
        logger.info(f"Processing job {job_id} for user {user_id}")
        JobManager.update_progress(job_id, 10, 0, "Initializing conversion environment...")

        # Define progress callback with cancellation check
        async def on_progress(percent: int, message: str):
            # Check for cancellation during conversion
            if JobManager.is_cancelled(job_id):
                logger.info(f"Job {job_id} cancellation requested during progress update")
                raise asyncio.CancelledError(f"Job {job_id} has been cancelled")
            JobManager.update_progress(job_id, percent, 0, message)

        # 1. Run Conversion
        file_content = await NodeConversionBridge.convert_slides(
            slides=slides,
            format=format,
            job_id=job_id,
            progress_callback=on_progress
        )
        
        # Check for cancellation after conversion completes
        if JobManager.is_cancelled(job_id):
            logger.info(f"Job {job_id} cancellation requested after conversion")
            JobManager.fail_job(job_id, "Cancelled")
            return
        
        JobManager.update_progress(job_id, 90, 0, "Uploading result...")

        # Check for cancellation before uploading to GCS
        if JobManager.is_cancelled(job_id):
            logger.info(f"Job {job_id} cancellation requested before upload")
            JobManager.fail_job(job_id, "Cancelled")
            return

        # 2. Upload to GCS
        ext = format
        mime = "application/pdf" if format == "pdf" else "application/vnd.openxmlformats-officedocument.presentationml.presentation"
        object_name = f"{user_id}/{job_id}.{ext}"

        logger.info(f"Uploading output to GCS: {object_name} ({len(file_content)} bytes)")
        
        # Upload
        public_url, signed_url = await run_in_threadpool(
            _sync_upload_blob,
            GCS_BUCKET_NAME,
            object_name,
            file_content,
            mime
        )
        logger.info(f"GCS Upload complete. Signed URL: {signed_url[:20]}...")

        result = {
            "downloadUrl": signed_url, # Use signed URL for secure download
            "publicUrl": public_url,
            "filename": f"presentation.{ext}",
            "size": len(file_content)
        }

        JobManager.complete_job(job_id, result)
        logger.info(f"Job {job_id} completed successfully and marked in Redis")

    except asyncio.CancelledError:
        logger.info(f"Job {job_id} was cancelled")
        JobManager.fail_job(job_id, "Cancelled")
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)
        JobManager.fail_job(job_id, str(e))

async def process_google_slides_conversion_task(job_id: str, slides: List[dict], google_token: str, user_id: str):
    """Background task to run conversion and upload to Google Drive"""
    try:
        logger.info(f"Processing Google Slides job {job_id} for user {user_id}")
        JobManager.update_progress(job_id, 10, 0, "Initializing conversion environment...")

        async def on_progress(percent: int, message: str):
            if JobManager.is_cancelled(job_id):
                 raise asyncio.CancelledError(f"Job {job_id} has been cancelled")
            JobManager.update_progress(job_id, percent, 0, message)

        # 1. Run Conversion (always PPTX for Google Slides)
        file_content = await NodeConversionBridge.convert_slides(
            slides=slides,
            format="pptx",
            job_id=job_id,
            progress_callback=on_progress
        )
        
        if JobManager.is_cancelled(job_id):
            JobManager.fail_job(job_id, "Cancelled")
            return
            
        JobManager.update_progress(job_id, 90, 0, "Uploading to Google Drive...")

        # 2. Upload to Google Drive
        webViewLink = await _upload_to_google_drive(
            google_token,
            file_content,
            f"Presentation_{job_id}.pptx"
        )
        
        result = {
            "downloadUrl": webViewLink,
            "webViewLink": webViewLink,
            "filename": f"Presentation_{job_id}",
            "isGoogleSlides": True
        }

        JobManager.complete_job(job_id, result)
        logger.info(f"Google Slides Job {job_id} completed successfully")

    except Exception as e:
        logger.error(f"Google Slides Job {job_id} failed: {e}", exc_info=True)
        JobManager.fail_job(job_id, str(e))

# --- API Endpoints ---

@router.post("/convert", status_code=202)
async def create_conversion_job(
    request: ConversionRequest,
    background_tasks: BackgroundTasks,
    current_user: AuthenticatedUser = Depends(get_current_user)
):
    """
    Create a new conversion job (Async).
    Returns a Job ID immediately.
    """
    # Validate that user is authenticated with a valid user_id
    user_id = current_user.user_id
    if not user_id:
        raise HTTPException(status_code=401, detail="Authenticated user must have a valid user ID")
    
    job_id = str(uuid.uuid4())
    
    # Create Job in Redis
    JobManager.create_job(job_id, request.format, user_id)
    
    # Serialize slides for the background task
    slides_data = [slide.model_dump() for slide in request.slides]
    
    # Start Background Task
    background_tasks.add_task(
        process_conversion_task,
        job_id,
        slides_data,
        request.format,
        user_id
    )
    
    return {
        "jobId": job_id,
        "format": request.format,
        "status": "queued",
        "message": "Conversion job queued",
        "checkStatusUrl": f"/job/{job_id}",
        "streamUrl": f"/job/{job_id}/stream"
    }

@router.post("/convert/google-slides", status_code=202)
async def create_google_slides_job(
    request: GoogleSlidesConversionRequest,
    background_tasks: BackgroundTasks,
    current_user: AuthenticatedUser = Depends(get_current_user)
):
    """
    Create a new Google Slides conversion job.
    """
    user_id = current_user.user_id
    job_id = str(uuid.uuid4())
    
    JobManager.create_job(job_id, "google-slides", user_id)
    
    slides_data = [slide.model_dump() for slide in request.slides]
    
    background_tasks.add_task(
        process_google_slides_conversion_task,
        job_id,
        slides_data,
        request.google_token,
        user_id
    )
    
    return {
        "jobId": job_id,
        "status": "queued",
        "checkStatusUrl": f"/job/{job_id}",
        "streamUrl": f"/job/{job_id}/stream"
    }

@router.get("/job/{job_id}")
async def get_job_status(job_id: str, current_user: AuthenticatedUser = Depends(get_current_user)):
    """Get the current status of a job"""
    job = JobManager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Check ownership - user can only access their own jobs
    job_user_id = job.get('userId')
    current_user_id = current_user.user_id
    
    if not job_user_id or job_user_id != current_user_id:
        raise HTTPException(status_code=403, detail="Unauthorized")
        
    response = {
        "jobId": job_id,
        "format": job.get("format"),
        "status": job.get("status"),
        "progress": job.get("progress"),
        "message": job.get("message"),
        "createdAt": job.get("createdAt"),
    }
    
    if job.get("status") == "completed":
        response["result"] = job.get("result") # This is already a dict if parsed by JobManager
        
    if job.get("status") == "failed":
        response["error"] = job.get("error")
        
    return response

@router.get("/job/{job_id}/stream")
async def stream_job_progress(job_id: str, current_user: AuthenticatedUser = Depends(get_current_user), request: Request = None):
    """
    Server-Sent Events (SSE) for job progress.
    """
    # Check ownership upfront
    initial_job = JobManager.get_job(job_id)
    if not initial_job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_user_id = initial_job.get('userId')
    current_user_id = current_user.user_id
    
    if not job_user_id or job_user_id != current_user_id:
        raise HTTPException(status_code=403, detail="Unauthorized")
    
    async def event_generator():
        last_status = None
        last_progress = -1
        
        while True:
            # Check for client disconnect
            if request and await request.is_disconnected():
                break

            job = JobManager.get_job(job_id)
            if not job:
                yield {"event": "error", "data": json.dumps({"error": "Job not found"})}
                break
            
            # Send update if changed
            if job["status"] != last_status or job["progress"] != last_progress:
                payload = {
                     "jobId": job_id,
                     "status": job["status"],
                     "progress": job["progress"],
                     "message": job["message"]
                }
                if job["status"] == "completed":
                    payload["result"] = job.get("result")
                elif job["status"] == "failed":
                    payload["error"] = job.get("error")
                
                yield {"data": json.dumps(payload)}
                
                last_status = job["status"]
                last_progress = job["progress"]

            if job["status"] in ["completed", "failed", "cancelled"]:
                break
                
            await asyncio.sleep(1) # Poll every 1 second
            
    return EventSourceResponse(event_generator())

@router.delete("/job/{job_id}")
async def cancel_job(job_id: str, current_user: AuthenticatedUser = Depends(get_current_user)):
    """Cancel a running job"""
    job = JobManager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Check ownership - user can only cancel their own jobs
    job_owner_id = job.get("user_id")
    current_user_id = current_user.user_id
    
    if job_owner_id != current_user_id:
        raise HTTPException(status_code=403, detail="Not authorized to cancel this job")
        
    # Only cancel if queued or processing (actually we can't easily kill the subprocess yet)
    # But we can mark it as cancelled so the UI stops.
    # Killing the subprocess would require storing the PID or using a more complex manager.
    # For now, we update Redis status.
    
    if job["status"] in ["queued", "processing"]:
        JobManager.cancel_job(job_id)
        return {"message": "Job cancelled"}
        
    return {"message": "Job cannot be cancelled (already finished)"}
