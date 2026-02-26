import os
import uuid
import mimetypes
from datetime import datetime, timedelta, timezone
from typing import List
from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, Form
from pydantic import BaseModel
from google.cloud import storage
from fastapi.concurrency import run_in_threadpool
from middleware.auth import get_current_user, AuthenticatedUser

router = APIRouter()

GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "shothik-ai-presentations")

class UploadResponseItem(BaseModel):
    filename: str
    public_url: str
    signed_url: str
    object_name: str

class UploadResponse(BaseModel):
    uploads: List[UploadResponseItem]

def _sync_upload_blob(bucket_name, object_name, content, content_type):
    """
    Sync GCS upload to be wrapped in run_in_threadpool.
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(object_name)
    blob.upload_from_string(content, content_type=content_type)
    
    # 1 hour expiration per audit requirement
    signed_url = blob.generate_signed_url(
        version="v4",
        expiration=timedelta(hours=1),
        method="GET",
    )
    return blob.public_url, signed_url

@router.post("/upload-file", response_model=UploadResponse)
async def upload_file(
    files: List[UploadFile] = File(...),
    current_user: AuthenticatedUser = Depends(get_current_user)
    ):
    """Upload files to GCS."""
    user_id = current_user.user_id
    
    async def process_one(file: UploadFile):
        content = await file.read()
        mime = file.content_type or mimetypes.guess_type(file.filename)[0] or "application/octet-stream"
        ext = os.path.splitext(file.filename)[1]
        obj_name = f"{user_id}/{uuid.uuid4().hex}{ext}"

        # Non-blocking upload
        public_url, signed_url = await run_in_threadpool(
            _sync_upload_blob,
            GCS_BUCKET_NAME,
            obj_name,
            content,
            mime,
        )
        return UploadResponseItem(
            filename=file.filename,
            public_url=public_url,
            signed_url=signed_url,
            object_name=obj_name,
        )

    results = await asyncio.gather(*(process_one(f) for f in files), return_exceptions=True)
    uploads = []
    for r in results:
        if isinstance(r, Exception):
            raise HTTPException(status_code=500, detail=str(r))
        uploads.append(r)

    return UploadResponse(uploads=uploads)
