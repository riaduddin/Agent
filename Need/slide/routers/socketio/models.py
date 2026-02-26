from pydantic import BaseModel
from typing import Optional, List

class FileUrlItem(BaseModel):
    """File URL item with name and url"""
    name: str
    url: str

class PresentationRequest(BaseModel):
    message: str
    # userId is optional - will be extracted from JWT token
    userId: Optional[str] = None
    # file_urls is optional - can be array of strings (legacy) or array of objects with name and url
    file_urls: Optional[List] = None  # Can be List[str] or List[FileUrlItem]
    # p_id is optional - if provided, will be used for edit_slide or insert_slide operations
    p_id: Optional[str] = None

class StartPresentationRequest(BaseModel):
    """Request body for starting presentation"""
    file_urls: Optional[List] = None  # Can be List[str] or List[dict] with name and url

class ClonePresentationRequest(BaseModel):
    original_p_id: str  # The p_id of the presentation to clone
