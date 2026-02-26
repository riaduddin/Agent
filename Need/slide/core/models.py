from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class TokenCounts(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    thoughts_tokens: int = 0

class PresentationModel(BaseModel):
    p_id: str
    user_id: str
    title: Optional[str] = None
    status: str = "queued"
    creation_date: str
    updated_at: str
    completion_date: Optional[str] = None
    file_urls: List[Dict[str, str]] = Field(default_factory=list)
    token_counts: TokenCounts = Field(default_factory=TokenCounts)
    total_slides: int = 0
    duration_seconds: float = 0.0
    duration_display: Optional[str] = None

class SlideModel(BaseModel):
    p_id: str
    slide_number: int
    title: str = ""
    content: str = ""
    slide_html: str = ""
    status: str = "pending"

class AgentOutputModel(BaseModel):
    p_id: str
    user_id: str
    role: str
    author: str
    content_type: str
    timestamp: str
    user_message: Optional[str] = None
    thinking: Optional[str] = None
    html_content: Optional[str] = None
    json_data: Optional[Dict[str, Any]] = None
    parsed_output: Optional[str] = None
