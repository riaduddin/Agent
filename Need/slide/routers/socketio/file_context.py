import asyncio
import os
import logging
import mimetypes
import httpx
import time as _time
from typing import Optional, List
from fastapi.concurrency import run_in_threadpool
from google import genai
from google.genai import types
from google.adk.sessions import DatabaseSessionService
from google.adk.events import Event, EventActions

from .utils import format_db_url_with_ssl, APP_NAME, safe_append_event

logger = logging.getLogger(__name__)

async def extract_text_from_url(file_url_or_obj, p_id: Optional[str] = None, user_id: Optional[str] = None) -> str:
    """Downloads a PDF, DOCX, or TXT file and extracts clean content for Gemini.
    
    Accepts either:
    - A string URL (legacy format)
    - A dict with 'url' and optionally 'name' keys (new format)
    
    If p_id and user_id are provided, token counts from LLM calls will be tracked in session state.
    """
    try:
        # Handle both string URLs and object format
        if isinstance(file_url_or_obj, dict):
            file_url = file_url_or_obj.get("url", "")
            file_name = file_url_or_obj.get("name", "")
        else:
            file_url = str(file_url_or_obj)
            file_name = ""
        
        if not file_url:
            logger.warning(f"Empty file URL provided")
            return ""
        
        # Use limits to prevent connection pool exhaustion
        limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)
        async with httpx.AsyncClient(timeout=60, limits=limits) as client:
            response = await client.get(file_url)
            response.raise_for_status()
            file_bytes = response.content
            content_type = response.headers.get("content-type")
        
        # Infer file type
        file_extension = file_url.split(".")[-1].lower()
        mime_type = content_type or mimetypes.guess_type(file_url)[0] or ""
        
        # === TXT === (No genai client needed)
        if "text" in mime_type or file_extension == "txt":
            return file_bytes.decode("utf-8", errors="ignore")

        # === PDF & DOCX === (Need genai client)
        genai_client = genai.Client()
        try:
            # === PDF ===
            if "pdf" in mime_type or file_extension == "pdf":
                parts = [
                    types.Part.from_bytes(data=file_bytes, mime_type="application/pdf"),
                    "Extract all text content from this PDF file. Preserve the document's structure, headings, paragraphs, and formatting. Include all text, tables, lists, and any readable content. Do not modify or interpret the content - just extract it as-is."
                ]
                # Auditor Recommendation: Wrap blocking SDK call
                result = await run_in_threadpool(genai_client.models.generate_content,
                    model="gemini-2.5-flash",
                    contents=parts,
                )
                
                # Extract and store token counts if p_id and user_id are provided
                if p_id and user_id:
                    try:
                        if hasattr(result, 'usage_metadata') and result.usage_metadata:
                            usage = result.usage_metadata
                            input_tokens = getattr(usage, 'prompt_token_count', 0) or 0
                            output_tokens = getattr(usage, 'candidates_token_count', 0) or 0
                            thoughts_tokens = getattr(usage, 'thoughts_token_count', 0) or 0
                            output_tokens += thoughts_tokens  # Add thoughts to output
                            
                            # Store tokens in session state
                            if input_tokens > 0 or output_tokens > 0:
                                db_url = os.getenv("DATABASE_URL")
                                if db_url:
                                    db_url_formatted = format_db_url_with_ssl(db_url)
                                    session_service = DatabaseSessionService(db_url=db_url_formatted)
                                    session = await asyncio.wait_for(
                                        session_service.get_session(app_name=APP_NAME, session_id=p_id, user_id=user_id),
                                        timeout=10.0
                                    )
                                    if session and hasattr(session, 'state'):
                                        if 'token_counts' not in session.state:
                                            session.state['token_counts'] = {
                                                'input_tokens': 0,
                                                'output_tokens': 0,
                                                'thoughts_tokens': 0
                                            }
                                        
                                        # Safely update counts ensuring we don't add to None
                                        tc = session.state['token_counts']
                                        tc['input_tokens'] = (tc.get('input_tokens') or 0) + input_tokens
                                        tc['output_tokens'] = (tc.get('output_tokens') or 0) + output_tokens
                                        tc['thoughts_tokens'] = (tc.get('thoughts_tokens') or 0) + thoughts_tokens
                                        
                                        # Persist to database
                                        actions = EventActions(state_delta={"token_counts": session.state['token_counts']})
                                        system_event = Event(
                                            invocation_id=f"file_extract_tokens_{p_id}_{int(_time.time()*1000)}",
                                            author="system",
                                            actions=actions,
                                            timestamp=_time.time(),
                                        )
                                        if await safe_append_event(session_service, p_id, user_id, system_event):
                                            logger.info(f"📊 File extraction (PDF) added {input_tokens} input and {output_tokens} output tokens")
                                        else:
                                            logger.warning(f"⚠️ Failed to persist PDF extraction tokens to session")
                    except Exception as token_error:
                        logger.warning(f"⚠️ Could not track tokens for PDF extraction: {token_error}")
                
                return result.text or ""

            # === DOCX ===
            elif "word" in mime_type or file_extension in ["docx", "doc"]:
                parts = [
                    types.Part.from_bytes(data=file_bytes, mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
                    "Extract all text content from this DOCX file. Preserve the document's structure, headings, paragraphs, and formatting. Include all text, tables, lists, and any readable content. Do not modify or interpret the content - just extract it as-is."
                ]
                # Auditor Recommendation: Wrap blocking SDK call
                result = await run_in_threadpool(genai_client.models.generate_content,
                    model="gemini-2.5-flash",
                    contents=parts,
                )
                
                # Extract and store token counts if p_id and user_id are provided
                if p_id and user_id:
                    try:
                        if hasattr(result, 'usage_metadata') and result.usage_metadata:
                            usage = result.usage_metadata
                            input_tokens = getattr(usage, 'prompt_token_count', 0) or 0
                            output_tokens = getattr(usage, 'candidates_token_count', 0) or 0
                            thoughts_tokens = getattr(usage, 'thoughts_token_count', 0) or 0
                            output_tokens += thoughts_tokens  # Add thoughts to output
                            
                            # Store tokens in session state
                            if input_tokens > 0 or output_tokens > 0:
                                db_url = os.getenv("DATABASE_URL")
                                if db_url:
                                    db_url_formatted = format_db_url_with_ssl(db_url)
                                    session_service = DatabaseSessionService(db_url=db_url_formatted)
                                    session = await asyncio.wait_for(
                                        session_service.get_session(app_name=APP_NAME, session_id=p_id, user_id=user_id),
                                        timeout=10.0
                                    )
                                    if session and hasattr(session, 'state'):
                                        if 'token_counts' not in session.state:
                                            session.state['token_counts'] = {
                                                'input_tokens': 0,
                                                'output_tokens': 0,
                                                'thoughts_tokens': 0
                                            }
                                        
                                        # Safely update counts ensuring we don't add to None
                                        tc = session.state['token_counts']
                                        tc['input_tokens'] = (tc.get('input_tokens') or 0) + input_tokens
                                        tc['output_tokens'] = (tc.get('output_tokens') or 0) + output_tokens
                                        tc['thoughts_tokens'] = (tc.get('thoughts_tokens') or 0) + thoughts_tokens
                                        
                                        # Persist to database
                                        actions = EventActions(state_delta={"token_counts": session.state['token_counts']})
                                        system_event = Event(
                                            invocation_id=f"file_extract_tokens_{p_id}_{int(_time.time()*1000)}",
                                            author="system",
                                            actions=actions,
                                            timestamp=_time.time(),
                                        )
                                        if await safe_append_event(session_service, p_id, user_id, system_event):
                                            logger.info(f"📊 File extraction (DOCX) added {input_tokens} input and {output_tokens} output tokens")
                                        else:
                                            logger.warning(f"⚠️ Failed to persist DOCX extraction tokens to session")
                    except Exception as token_error:
                        logger.warning(f"⚠️ Could not track tokens for DOCX extraction: {token_error}")
                
                return result.text or ""
            
            else:
                logger.warning(f"Unsupported file type: {mime_type} / {file_extension}")
                return ""
        finally:
            # Always close the client
            try:
                genai_client.close()
            except Exception:
                pass

    except Exception as e:
        logger.warning(f"Failed to extract text from {file_url}: {e}")
        return ""

async def build_file_context(urls: Optional[List], p_id: Optional[str] = None, user_id: Optional[str] = None) -> str:
    """Build file context from URLs.
    """
    if not urls:
        return ""
    tasks = [extract_text_from_url(u, p_id=p_id, user_id=user_id) for u in urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    ok = []
    for r in results:
        if isinstance(r, Exception):
            logger.warning(f"[file_context] one file failed: {r}")
        elif isinstance(r, str) and r.strip():
            ok.append(r)
    return "\n\n".join(ok)
