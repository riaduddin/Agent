import asyncio
import logging
import time
import os
import json
import uuid
import traceback
import aio_pika
import httpx
from typing import Optional, List
from aiohttp.client_exceptions import ClientConnectorError
from fastapi.concurrency import run_in_threadpool

from utils.feature_logger import FeatureLogger

from google import genai
from google.genai import types
from google.adk.runners import Runner
from google.adk.agents import RunConfig
from google.adk.events import Event, EventActions
from core.socketio_manager import get_manager
from core.database import get_db, APP_NAME, get_session_service
from utils.logging import log_event_to_db
from root_agent.agent import SlideOrchestrationAgent

from .utils import (
    get_utc_timestamp_iso,
    _get_context_key,
    _normalize_tool_args,
    safe_append_event,
    slide_op_manager
)
from .token_tracker import (
    add_tokens,
    get_total_tokens,
    reset_token_tracker,
    token_api_base_url
)
from .file_context import build_file_context
from .socketio_logger import log_socketio_message
from core.token_logger import log_token_usage

logger = logging.getLogger(__name__)

async def log_to_rabbitmq(log_data: dict):
    """Helper to log data to RabbitMQ feature_usage_queue"""
    try:
        rabbitmq_url = os.getenv("RABBITMQ_URL")
        if not rabbitmq_url:
            return
        
        connection = await aio_pika.connect_robust(rabbitmq_url)
        async with connection:
            channel = await connection.channel()
            await FeatureLogger.dispatch_to_queue(channel, log_data)
    except Exception as e:
        logger.warning(f"⚠️ RabbitMQ logging helper failed: {e}")

async def execute_agent_for_presentation(p_id: str, user_id: str, manager, user_message: str):
    """Execute the agent for a specific presentation"""
    MAX_RETRIES = 3
    
    # Capture start time for duration calculation
    task_start_time = time.time()
    
    try:
        # Update status to processing
        db = get_db()
        await run_in_threadpool(db.presentations.update_one, 
            {"p_id": p_id},
            {"$set": {"status": "processing", "updated_at": get_utc_timestamp_iso()}}
        )
        
        # Get worker info
        worker_id = manager._worker_id
        
        # Get presentation details for logging
        pres = await run_in_threadpool(db.presentations.find_one, {"p_id": p_id}) or {}
        usage_key = pres.get("usage_key")
        user_email = pres.get("user_email")
        
        # Broadcast started event
        started_event = {
            "type": "event",
            "event": "started",
            "p_id": p_id,
            "status": "processing",
            "worker_id": worker_id,
            "timestamp": get_utc_timestamp_iso()
        }
        log_socketio_message(started_event, p_id, "outgoing")
        await manager.broadcast_to_presentation(started_event, p_id, user_id)
        
        # Log "started" step to RabbitMQ
        await log_to_rabbitmq({
            "feature_endpoint_value": "agents-presentation-generation",
            "user_id": user_id,
            "user_email": user_email,
            "usage_key": usage_key,
            "method": "POST",
            "params": {"p_id": p_id},
            "payload": {"message": user_message},
            "response": {"status": "started"},
            "code": 100,
            "status": "success"
        })
        
        # Store user message in agent_outputs_2 collection (ONLY ONCE, before retry loop)
        user_oid = None
        try:
            # Check if user message already exists for this p_id to avoid duplicates
            existing_user_msg = await run_in_threadpool(db.agent_outputs_2.find_one, {
                "p_id": p_id,
                "author": "user",
                "user_message": user_message
            })
            
            if not existing_user_msg:
                # Fetch file_urls to persist alongside the user message if present
                try:
                    pres_for_urls = await run_in_threadpool(db.presentations.find_one, {"p_id": p_id}) or {}
                    file_urls_for_log = pres_for_urls.get("file_urls", [])
                    # Normalize file_urls: convert strings to objects if needed
                    if file_urls_for_log and len(file_urls_for_log) > 0:
                        if isinstance(file_urls_for_log[0], str):
                            file_urls_for_log = [{"name": url, "url": url} for url in file_urls_for_log]
                        elif isinstance(file_urls_for_log[0], dict):
                            file_urls_for_log = [
                                {"name": item.get("name", item.get("url", "")), "url": item.get("url", "")}
                                for item in file_urls_for_log
                            ]
                except Exception:
                    file_urls_for_log = []

                user_message_entry = {
                    "user_id": user_id,
                    "session_id": p_id,
                    "role": "user",
                    "author": "user",
                    "p_id": p_id,
                    "user_message": user_message,
                    "file_urls": file_urls_for_log if file_urls_for_log else None,
                    "content_type": "text",
                    "timestamp": get_utc_timestamp_iso()
                }
                
                # Auditor Recommendation: Wrap blocking insert
                insert_result = await run_in_threadpool(db.agent_outputs_2.insert_one, user_message_entry)
                user_oid = insert_result.inserted_id
                logger.info(f"📝 Stored user message in agent_outputs_2: {user_message[:100]}...")
                
                # Broadcast user message event
                user_event_data = {
                    "type": "chunk",
                    "author": "user",
                    "p_id": p_id,
                    "user_message": user_message,
                    "timestamp": get_utc_timestamp_iso(),
                    "file_urls": file_urls_for_log if file_urls_for_log else None,
                    "event_id": str(user_oid)
                }
                
                log_socketio_message(user_event_data, p_id, "outgoing")
                await manager.broadcast_to_presentation(user_event_data, p_id, user_id)
                logger.info(f"📤 Broadcasted user message for p_id={p_id}")
            else:
                user_oid = existing_user_msg.get("_id")
                logger.info(f"📝 User message already exists for p_id={p_id}, skipping duplicate save")
        except Exception as user_msg_error:
            logger.warning(f"⚠️ Failed to store user message: {user_msg_error}")
            # Log storage error to RabbitMQ
            await log_to_rabbitmq({
                "feature_endpoint_value": "agents-presentation-generation",
                "user_id": user_id,
                "user_email": user_email,
                "usage_key": usage_key,
                "method": "POST",
                "params": {"p_id": p_id},
                "payload": {"message": user_message},
                "response": {"error": f"Message Storage Error: {str(user_msg_error)}"},
                "code": 500,
                "status": "failed"
            })
        
        # Execute agent with retry logic
        for retry_count in range(MAX_RETRIES):
            try:
                # Use centralized session service singleton to avoid connection pool exhaustion
                session_service = get_session_service()
                
                # Check if session exists
                try:
                    session = await asyncio.wait_for(
                        session_service.get_session(app_name=APP_NAME, session_id=p_id, user_id=user_id),
                        timeout=30.0
                    )
                except Exception as e:
                    logger.error(f"❌ Session lookup failed: {e}")
                    raise
                
                # If session exists, rebuild file_context if needed
                if session:
                    try:
                        presentation = await run_in_threadpool(db.presentations.find_one, {"p_id": p_id}) or {}
                        file_urls = presentation.get("file_urls", [])
                        if file_urls:
                            # Normalize
                            if len(file_urls) > 0:
                                if isinstance(file_urls[0], str):
                                    file_urls = [{"name": url, "url": url} for url in file_urls]
                                elif isinstance(file_urls[0], dict):
                                    file_urls = [{"name": item.get("name", item.get("url", "")), "url": item.get("url", "")} for item in file_urls]
                            
                            logger.info(f"🔄 Updating session state with file_context (count={len(file_urls)})")
                            new_file_context = await build_file_context(file_urls, p_id=p_id, user_id=user_id)
                            # Qdrant store
                            try:
                                from tools.qdrant_utils import get_qdrant_manager
                                qdrant = get_qdrant_manager()
                                qdrant.store_file_context(text=new_file_context, user_id=user_id, p_id=p_id)
                            except: pass
                            
                            actions = EventActions(state_delta={"file_context": new_file_context})
                            system_event = Event(
                                invocation_id=f"update_file_context_{p_id}_{int(time.time()*1000)}",
                                author="system", actions=actions, timestamp=time.time(),
                            )
                            await safe_append_event(session_service, p_id, user_id, system_event)
                    except Exception as e:
                        logger.warning(f"⚠️ Could not update session file_context: {e}")

                if not session:
                    logger.info(f"🔍 No session found, creating new session for p_id={p_id}")
                    try:
                        presentation = await run_in_threadpool(db.presentations.find_one, {"p_id": p_id})
                        file_urls = presentation.get("file_urls", []) if presentation else []
                        if file_urls and len(file_urls) > 0:
                            if isinstance(file_urls[0], str): file_urls = [{"name": url, "url": url} for url in file_urls]
                            elif isinstance(file_urls[0], dict): file_urls = [{"name": item.get("name", item.get("url", "")), "url": item.get("url", "")} for item in file_urls]
                        
                        file_context = await build_file_context(file_urls, p_id=p_id, user_id=user_id)
                        try:
                            if file_context:
                                from tools.qdrant_utils import get_qdrant_manager
                                qdrant = get_qdrant_manager()
                                qdrant.store_file_context(text=file_context, user_id=user_id, p_id=p_id)
                        except: pass
                        
                        initial_state = {"p_id": p_id, "user_id": user_id, "file_context": file_context}
                        await asyncio.wait_for(
                            session_service.create_session(app_name=APP_NAME, user_id=user_id, session_id=p_id, state=initial_state),
                            timeout=30.0
                        )
                    except Exception as e:
                        logger.error(f"❌ Session creation failed: {e}")
                        raise
                
                runner = Runner(agent=SlideOrchestrationAgent, app_name=APP_NAME, session_service=session_service)
                user_msg = types.Content(role="user", parts=[types.Part(text=user_message)])
                
                await reset_token_tracker(p_id)
                event_count = 0
                agent_completed_successfully = False
                
                try:
                    async for event in runner.run_async(
                        user_id=user_id, session_id=p_id, new_message=user_msg, run_config=RunConfig(max_llm_calls=1000)
                    ):
                        event_count += 1
                        if hasattr(event, 'usage_metadata') and event.usage_metadata:
                            usage = event.usage_metadata
                            i_t = getattr(usage, 'prompt_token_count', 0) or 0
                            o_t = getattr(usage, 'candidates_token_count', 0) or 0
                            th_t = getattr(usage, 'thoughts_token_count', 0) or 0
                            
                            if i_t > 0 or o_t > 0:
                                # 1. Update in-memory tracker
                                await add_tokens(p_id, input_tokens=i_t, output_tokens=o_t+th_t, thoughts_tokens=th_t)
                                
                                # 2. Incrementally update Database (Persistence)
                                # await run_in_threadpool(
                                #     db.presentations.update_one,
                                #     {"p_id": p_id},
                                #     {"$inc": {
                                #         "token_counts.input_tokens": i_t,
                                #         "token_counts.output_tokens": o_t + th_t,
                                #         "token_counts.thoughts_tokens": th_t
                                #     }}
                                # )
                                
                                # 3. Log granular step usage to file (if enabled)
                                log_token_usage(p_id, event.author or "unknown", i_t, o_t + th_t, th_t)
                        
                        if event.content and event.content.parts:
                            # 🛑 Granular filtering logic
                            author = (event.author or "").lower()
                            is_template_selector = "template_selector" in author
                            # Technical agents whose tool calls (but not text progress) should be hidden
                            is_technical_agent = any(x in author for x in ["enhanced_slide_generator", "keywordresearchagent", "search_query_agent"])
                            
                            for part in event.content.parts:
                                if getattr(part, "text", None):
                                    ev_d = await process_agent_output(event, part, p_id, user_id, db, user_message, event_count == 1, usage_key=usage_key, user_email=user_email)
                                    # Always skip template_selector. Allow enhanced_gen text/HTML.
                                    if ev_d and not is_template_selector:
                                        log_socketio_message(ev_d, p_id, "outgoing")
                                        await manager.broadcast_to_presentation(ev_d, p_id, user_id)
                                        
                                elif getattr(part, "function_call", None):
                                    fc_d = await process_function_call_output(event, part, p_id, user_id, db, usage_key=usage_key, user_email=user_email)
                                    # Skip tool calls for both template_selector AND technical agents
                                    if fc_d and not is_template_selector and not is_technical_agent:
                                        log_socketio_message(fc_d, p_id, "outgoing")
                                        await manager.broadcast_to_presentation(fc_d, p_id, user_id)
                                        
                                elif getattr(part, "function_response", None):
                                    fr_d = await process_function_response_output(event, part, p_id, user_id, db, usage_key=usage_key, user_email=user_email)
                                    # Skip tool responses for both template_selector AND technical agents
                                    if fr_d and not is_template_selector and not is_technical_agent:
                                        log_socketio_message(fr_d, p_id, "outgoing")
                                        await manager.broadcast_to_presentation(fr_d, p_id, user_id)
                        
                        try:
                            # Note: log_event_to_db performs its own DB wrapping internally in latest version
                            log_event_to_db(event, db, session_id=p_id, user_id=user_id)
                        except: pass
                    
                    agent_completed_successfully = True
                except Exception as agent_error:
                    logger.error(f"❌ Agent execution error: {agent_error}")
                    # Log specific agent error to RabbitMQ
                    await log_to_rabbitmq({
                        "feature_endpoint_value": "agents-presentation-generation",
                        "user_id": user_id,
                        "user_email": user_email,
                        "usage_key": usage_key,
                        "method": "POST",
                        "params": {"p_id": p_id},
                        "payload": {"message": user_message},
                        "response": {"error": f"Agent Error: {str(agent_error)}"},
                        "code": 500,
                        "status": "failed"
                    })
                    raise
                
                if agent_completed_successfully:
                    # Token aggregation phase
                    summary = await get_total_tokens(p_id)
                    m_i, m_o, m_th = summary["input_tokens"], summary["output_tokens"], summary["thoughts_tokens"]
                    print(f"Model token input: {m_i}")
                    print(f"Model token output: {m_o}")
                    print(f"Model token thoughts: {m_th}")
                    
                    s_i, s_o, s_th = 0, 0, 0
                    try:
                        session = await session_service.get_session(app_name=APP_NAME, session_id=p_id, user_id=user_id)
                        if session and hasattr(session, 'state') and session.state:
                            tc = session.state.get('token_counts', {})
                            s_i, s_o, s_th = tc.get('input_tokens', 0) or 0, tc.get('output_tokens', 0) or 0, tc.get('thoughts_tokens', 0) or 0
                            print(f"Session token counts: {tc}")
                            print(f"Session token input: {s_i}")
                            print(f"Session token output: {s_o}")
                            print(f"Session token thoughts: {s_th}")
                    except: pass
                    
                    db_i, db_o, db_th = 0, 0, 0
                    try:
                        presentation = await run_in_threadpool(db.presentations.find_one, {"p_id": p_id})
                        if presentation:
                            tc = presentation.get("token_counts", {})
                            db_i, db_o, db_th = tc.get("input_tokens", 0) or 0, tc.get("output_tokens", 0) or 0, tc.get("thoughts_tokens", 0) or 0
                            print(f"DB token counts: {tc}")
                            print(f"DB token input: {db_i}")
                            print(f"DB token output: {db_o}")
                            print(f"DB token thoughts: {db_th}")
                    except: pass
                    
                    total_i = m_i + db_i + s_i
                    total_o = m_o + db_o + s_o
                    
                    # Log final totals before reset for debugging
                    logger.info(f"📊 Final Token Aggregation: Input={total_i}, Output={total_o} (from DB source)")
                    
                    # Reset counts for this presentation in DB - standardizing on 'end' reporting
                    await run_in_threadpool(db.presentations.update_one,
                        {"p_id": p_id},
                        {"$set": {"token_counts": {"input_tokens": 0, "output_tokens": 0, "thoughts_tokens": 0}}}
                    )
                    
                    total_duration = time.time() - task_start_time
                    duration_display = f"{int(total_duration // 60)}m {int(total_duration % 60)}s" if total_duration > 60 else f"{int(total_duration)}s"
                    
                    # Report tokens and log usage via RabbitMQ
                    s_key = os.getenv("SERVER_API_KEY")
                    f_id = os.getenv("FEATURE_ENDPOINT_ID")
                    if s_key and f_id:
                        try:
                            rabbitmq_url = os.getenv("RABBITMQ_URL")
                            if rabbitmq_url:
                                connection = await aio_pika.connect_robust(rabbitmq_url)
                                async with connection:
                                    channel = await connection.channel()
                                    
                                    # 1. Credits Process Settlement
                                    settlement_payload = {
                                        "user_id": user_id,
                                        "usage_key": usage_key,
                                        "feature_endpoint_id": f_id,
                                        "usages": [
                                            {
                                                "ai_model": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
                                                "input_tokens": total_i,
                                                "output_tokens": total_o,
                                            }
                                        ]
                                    }
                                    await FeatureLogger.settle_via_queue(channel, settlement_payload)
                                    logger.info(f"✅ Credit settlement published for p_id={p_id}")

                                    # 2. Feature Usage Log (Success)
                                    log_payload = {
                                        "feature_endpoint_value": "agents-presentation-generation",
                                        "user_id": user_id,
                                        "user_email": user_email,
                                        "usage_key": usage_key,
                                        "method": "POST",
                                        "query": {},
                                        "params": {"p_id": p_id},
                                        "payload": {"message": user_message},
                                        "response": {"status": "completed", "duration": duration_display},
                                        "code": 200,
                                        "status": "success"
                                    }
                                    await FeatureLogger.dispatch_to_queue(channel, log_payload)
                                    logger.info(f"✅ Feature usage log published for p_id={p_id}")
                            else:
                                # Fallback to sync API if RabbitMQ is not available
                                payload = {"user_id": user_id, "feature_endpoint_id": f_id, "input_token": total_i, "output_token": total_o, "model_name": os.getenv("GEMINI_MODEL", "gemini-2.5-flash")}
                                headers = {"x-server-api-key": s_key, "Content-Type": "application/json"}
                                async with httpx.AsyncClient(timeout=10.0) as client:
                                    await client.post(f"{token_api_base_url}/api/token-process/end", json=payload, headers=headers)
                        except Exception as log_err:
                            logger.error(f"❌ Failed to log/settle usage: {log_err}")
                    
                    await reset_token_tracker(p_id)
                    try:
                        actions = EventActions(state_delta={"token_counts": {"input_tokens": 0, "output_tokens": 0, "thoughts_tokens": 0}})
                        await safe_append_event(session_service, p_id, user_id, Event(invocation_id=f"reset_{p_id}_{int(time.time())}", author="system", actions=actions, timestamp=time.time()))
                    except: pass
                    
                    await run_in_threadpool(db.presentations.update_one,
                        {"p_id": p_id},
                        {"$set": {"status": "completed", "completion_date": get_utc_timestamp_iso(), "duration_seconds": round(total_duration, 2), "duration_display": duration_display}}
                    )
                    
                    completed_event = {"author": "terminal", "event": "completed", "p_id": p_id, "status": "completed", "timestamp": get_utc_timestamp_iso(), "duration": duration_display}
                    log_socketio_message(completed_event, p_id, "outgoing")
                    await manager.broadcast_to_presentation(completed_event, p_id, user_id)
                    break
                    
            except Exception as run_error:
                logger.warning(f"⚠️ Agent error (attempt {retry_count + 1}): {run_error}")
                if retry_count < MAX_RETRIES - 1: 
                    # Log internal retry error to RabbitMQ
                    await log_to_rabbitmq({
                        "feature_endpoint_value": "agents-presentation-generation",
                        "user_id": user_id,
                        "user_email": user_email,
                        "usage_key": usage_key,
                        "method": "POST",
                        "params": {"p_id": p_id},
                        "payload": {"message": user_message, "retry_count": retry_count + 1},
                        "response": {"error": str(run_error), "status": "retrying"},
                        "code": 499, # Custom code for retry
                        "status": "failed"
                    })
                    await asyncio.sleep(10)
                else:
                    await run_in_threadpool(db.presentations.update_one, {"p_id": p_id}, {"$set": {"status": "failed", "error": str(run_error)}})
                    failed_event = {"type": "terminal", "event": "failed", "p_id": p_id, "status": "failed", "message": str(run_error), "timestamp": get_utc_timestamp_iso()}
                    log_socketio_message(failed_event, p_id, "outgoing")
                    await manager.broadcast_to_presentation(failed_event, p_id, user_id)
                    
                    # Log failed run to RabbitMQ
                    try:
                        rabbitmq_url = os.getenv("RABBITMQ_URL")
                        if rabbitmq_url:
                            connection = await aio_pika.connect_robust(rabbitmq_url)
                            async with connection:
                                channel = await connection.channel()
                                log_payload = {
                                    "feature_endpoint_value": "agents-presentation-generation",
                                    "user_id": user_id,
                                    "user_email": user_email,
                                    "usage_key": usage_key,
                                    "method": "POST",
                                    "params": {"p_id": p_id},
                                    "payload": {"message": user_message},
                                    "response": {"error": str(run_error), "retries": retry_count + 1},
                                    "code": 500,
                                    "status": "failed"
                                }
                                await FeatureLogger.dispatch_to_queue(channel, log_payload)
                    except: pass
        
    except Exception as e:
        logger.error(f"❌ Critical error in agent execution: {e}")
        await run_in_threadpool(db.presentations.update_one, {"p_id": p_id}, {"$set": {"status": "failed", "error": str(e)}})
        critical_failed_event = {"type": "terminal", "event": "failed", "p_id": p_id, "status": "failed", "message": str(e), "timestamp": get_utc_timestamp_iso()}
        log_socketio_message(critical_failed_event, p_id, "outgoing")
        await manager.broadcast_to_presentation(critical_failed_event, p_id, user_id)
        
        # Log critical failure to RabbitMQ
        try:
            rabbitmq_url = os.getenv("RABBITMQ_URL")
            if rabbitmq_url:
                connection = await aio_pika.connect_robust(rabbitmq_url)
                async with connection:
                    channel = await connection.channel()
                    log_payload = {
                        "feature_endpoint_value": "agents-presentation-generation",
                        "user_id": user_id,
                        "user_email": user_email,
                        "usage_key": usage_key,
                        "method": "POST",
                        "params": {"p_id": p_id},
                        "payload": {"message": user_message},
                        "response": {"error": str(e)},
                        "code": 500,
                        "status": "failed"
                    }
                    await FeatureLogger.dispatch_to_queue(channel, log_payload)
        except Exception as log_err:
            logger.warning(f"⚠️ Failed to log critical failure to queue: {log_err}")

async def process_agent_output(ev, part, p_id: str, user_id: str, db, user_message: str, is_first_message: bool = False, usage_key: str = None, user_email: str = None) -> Optional[dict]:
    """Process agent event and return formatted data"""
    db_entry = {"user_id": user_id, "session_id": p_id, "role": "agent", "author": ev.author or "agent", "p_id": p_id, "timestamp": get_utc_timestamp_iso()}
    event_data = {"type": "chunk", "author": ev.author or "agent", "p_id": p_id, "timestamp": get_utc_timestamp_iso()}
    
    if is_first_message:
        event_data["user_message"] = user_message
        db_entry["user_message"] = user_message
    
    author = ev.author or "agent"
    agent_name = author
    # Agent name mapping
    if "slide_insertion" in author.lower(): agent_name = "slide_insertion_orchestrator"
    elif "multi_slide" in author.lower(): agent_name = "multi_slide_modification_orchestrator"
    elif "single_slide_modifier" in author: agent_name = "single_slide_modifier"
    elif "slide_modification" in author.lower(): agent_name = "slide_modification_agent"
    elif "slidecreation" in author.lower(): agent_name = "SlideCreationPipeline"
    elif "planning_agent" in author.lower(): agent_name = "planning_agent"
    elif "slide_generator_agent" in author.lower(): agent_name = "slide_generator_agent"
    elif "presentation_spec" in author.lower(): agent_name = "presentation_spec_extractor_agent"
    elif "enhanced_slide" in author.lower(): agent_name = "enhanced_slide_generator"
    elif "keyword_research" in author.lower(): agent_name = "keyword_research_agent"
    
    db_entry["agent_name"] = agent_name
    event_data["agent_name"] = agent_name
    
    if "enhanced_slide_generator" in author:
        text = part.text.strip()
        is_html = text.startswith("```html") or text.startswith("<!DOCTYPE") or text.startswith("<html")
        content = text.removeprefix("```html").removesuffix("```").strip() if text.startswith("```html") else text
        
        if is_html:
            event_data["html_content"] = content
            db_entry.update({"content_type": "html", "html_content": content})
        else:
            event_data["thinking"] = text
            db_entry.update({"content_type": "thinking", "thinking": text})
    else:
        text = part.text.strip()
        json_content = text if not text.startswith("```") else text.split("```")[1].removeprefix("json").strip()
        try:
            json_data = json.loads(json_content)
            if isinstance(json_data, dict):
                if "presentation_spec_extractor_agent" in author:
                    topic = json_data.get("topic", "")
                    if topic:
                        event_data["topic"] = topic
                        await run_in_threadpool(db.presentations.update_one, {"user_id": user_id, "p_id": p_id}, {"$set": {"title": topic}}, upsert=True)
                        # Log topic detection step to RabbitMQ
                        asyncio.create_task(log_to_rabbitmq({
                            "feature_endpoint_value": "agents-presentation-generation",
                            "user_id": user_id,
                            "user_email": user_email,
                            "usage_key": usage_key,
                            "method": "POST",
                            "params": {"p_id": p_id},
                            "payload": {"topic": topic},
                            "response": {"status": "topic_detected"},
                            "code": 200,
                            "status": "success"
                        }))
                event_data.update(json_data)
                db_entry.update({"content_type": "json", "json_data": json_data, "parsed_output": json_content})
            else:
                event_data["text"] = text
                db_entry.update({"content_type": "text", "parsed_output": text})
        except:
            event_data["text"] = text
            db_entry.update({"content_type": "text", "parsed_output": text})
    
    # Exclude internal/technical agents from user-facing logs
    excluded_authors = ["enhanced_slide_generator", "template_selector"]
    if not any(excluded in author for excluded in excluded_authors):
        res = await run_in_threadpool(db.agent_outputs_2.insert_one, db_entry)
        event_data["event_id"] = str(res.inserted_id)
    else:
        event_data["event_id"] = f"slide_{p_id}_{int(time.time())}"
    
    return event_data

async def process_function_call_output(ev, part, p_id: str, user_id: str, db, usage_key: str = None, user_email: str = None) -> Optional[dict]:
    """Process function_call and store in agent_outputs_2"""
    try:
        fc = part.function_call
        tool_name = getattr(fc, "name", "unknown_tool")
        if tool_name == "fetch_slide_data": tool_name = "fetch_slide_data_tool"
        args = _normalize_tool_args(getattr(fc, "args", {}))
        agent_name = ev.author or "unknown_agent"
        
        # Mappings
        if "slide_orchestration" in agent_name.lower(): agent_name = "slide_orchestration_agent"
        elif "slide_insertion" in agent_name.lower(): agent_name = "slide_insertion_orchestrator"
        elif "multi_slide" in agent_name.lower(): agent_name = "multi_slide_modification_orchestrator"
        
        context = slide_op_manager.get_context(p_id, user_id)
        
        if agent_name == "slide_orchestration_agent":
            if tool_name in ["topic_checker_agent", "query_enhancer_agent"]:
                content = "checking topic" if tool_name == "topic_checker_agent" else "enhancing the user query"
                log = {"author": tool_name, "p_id": p_id, "timestamp": get_utc_timestamp_iso(), "session_id": p_id, "user_id": user_id, "parsed_output": content}
                res = await run_in_threadpool(db.agent_outputs_2.insert_one, log)
                return {"type": "tool_call", "agent_name": tool_name, "p_id": p_id, "text": content, "timestamp": get_utc_timestamp_iso(), "event_id": str(res.inserted_id)}
            return None
        
        slide_num = args.get("slide_number") or args.get("slideNumber") or args.get("slide_num")
        if slide_num:
            context["slide_number"] = int(slide_num)
            context["slide_index"] = max(int(slide_num) - 1, 0)
            
        entry = {
            "user_id": user_id, "session_id": p_id, "role": "agent", "author": agent_name, "p_id": p_id,
            "content_type": "function_call", "tool_name": tool_name, "tool_arguments": args,
            "slide_number": context.get("slide_number"), "slide_index": context.get("slide_index"),
            "timestamp": get_utc_timestamp_iso()
        }
        # List of internal agent tool names that should not appear in user logs
        internal_tools = [
            "keyword_research_agent", 
            "search_query_agent",
            "presentation_spec_extractor_agent",
            "planning_agent",
            "template_selector_agent",
            "retrieve_research_context",  # Enhanced slide generator tool
            "search_images"  # Enhanced slide generator tool
        ]
        
        # Only log to agent_outputs_2 if it's not an internal tool and not from enhanced_slide_generator or template_selector
        excluded_agent_names = ["enhanced_slide_generator", "template_selector"]
        if not any(excluded in agent_name for excluded in excluded_agent_names) and tool_name not in internal_tools:
            res = await run_in_threadpool(db.agent_outputs_2.insert_one, entry)
            event_id = str(res.inserted_id)
        else:
            event_id = f"fc_{p_id}_{int(time.time())}"
        
        return {
            "type": "tool_call", 
            "author": agent_name, "p_id": p_id, "tool_name": tool_name,
            "tool_arguments": args, "timestamp": get_utc_timestamp_iso(), "event_id": event_id,
            "slide_index": context.get("slide_index"), "tool_event": tool_name
        }
    except Exception as e:
        logger.error(f"Error in fc output: {e}")
        # Log function call processing error to RabbitMQ
        asyncio.create_task(log_to_rabbitmq({
            "feature_endpoint_value": "agents-presentation-generation",
            "user_id": user_id,
            "user_email": user_email,
            "usage_key": usage_key,
            "method": "POST",
            "params": {"p_id": p_id},
            "response": {"error": f"Function Call Processing Error: {str(e)}"},
            "code": 500,
            "status": "failed"
        }))
        return None

async def process_function_response_output(ev, part, p_id: str, user_id: str, db, usage_key: str = None, user_email: str = None) -> Optional[dict]:
    """Process function_response and store in agent_outputs_2"""
    try:
        fr = part.function_response
        t_name = getattr(fr, "name", "unknown_tool")
        t_res = getattr(fr, "response", {})
        
        data = t_res if not isinstance(t_res, str) else (json.loads(t_res) if t_res.startswith("{") else {"result": t_res})
        agent_name = ev.author or "unknown_agent"
        
        if "slide_insertion" in agent_name.lower(): agent_name = "slide_insertion_orchestrator"
        elif "multi_slide" in agent_name.lower(): agent_name = "multi_slide_modification_orchestrator"
        
        context = slide_op_manager.get_context(p_id, user_id)
        
        if agent_name == "slide_insertion_orchestrator" and t_name == "validate_insertion_position":
            res = data.get("result") if isinstance(data, dict) else data
            try:
                p = json.loads(res) if isinstance(res, str) else res
                ip = p.get("insert_position")
                if ip is not None:
                    context["slide_index"] = int(ip)
                    context["slide_number"] = int(ip) + 1
            except: pass
            return None
        
        track = (agent_name == "multi_slide_modification_orchestrator") or (t_name == "generate_slide_html_sync") or (t_name == "content_modifier_agent")
        if not track: return None
        
        html, thinking = None, None
        if isinstance(data, dict):
            html = data.get("generated_html") or data.get("html") or data.get("html_content")
            thinking = data.get("thinking")
            if not html and isinstance(data.get("result"), str):
                try: 
                    p = json.loads(data["result"])
                    html = p.get("generated_html"); thinking = thinking or p.get("thinking")
                except: pass
        
        if html: 
            context["html_content"] = html
            # Log slide generation step to RabbitMQ
            asyncio.create_task(log_to_rabbitmq({
                "feature_endpoint_value": "agents-presentation-generation",
                "user_id": user_id,
                "user_email": user_email,
                "usage_key": usage_key,
                "method": "POST",
                "params": {"p_id": p_id, "slide_index": context.get("slide_index")},
                "payload": {"tool_name": t_name},
                "response": {"status": "slide_generated", "slide_number": context.get("slide_number")},
                "code": 200,
                "status": "success"
            }))
        
        entry = {
            "user_id": user_id, "session_id": p_id, "role": "agent", "author": agent_name, "p_id": p_id,
            "content_type": "function_response", "tool_name": t_name, "tool_response": data,
            "slide_number": context.get("slide_number"), "slide_index": context.get("slide_index"),
            "html_content": html, "timestamp": get_utc_timestamp_iso()
        }
        
        # Exclude internal/technical agents from user-facing logs
        excluded_agent_names = ["enhanced_slide_generator", "template_selector"]
        if not any(excluded in agent_name for excluded in excluded_agent_names) and not (agent_name == "slide_insertion_orchestrator" and t_name == "generate_slide_html_sync"):
            res = await run_in_threadpool(db.agent_outputs_2.insert_one, entry)
            event_id = str(res.inserted_id)
        else:
            event_id = f"fr_{p_id}_{int(time.time())}"
            
        ev_a = "enhanced_slide_generator" if "enhanced_slide_generator" in agent_name or t_name == "generate_slide_html_sync" else agent_name
        ev_d = {
            "type": "tool_response", "author": ev_a, "p_id": p_id, "tool_name": t_name,
            "tool_response": data, "tool_event": t_name, "timestamp": get_utc_timestamp_iso(),
            "event_id": event_id, "slide_index": context.get("slide_index")
        }
        if thinking: ev_d["thinking"] = thinking
        if html: ev_d["html_content"] = html
        return ev_d
    except Exception as e:
        logger.error(f"Error in fr output: {e}")
        # Log function response processing error to RabbitMQ
        asyncio.create_task(log_to_rabbitmq({
            "feature_endpoint_value": "agents-presentation-generation",
            "user_id": user_id,
            "user_email": user_email,
            "usage_key": usage_key,
            "method": "POST",
            "params": {"p_id": p_id},
            "response": {"error": f"Function Response Processing Error: {str(e)}"},
            "code": 500,
            "status": "failed"
        }))
        return None
