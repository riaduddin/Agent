import logging
import json
import re
import os
import asyncio
from typing import AsyncGenerator
from pydantic import Field
from google.adk.agents import BaseAgent, ParallelAgent
from google.adk.events import Event, EventActions
from google.genai import types

from ..utils.db_utils import get_redis_client, test_redis_connection
from ..utils.json_utils import extract_valid_json
from .make_browser_worker import make_browser_worker

# Import Qdrant utilities
import sys
from pathlib import Path
# Assuming the root is 3 levels up from this file
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from tools.qdrant_utils import get_qdrant_manager
from core.token_logger import log_token_usage

logger = logging.getLogger(__name__)

class BrowserAgent(BaseAgent):
    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)

    async def _run_async_impl(self, ctx) -> AsyncGenerator[Event, None]:
        p_id = ctx.session.state.get("p_id")
        user_id = ctx.session.state.get("user_id", "unknown")
        
        redis_client = get_redis_client()
        redis_connected = await test_redis_connection()
        
        # Initialize Qdrant manager
        try:
            qdrant = get_qdrant_manager()
            logger.info(f"✅ Qdrant manager initialized for user: {user_id}, p_id: {p_id}")
        except Exception as e:
            logger.error(f"❌ Qdrant initialization failed: {e}")
            qdrant = None
        
        kw_raw = ctx.session.state.get("keywords", "")
        
        # Strip triple backticks and extract the actual JSON
        match = re.search(r'{.*}', kw_raw, re.DOTALL)
        if match:
            try:
                kw_obj = json.loads(match.group(0))
                kw_list = kw_obj.get("keywords", [])
            except json.JSONDecodeError as e:
                logger.error(f"❌ Failed to parse keyword JSON: {e}")
                kw_list = []
        else:
            kw_list = []

        if not kw_list:
            yield Event(
                author=self.name,
                content=types.Content(parts=[types.Part(text="No keywords to search.")]),
                actions=EventActions(escalate=True),
            )
            return

        batch_size = int(os.getenv("KEYWORD_BATCH_SIZE", "15"))
        all_summaries = []

        for start in range(0, len(kw_list), batch_size):
            batch = kw_list[start:start + batch_size]
            batch_summaries = []
            cache_results = {kw: None for kw in batch}

            if redis_connected:
                for kw in batch:
                    try:
                        cached = await redis_client.get(f"browser_cache:{kw}")
                        if cached:
                            cached_payload = json.loads(cached.decode("utf-8") if isinstance(cached, bytes) else cached)
                            summary = cached_payload.get("summary", "")
                            sources = cached_payload.get("sources", [])
                            
                            # Yield sources
                            for src in sources:
                                source_event_data = {"domain": src.get("title", "Unknown Title"), "url": src.get("url", "Unknown URL")}
                                yield Event(author=self.name, content=types.Content(parts=[types.Part(text=json.dumps(source_event_data))]))
                            
                            # Yield summary
                            summary_event_data = {"summary": summary}
                            yield Event(author=self.name, content=types.Content(parts=[types.Part(text=json.dumps(summary_event_data))]))
                            
                            all_summaries.append(summary)
                            batch_summaries.append(summary)
                            cache_results[kw] = summary
                    except Exception as e:
                        logger.warning(f"⚠️ Redis cache check failed for '{kw}': {e}")

            uncached_keywords = [kw for kw in batch if cache_results.get(kw) is None]
            if uncached_keywords:
                BROWSER_BATCH_SIZE = int(os.getenv("BROWSER_BATCH_SIZE", "3"))
                BROWSER_BATCH_DELAY = float(os.getenv("BROWSER_BATCH_DELAY", "0.5"))
                
                for browser_batch_start in range(0, len(uncached_keywords), BROWSER_BATCH_SIZE):
                    browser_batch_end = min(browser_batch_start + BROWSER_BATCH_SIZE, len(uncached_keywords))
                    browser_batch_kws = uncached_keywords[browser_batch_start:browser_batch_end]
                    
                    if browser_batch_start > 0:
                        await asyncio.sleep(BROWSER_BATCH_DELAY)
                    
                    workers = [make_browser_worker(kw, start + browser_batch_start + idx) for idx, kw in enumerate(browser_batch_kws)]
                    parallel_browser = ParallelAgent(
                        name=f"parallel_browser_{start}_{browser_batch_start}", 
                        sub_agents=workers,
                        description=f"browsing information in parallel"
                    )
        
                    idx_to_keyword = {start + browser_batch_start + i: kw for i, kw in enumerate(browser_batch_kws)}
                    
                    async for ev in parallel_browser.run_async(ctx):
                        # Token counting
                        if hasattr(ev, 'usage_metadata') and ev.usage_metadata:
                            usage = ev.usage_metadata
                            i_t = getattr(usage, 'prompt_token_count', 0) or 0
                            o_t = getattr(usage, 'candidates_token_count', 0) or 0
                            th_t = getattr(usage, 'thoughts_token_count', 0) or 0
                            
                            state_exists = 'token_counts' in ctx.session.state
                            if not state_exists:
                                msg = f"Initializing token_counts in ctx.session.state for p_id: {p_id}"
                                logger.info(f"📝 {msg}")
                                ctx.session.state['token_counts'] = {'input_tokens': 0, 'output_tokens': 0, 'thoughts_tokens': 0}
                                # Log initialization to token auditor
                                log_token_usage(p_id=p_id or "unknown", author=self.name, message=msg)
                            
                            # Update session state
                            ctx.session.state['token_counts']['input_tokens'] += i_t
                            ctx.session.state['token_counts']['output_tokens'] += o_t
                            ctx.session.state['token_counts']['thoughts_tokens'] += th_t
                            message=f"Updating token_counts in ctx.session.state for browser p_id: {p_id}"
                            # Log to centralized token auditor (file)
                            log_token_usage(
                                p_id=p_id or "unknown",
                                author=ev.author or self.name,
                                input_tokens=i_t,
                                output_tokens=o_t,
                                thoughts_tokens=th_t,
                                message=message
                            )
                            
                            if i_t > 0 or o_t > 0:
                                logger.debug(f"📊 Browser Token Usage ({ev.author}): In={i_t}, Out={o_t}, StateExists={state_exists}")
                        
                        worker_idx = None
                        keyword = None
                        if hasattr(ev, 'author') and ev.author and 'browser_worker_' in ev.author:
                            try:
                                worker_idx = int(ev.author.split('browser_worker_')[1])
                                keyword = idx_to_keyword.get(worker_idx)
                            except: pass
                        
                        # Grounding metadata
                        sources = []
                        if hasattr(ev, 'grounding_metadata') and ev.grounding_metadata:
                            if hasattr(ev.grounding_metadata, 'grounding_chunks') and ev.grounding_metadata.grounding_chunks:
                                for chunk in ev.grounding_metadata.grounding_chunks:
                                    if hasattr(chunk, 'web') and chunk.web:
                                        source = {"title": getattr(chunk.web, 'title', 'Unknown Title'), "url": getattr(chunk.web, 'uri', 'Unknown URL')}
                                        yield Event(author=ev.author, content=types.Content(parts=[types.Part(text=json.dumps({"domain": chunk.web.title, "url": chunk.web.uri}))]))
                                        sources.append(source)
                        
                        research_text = ""
                        if hasattr(ev, 'content') and ev.content and hasattr(ev.content, 'parts') and ev.content.parts:
                            for part in ev.content.parts:
                                if hasattr(part, 'text') and part.text:
                                    research_text = part.text
                                    break
                        
                        if research_text and keyword:
                            if qdrant:
                                try:
                                    qdrant.store_research_data(text=research_text, user_id=user_id, p_id=p_id, keyword=keyword, sources=sources)
                                except Exception as e:
                                    logger.error(f"❌ Qdrant storage failed: {e}")
                            
                            summary_data = {"summary": research_text}
                            yield Event(author=ev.author, content=types.Content(parts=[types.Part(text=json.dumps(summary_data))]))
                            
                            if redis_connected:
                                try:
                                    redis_payload = {"status": "success", "summary": research_text, "sources": sources}
                                    await redis_client.set(f"browser_cache:{keyword}", json.dumps(redis_payload), ex=86400)
                                except Exception as e:
                                    logger.warning(f"⚠️ Redis cache set failed: {e}")
                        elif ev.content and hasattr(ev.content, 'parts') and ev.content.parts:
                            yield ev
