import os
import logging
import asyncio
from dotenv import load_dotenv

# ADK and Core Imports
from google.adk.agents import RunConfig, SequentialAgent
from google.adk.sessions import DatabaseSessionService
from google.genai import types

# Sub-agent Imports
from .sub_agents.presentation_spec_extractor_agent import create_presentation_spec_extractor_agent
from .keyword_research_agent.agent import create_keyword_research_agent
from .sub_agents.lightweight_slide_pipeline import create_lightweight_slide_generation_agent
from .sub_agents.lightweight_planning_agent import create_lightweight_planning_agent

# Local Refactored Imports
from .browser_agent.browser_agent import BrowserAgent
from .utils.db_utils import get_db, test_redis_connection, log_event_to_db

# Load environment variables
load_dotenv()

# Setup logging
logger = logging.getLogger(__name__)

# Configuration
GEMINI_MODEL = os.getenv("GEMINI_MODEL_FLASH", "gemini-2.0-flash")
GEMINI_MODEL_PRO = os.getenv("GEMINI_MODEL_PRO", "gemini-2.0-flash")

class LoggingSequentialAgent(SequentialAgent):
    async def _run_async_impl(self, ctx):
        MAX_PART_RETRIES = 3
        RETRY_DELAY = 1  # seconds
        db = get_db()
        
        for agent in self.sub_agents:
            async for event in agent.run_async(ctx):
                # Yield the original event from the sub-agent
                yield event
                logger.debug(f"Event from {agent.name}: {event.id} - {event.author}")
                
                # Logic for handling content parts with retries
                for part_attempt in range(MAX_PART_RETRIES):
                    if event.content and getattr(event.content, "parts", None):
                        break
                    if part_attempt < MAX_PART_RETRIES - 1:
                        logger.warning(f"⚠️ event.content.parts is None (attempt {part_attempt+1}/{MAX_PART_RETRIES}) – retrying.")
                        await asyncio.sleep(RETRY_DELAY)
                    else:
                        logger.error(f"❌ Skipping event {event.id} – no content.parts after {MAX_PART_RETRIES} attempts.")
                        continue

# Main Pipeline Definition
pipeline = LoggingSequentialAgent(
    name="SlideCreationPipeline",
    description="Pipeline for creating a presentation with multiple agents",
    sub_agents=[
        create_presentation_spec_extractor_agent(),
        create_keyword_research_agent(),
        BrowserAgent(name="BrowserAgent"),
        create_lightweight_planning_agent(),
        create_lightweight_slide_generation_agent()
    ]
)