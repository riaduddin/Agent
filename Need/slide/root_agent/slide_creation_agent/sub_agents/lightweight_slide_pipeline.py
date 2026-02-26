"""
Lightweight Slide Generation Pipeline
Uses outline-based planning with Qdrant retrieval per slide
"""
from google.adk.agents import BaseAgent, ParallelAgent
from google.adk.events import Event, EventActions
from google.genai import types
from typing import AsyncGenerator
import json
import logging
import re
import sys
import asyncio
from pathlib import Path
from datetime import datetime, timezone
from bs4 import BeautifulSoup
import os
import re
import logging
import asyncio
from typing import AsyncGenerator
import json
import random
from dotenv import load_dotenv
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
# Add root directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from tools.template_retrieval import retrieve_html_templates
from core.database import get_mongo_client
from core.token_logger import log_token_usage
import random

from .lightweight_planning_agent import create_lightweight_planning_agent
from .template_selector_agent import create_template_selector_agent
from .enhanced_slide_generator import create_enhanced_slide_generator
from .slide_validation_agent import validate_and_fix_slide

logger = logging.getLogger(__name__)

# Database connection - reuse existing client
load_dotenv()
client = get_mongo_client()
db = client["slide_creator_db"]
# Slide validation configuration
ENABLE_SLIDE_VALIDATION = os.getenv("ENABLE_SLIDE_VALIDATION", "true").lower() == "true"
SLIDE_VALIDATION_RETRIES = int(os.getenv("SLIDE_VALIDATION_RETRIES", "5"))
SLIDE_VALIDATION_MAX_ITERATIONS = int(os.getenv("SLIDE_VALIDATION_MAX_ITERATIONS", "5"))
MAX_VALIDATION_WORKERS = int(os.getenv("MAX_VALIDATION_WORKERS", "2"))

# Dynamic Status Messages for a better user experience
from .status_messages import STATUS_MESSAGES, get_status_msg

# Multiprocessing wrapper for validation (runs async function in separate process)
def _validate_slide_worker(args):
    """
    Worker function for multiprocessing validation.
    Runs the async validation function in a new event loop.
    
    Args:
        args: Tuple of (html_content, p_id, slide_index, max_retries, max_iterations, session_state)
        session_state: Dictionary containing session state from the original ctx
    
    Returns:
        Tuple of (slide_index, fixed_html, thinking_text, validation_iterations_count, error)
    """
    html_content, p_id, slide_index, max_retries, max_iterations, session_state = args
    
    loop = None
    try:
        # Create a new event loop for this process
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Reconstruct context using InvocationContext (the actual Context class in ADK)
        # Import from the correct locations
        from google.adk.agents.invocation_context import InvocationContext, new_invocation_context_id
        from google.adk.sessions import Session, InMemorySessionService
        from google.adk.agents.base_agent import BaseAgent
        
        # Create a minimal session service for the worker process
        session_service = InMemorySessionService()
        
        # Create session with the state from the original ctx
        session_id = session_state.get("p_id", f"validation_{p_id}_{slide_index}")
        app_name = session_state.get("app_name", "presentation_service")
        user_id = session_state.get("user_id", "unknown")
        
        # Create session with state from the original ctx
        # Session requires 'id' (not 'session_id'), 'app_name', and 'user_id'
        session = Session(
            id=session_id,  # Use 'id' not 'session_id'
            app_name=app_name,
            user_id=user_id,
            state=session_state.copy() if session_state else {"p_id": p_id}
        )
        
        # Create a minimal agent for the context (validation agent will be created later)
        # We need a dummy agent just for InvocationContext creation
        class DummyAgent(BaseAgent):
            def __init__(self):
                super().__init__(name="dummy_validation_agent")
        
        dummy_agent = DummyAgent()
        
        # Create InvocationContext (the actual Context class) - this is the real ctx
        ctx = InvocationContext(
            session=session,
            agent=dummy_agent,
            session_service=session_service,
            invocation_id=new_invocation_context_id()
        )
        
        # Run the async validation function
        result = loop.run_until_complete(
            validate_and_fix_slide(
                html_content,
                p_id,
                slide_index,
                ctx,
                max_retries,
                max_iterations
            )
        )
        
        fixed_html, thinking_text, validation_iterations_count = result
        
        return (slide_index, fixed_html, thinking_text, validation_iterations_count, None)
        
    except Exception as e:
        import logging
        worker_logger = logging.getLogger(__name__)
        worker_logger.error(f"❌ Validation worker error for slide {slide_index + 1}: {e}")
        import traceback
        worker_logger.error(traceback.format_exc())
        # Return original HTML on error
        return (slide_index, html_content, None, 0, str(e))
    finally:
        # Ensure loop is closed
        if loop:
            try:
                loop.close()
            except:
                pass


def save_slide_to_database(
    p_id: str,
    user_id: str,
    slide_index: int,
    slide_html: str,
    slide_outline: dict,
    template_info: dict,
    global_theme: dict,
    thinking_text: str = None,
    screenshot_path: str = None,
    original_html: str = None,
    original_screenshot_path: str = None,
    validation_iterations_count: int = 0
    ) -> bool:
    """
    Save generated slide with all required metadata for future modifications
    
    CRITICAL: Stores RAW HTML exactly as generated to preserve structure for content-only modifications.
    
    Args:
        p_id: Presentation ID
        user_id: User ID
        slide_index: 0-based slide index (slide 1 = index 0)
        slide_html: Complete generated HTML content (may include ```html markers)
        slide_outline: Slide plan from planning agent
        template_info: Information about the selected template
        global_theme: Presentation theme colors and fonts
        thinking_text: Optional thinking/reasoning text from agent
        validation_iterations_count: Number of validation iterations performed (0 if no validation needed)
    
    Returns:
        bool: True if save was successful, False otherwise
    """
    try:
        # Parse HTML content to remove markdown code blocks (like ```html)
        html_content = slide_html
        text_stripped = html_content.strip()
        
        # Remove markdown code blocks if present
        if text_stripped.startswith("```html"):
            html_content = text_stripped.removeprefix("```html").removesuffix("```").strip()
        elif text_stripped.startswith("<!DOCTYPE") or text_stripped.startswith("<html"):
            html_content = text_stripped
        
        # Parse HTML ONLY for metadata extraction, do NOT modify the original HTML
        # This preserves the exact structure for content-only modifications
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract title
        h1_tag = soup.find('h1')
        extracted_title = h1_tag.get_text(strip=True) if h1_tag else ""
        
        # Extract all headings
        extracted_headings = [h.get_text(strip=True) for h in soup.find_all(['h1', 'h2', 'h3', 'h4'])]
        
        # Extract paragraphs
        extracted_paragraphs = [p.get_text(strip=True) for p in soup.find_all('p') if p.get_text(strip=True)]
        
        # Check for charts
        has_charts = bool(soup.find('canvas'))
        
        # Check for lists
        has_lists = bool(soup.find(['ul', 'ol']))
        
        # Extract Font Awesome icon classes
        icon_elements = soup.find_all('i', class_=True)
        icon_names = []
        for icon in icon_elements:
            classes = icon.get('class', [])
            if isinstance(classes, list):
                icon_names.extend(classes)
            else:
                icon_names.append(classes)
        
        # Calculate text length
        text_length = len(soup.get_text())
        
        # Create content metadata for quick reference (modification system can use this)
        content_metadata = {
            "extracted_title": extracted_title,
            "extracted_headings": extracted_headings,
            "extracted_paragraphs": extracted_paragraphs,
            "has_charts": has_charts,
            "has_lists": has_lists,
            "icon_names": list(set(icon_names)),  # Remove duplicates
            "text_length": text_length
        }
        
        # Create slide document
        slide_doc = {
            # Identifiers
            "p_id": p_id,
            "user_id": user_id,
            "slide_index": slide_index,
            "slide_number": slide_index + 1,
            
            # HTML Content (CRITICAL: Store cleaned HTML to preserve exact structure)
            # Modification system will parse this to change ONLY content, not structure
            "body": html_content,  # ⭐ STORED AS cleaned HTML (no markdown blocks)
            
            # Thinking/Reasoning Text (from agent's thought process)
            "thinking": thinking_text if thinking_text else None,
            
            # Slide Plan (CRITICAL FOR CONTEXT)
            "slide_plan": slide_outline,
            
            # Template Information (CRITICAL FOR REGENERATION)
            "template_info": template_info,
            
            # Content Metadata (USEFUL FOR SMART MODIFICATION)
            "content_metadata": content_metadata,
            
            # Screenshot path (for manual review) - validated HTML screenshot
            "screenshot_path": screenshot_path if screenshot_path else None,
            
            # Original HTML before validation (for comparison)
            "original_body": original_html if original_html else None,
            
            # Original screenshot path (before validation)
            "original_screenshot_path": original_screenshot_path if original_screenshot_path else None,
            
            # Validation metadata
            "validation_iterations_count": validation_iterations_count,  # Number of validation iterations (0 if no validation needed)
            
            # Global Theme Reference (stored in presentation doc)
            # We don't duplicate theme here, just reference it
            
            # Timestamps
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
            "timestamp": datetime.now(timezone.utc)
        }
        
        # Insert into database
        result = db.slide_html.insert_one(slide_doc)
        
        if result.inserted_id:
            logger.info(f"✅ Saved slide {slide_index + 1} to database (ID: {result.inserted_id})")
            return True
        else:
            logger.error(f"❌ Failed to save slide {slide_index + 1} - no inserted_id")
            return False
            
    except Exception as e:
        logger.error(f"❌ Exception saving slide {slide_index + 1} to database: {e}")
        import traceback
        traceback.print_exc()
        return False


def parse_loose_json(text: str) -> dict:
    """Parse JSON from possibly fenced/dirty strings.
    - Strips ```json fences
    - Extracts the outermost {...}
    - Removes trailing commas before } and ]
    Returns {} on failure.
    """
    try:
        if not isinstance(text, str):
            return {}
        s = text.strip()
        # Remove common code fences
        s = re.sub(r"^```json\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"^```\s*", "", s)
        s = re.sub(r"```\s*$", "", s)
        # Extract JSON object region
        m = re.search(r"\{[\s\S]*\}", s)
        if m:
            s = m.group(0)
        # Remove trailing commas like ,}\n or ,]\n
        s = re.sub(r",\s*([}\]])", r"\1", s)
        return json.loads(s)
    except Exception:
        return {}


class LightweightSlideGenerationAgent(BaseAgent):
    """
    Orchestrates lightweight slide generation:
    1. Creates outline with specific search queries
    2. Generates slides in parallel using Qdrant retrieval
    """
    
    def __init__(self, name="LightweightSlideGeneration", **kwargs):
        super().__init__(name=name, **kwargs)
    
    async def _run_async_impl(self, ctx) -> AsyncGenerator[Event, None]:
        """Run the lightweight slide generation pipeline"""
        
        # Get inputs from session state
        presentation_spec_raw = ctx.session.state.get("presentation_spec", {})
        
        # Ensure presentation_spec is a dictionary (handle if it's a JSON string)
        if isinstance(presentation_spec_raw, str):
            parsed = parse_loose_json(presentation_spec_raw)
            if parsed:
                presentation_spec = parsed
                logger.info("📋 Parsed presentation_spec from loose JSON string")
            else:
                logger.warning("⚠️ presentation_spec string could not be parsed, applying sane defaults")
                presentation_spec = {}
        elif isinstance(presentation_spec_raw, dict):
            presentation_spec = presentation_spec_raw
        else:
            logger.warning(f"⚠️ presentation_spec has unexpected type: {type(presentation_spec_raw)}, applying sane defaults")
            presentation_spec = {}

        # Normalize required keys with defaults without discarding parsed values
        normalized_defaults = {
            "presentation_type": "regular_presentation",
            "slide_count": 10,
            "color_theme": "#2563EB",
            "tone": "professional",
            "content_focus": [],
            "audience_type": "general"
        }
        for k, v in normalized_defaults.items():
            presentation_spec.setdefault(k, v)
        
        keywords = ctx.session.state.get("keywords", [])
        user_id = ctx.session.state.get("user_id", "unknown")
        p_id = ctx.session.state.get("p_id", "unknown")
        
        # Update presentations collection with slide_count from presentation_spec
        slide_count = presentation_spec.get("slide_count", 10)
        try:
            db.presentations.update_one(
                {"p_id": p_id},
                {"$set": {"total_slides": slide_count, "updated_at": datetime.now(timezone.utc)}}
            )
            logger.info(f"📊 Updated presentations collection with slide_count: {slide_count}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to update presentations collection with slide_count: {e}")
        
        logger.info(f"🎨 Starting lightweight slide generation for p_id: {p_id}")
        
        # STEP 1: Check if plan already exists in session state
        raw_plan = ctx.session.state.get("lightweight_plan", "")
        # print("raw_plan", raw_plan)
        
        plan_result = None
        plan_text = ""
        
        # If plan already exists, skip planning agent and parse it directly
        if raw_plan:
            logger.info("📋 Found existing plan in session state, skipping planning agent...")
            # yield Event(
            #     author=self.name,
            #     content=types.Content(parts=[
            #         types.Part(text="📋 Using existing presentation outline...")
            #     ]),
            # )
            
            # Parse the existing plan
            if isinstance(raw_plan, dict):
                plan_result = raw_plan
                logger.info(f"✅ Plan is already a dictionary with {len(plan_result.get('slide_outline', []))} slides")
            elif isinstance(raw_plan, str):
                plan_text = raw_plan
                
                # Method 1: Try direct JSON parsing
                try:
                    plan_result = json.loads(plan_text)
                    logger.info(f"✅ Extracted plan directly with {len(plan_result.get('slide_outline', []))} slides")
                except json.JSONDecodeError:
                    pass
                
                # Method 2: Remove markdown code blocks
                if not plan_result:
                    try:
                        # Remove ```json and ``` markers
                        cleaned_text = re.sub(r'^```json\s*', '', plan_text, flags=re.MULTILINE)
                        cleaned_text = re.sub(r'```\s*$', '', cleaned_text, flags=re.MULTILINE)
                        cleaned_text = cleaned_text.strip()
                        
                        plan_result = json.loads(cleaned_text)
                        #print("plan_result", plan_result)
                        logger.info(f"✅ Extracted plan after removing markdown with {len(plan_result.get('slide_outline', []))} slides")
                    except json.JSONDecodeError:
                        pass
                
                # Method 3: Find JSON object in text
                if not plan_result:
                    try:
                        json_match = re.search(r'\{[\s\S]*\}', plan_text)
                        if json_match:
                            plan_result = json.loads(json_match.group(0))
                            print("plan_result_with_regex", plan_result)
                            logger.info(f"✅ Extracted plan from regex match with {len(plan_result.get('slide_outline', []))} slides")
                    except:
                        pass
                
                # Method 4: Try to find JSON between braces
                if not plan_result:
                    try:
                        start_idx = plan_text.find('{')
                        end_idx = plan_text.rfind('}')
                        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                            json_str = plan_text[start_idx:end_idx+1]
                            plan_result = json.loads(json_str)
                            logger.info(f"✅ Extracted plan from braces with {len(plan_result.get('slide_outline', []))} slides")
                    except:
                        logger.warning(f"⚠️ Failed to parse plan text as JSON. Text length: {len(plan_text)}")
        else:
            # No existing plan, run planning agent to create one
            logger.info("📋 No existing plan found, running planning agent...")
            yield Event(
                author=self.name,
                content=types.Content(parts=[
                    types.Part(text=get_status_msg("creating_outline"))
                ]),
            )
            
            planning_agent = create_lightweight_planning_agent()
            
            # Run planning agent
            async for ev in planning_agent.run_async(ctx):
                # Forward planning events
                yield ev
                #print("EV_value", ev.content)
                # Extract plan from final response
                if ev.content and ev.content.parts:
                    for part in ev.content.parts:
                        if hasattr(part, 'text') and part.text:
                            plan_text = part.text
                            
                            # Method 1: Try direct JSON parsing
                            try:
                                plan_result = json.loads(plan_text)
                                logger.info(f"✅ Extracted plan directly with {len(plan_result.get('slide_outline', []))} slides")
                                continue
                            except json.JSONDecodeError:
                                pass
                            
                            # Method 2: Remove markdown code blocks
                            try:
                                # Remove ```json and ``` markers
                                cleaned_text = re.sub(r'^```json\s*', '', plan_text, flags=re.MULTILINE)
                                cleaned_text = re.sub(r'```\s*$', '', cleaned_text, flags=re.MULTILINE)
                                cleaned_text = cleaned_text.strip()
                                
                                plan_result = json.loads(cleaned_text)
                                logger.info(f"✅ Extracted plan after removing markdown with {len(plan_result.get('slide_outline', []))} slides")
                                continue
                            except json.JSONDecodeError:
                                pass
                            
                            # Method 3: Find JSON object in text
                            try:
                                json_match = re.search(r'\{[\s\S]*\}', plan_text)
                                if json_match:
                                    plan_result = json.loads(json_match.group(0))
                                    logger.info(f"✅ Extracted plan from regex match with {len(plan_result.get('slide_outline', []))} slides")
                                    continue
                            except:
                                pass
                            
                            # Method 4: Try to find JSON between braces
                            try:
                                start_idx = plan_text.find('{')
                                end_idx = plan_text.rfind('}')
                                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                                    json_str = plan_text[start_idx:end_idx+1]
                                    plan_result = json.loads(json_str)
                                    logger.info(f"✅ Extracted plan from braces with {len(plan_result.get('slide_outline', []))} slides")
                                    continue
                            except:
                                logger.warning(f"⚠️ Failed to parse plan text as JSON. Text length: {len(plan_text)}")
        
        # Validate plan result
        if not plan_result or 'slide_outline' not in plan_result:
            error_msg = "❌ Failed to generate valid slide outline"
            logger.error(error_msg)
            if plan_text:
                logger.error(f"📄 Plan text preview (first 500 chars): {plan_text[:500]}")
                logger.error(f"📄 Plan text preview (last 500 chars): {plan_text[-500:]}")
                logger.error(f"📄 Total plan text length: {len(plan_text)}")
            if plan_result:
                logger.error(f"📊 Parsed result keys: {list(plan_result.keys())}")
            
            yield Event(
                author=self.name,
                content=types.Content(parts=[types.Part(text=f"{error_msg}\n\nPlease check the logs for details. The planning agent may have returned text instead of JSON.")]),
                actions=EventActions(escalate=True),
            )
            return
        
        slide_outline = plan_result['slide_outline']
        global_theme = plan_result.get('global_theme', {})
        total_slides = len(slide_outline)
        
        logger.info(f"📊 Plan created: {total_slides} slides")
        
        # Store plan in session state
        ctx.session.state["slide_plan"] = plan_result
        
        # Store global_theme to presentations collection
        if global_theme:
            try:
                if p_id:
                    result = db.presentations.update_one(
                        {"p_id": p_id, "user_id": user_id},
                        {"$set": {
                            "global_theme": global_theme,
                            "updated_at": datetime.now(timezone.utc)
                        }}
                    )
                    
                    if result.modified_count > 0:
                        logger.info("✅ global_theme successfully stored in presentations collection")
                    else:
                        logger.info("ℹ️ global_theme already exists in presentations collection or no changes made")
                else:
                    logger.warning("⚠️ Cannot store global_theme: p_id not found in session state")
            except Exception as e:
                logger.error(f"❌ Failed to store global_theme to presentations collection: {e}")
        else:
            logger.warning("⚠️ No global_theme found in plan_result to store")
        
        # STEP 2: Generate Status Event
        yield Event(
            author=self.name,
            content=types.Content(parts=[
                types.Part(text=get_status_msg("outline_created", total_slides=total_slides))
            ]),
        )
        
        yield Event(
            author=self.name,
            content=types.Content(parts=[
                types.Part(text=get_status_msg("generating_slides", total_slides=total_slides))
            ]),
        )
        
        # STEP 3: Retrieve Templates and Select Best for Each Slide
        # Get presentation type from presentation_spec or plan metadata
        presentation_type = presentation_spec.get("presentation_type")
        if not presentation_type:
            # Fallback: try to get from plan_result metadata
            presentation_type = plan_result.get("presentation_metadata", {}).get("presentation_type", "regular_presentation")
            logger.info(f"📋 Using presentation_type from plan: {presentation_type}")
        else:
            logger.info(f"📋 Using presentation_type from presentation_spec: {presentation_type}")
        
        slide_generators = []
        # Track template info for each slide for database storage
        slide_template_info = {}  # {slide_index: template_info_dict}
        
        # Global logo cache to ensure consistency across all slides
        # global_logo_cache = {}
        logger.info(f"📋 Starting parallel template selection for {total_slides} slides...")
        
        # Async function to process a single slide's template selection
        async def process_slide_template(idx: int, slide_def: dict):
            """Process template selection for a single slide"""
            slide_title = slide_def.get('slide_title', f'Slide {idx+1}')
            slide_purpose = slide_def.get('slide_purpose', 'content')
            events = []
            
            logger.info(f"🔍 Processing slide {idx+1}/{total_slides}: {slide_title}")
            
            try:
                # Sub-step A: Retrieve available templates
                logger.info(f"  📥 Retrieving templates for purpose: {slide_purpose}")
                
                all_templates_text = retrieve_html_templates(
                    slide_purpose=slide_purpose,
                    presentation_type=presentation_type
                )
                
                # Check if templates were retrieved successfully
                if "Failed to retrieve templates" in all_templates_text or "No templates found" in all_templates_text:
                    logger.warning(f"  ⚠️ Template retrieval issue for slide {idx+1}: {all_templates_text[:100]}")
                    # Store fallback template info
                    slide_template_info[idx] = {
                        "template_id": None,
                        "template_category": "none",
                        "template_type": slide_purpose,
                        "selected_reasoning": "No template available - generated from scratch"
                    }
                    # Create generator without template (will use default HTML generation)
                    generator = create_enhanced_slide_generator(
                        slide_outline=slide_def,
                        global_theme=global_theme,
                        selected_template_html="<!-- No template available, generate from scratch -->",
                        idx=idx
                    )
                    return (idx, generator, "fallback", events)
                
                # Sub-step B: Create template selector agent
                logger.info(f"  🤖 Running template selector agent for slide {idx+1}...")
                template_selector = create_template_selector_agent(
                    slide_outline=slide_def,
                    all_templates=all_templates_text
                )
                
                # Sub-step C: Run template selector to get best template
                selected_template_html = None
                selection_result_text = ""
                
                async for ev in template_selector.run_async(ctx):
                    # Store the event to be yielded later by the main loop
                    events.append(ev)
                    
                    # Extract token counts from usage_metadata if available
                    # COMPLETED: implement token counting is already implemented in the EXECUTE Agent
                    # if hasattr(ev, 'usage_metadata') and ev.usage_metadata:
                    #     usage = ev.usage_metadata
                    #     input_tokens = 0
                    #     output_tokens = 0
                    #     thoughts_tokens = 0
                        
                    #     if hasattr(usage, 'prompt_token_count') and usage.prompt_token_count:
                    #         input_tokens = usage.prompt_token_count
                        
                    #     if hasattr(usage, 'candidates_token_count') and usage.candidates_token_count:
                    #         output_tokens = usage.candidates_token_count
                        
                    #     if hasattr(usage, 'thoughts_token_count') and usage.thoughts_token_count:
                    #         thoughts_tokens = usage.thoughts_token_count
                    #         output_tokens += thoughts_tokens
                        
                    #     # Store tokens in session state for later aggregation
                    #     if input_tokens > 0 or output_tokens > 0:
                    #         p_id = ctx.session.state.get('p_id', 'unknown')
                    #         state_exists = 'token_counts' in ctx.session.state
                            
                    #         if not state_exists:
                    #             msg = f"Initializing token_counts in ctx.session.state for p_id: {p_id}"
                    #             logger.info(f"📝 {msg}")
                    #             ctx.session.state['token_counts'] = {
                    #                 'input_tokens': 0,
                    #                 'output_tokens': 0,
                    #                 'thoughts_tokens': 0
                    #             }
                    #             # Log initialization to token auditor
                    #             log_token_usage(p_id=p_id, author=self.name, message=msg)
                            
                    #         # Update session state
                    #         ctx.session.state['token_counts']['input_tokens'] += input_tokens
                    #         ctx.session.state['token_counts']['output_tokens'] += output_tokens
                    #         ctx.session.state['token_counts']['thoughts_tokens'] += thoughts_tokens
                    #         msg=f"Updating token_counts in ctx.session.state for template selector p_id: {p_id}"
                    #         # Log to centralized token auditor (file)
                    #         log_token_usage(
                    #             p_id=p_id,
                    #             author=self.name,
                    #             input_tokens=input_tokens,
                    #             output_tokens=output_tokens,
                    #             thoughts_tokens=thoughts_tokens,
                    #             message=msg
                    #         )
                            
                    #         logger.debug(f"📊 Pipeline (Template) Token Usage ({self.name}): In={input_tokens}, Out={output_tokens}, StateExists={state_exists}")
                    
                    if ev.content and ev.content.parts:
                        for part in ev.content.parts:
                            if hasattr(part, 'text') and part.text:
                                selection_result_text = part.text
                                
                                # Parse the selection JSON
                                try:
                                    # Remove any markdown code blocks
                                    cleaned = re.sub(r'^```json\s*', '', selection_result_text, flags=re.MULTILINE)
                                    cleaned = re.sub(r'```\s*$', '', cleaned, flags=re.MULTILINE)
                                    cleaned = cleaned.strip()
                                    
                                    selection = json.loads(cleaned)
                                    selected_template_num = selection.get("selected_template_number")
                                    selected_template_id = selection.get("selected_template_id")
                                    selection_reasoning = selection.get("selection_reasoning", "N/A")
                                    
                                    logger.info(f"  ✅ Template {selected_template_num} (ID: {selected_template_id}) selected for slide {idx+1}")
                                    logger.info(f"     Reason: {selection_reasoning}")
                                    
                                    # Store template info for database storage
                                    slide_template_info[idx] = {
                                        "template_id": selected_template_id,
                                        "template_number": selected_template_num,
                                        "template_category": "business",
                                        "template_type": slide_purpose,
                                        "selected_reasoning": selection_reasoning
                                    }
                                    
                                    # Extract the selected template HTML from all_templates_text
                                    template_pattern = "## TEMPLATE " + str(selected_template_num) + ":.*?\\*\\*Complete HTML Code:\\*\\*\\n```html\\n(.*?)\\n```"
                                    match = re.search(template_pattern, all_templates_text, re.DOTALL)
                                    
                                    if match:
                                        selected_template_html = match.group(1)
                                        logger.info(f"  📄 Extracted template HTML ({len(selected_template_html)} chars)")
                                    else:
                                        logger.warning(f"  ⚠️ Could not extract template {selected_template_num} HTML")
                                        # Fallback: try to extract first template
                                        first_template_match = re.search(
                                            r"## TEMPLATE 1:.*?\*\*Complete HTML Code:\*\*\n```html\n(.*?)\n```",
                                            all_templates_text,
                                            re.DOTALL
                                        )
                                        if first_template_match:
                                            selected_template_html = first_template_match.group(1)
                                            logger.info(f"  📄 Using fallback template 1 ({len(selected_template_html)} chars)")
                                    
                                except json.JSONDecodeError as e:
                                    logger.error(f"  ❌ Failed to parse template selection: {e}")
                                    logger.error(f"     Selection text: {selection_result_text[:200]}")
                                    # Fallback: use first template
                                    first_template_match = re.search(
                                        r"## TEMPLATE 1:.*?\*\*Complete HTML Code:\*\*\n```html\n(.*?)\n```",
                                        all_templates_text,
                                        re.DOTALL
                                    )
                                    if first_template_match:
                                        selected_template_html = first_template_match.group(1)
                                        logger.warning(f"  ⚠️ Using fallback template 1 for slide {idx+1}")
                
                # Sub-step D: Create slide generator with selected template
                if selected_template_html:
                    generator = create_enhanced_slide_generator(
                        slide_outline=slide_def,
                        global_theme=global_theme,
                        selected_template_html=selected_template_html,
                        idx=idx
                    )
                    logger.info(f"  ✅ Slide generator created for slide {idx+1}")
                    return (idx, generator, "success", events)
                else:
                    logger.error(f"  ❌ No template selected for slide {idx+1}, using fallback")
                    # Store fallback template info
                    slide_template_info[idx] = {
                        "template_id": None,
                        "template_category": "none",
                        "template_type": slide_purpose,
                        "selected_reasoning": "Template selection failed - generated from scratch"
                    }
                    generator = create_enhanced_slide_generator(
                        slide_outline=slide_def,
                        global_theme=global_theme,
                        selected_template_html="<!-- Template selection failed, generate from scratch -->",
                        idx=idx
                    )
                    return (idx, generator, "fallback", events)
                    
            except Exception as e:
                logger.error(f"  ❌ Error processing slide {idx+1}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                # Return fallback generator on error
                slide_template_info[idx] = {
                    "template_id": None,
                    "template_category": "none",
                    "template_type": slide_def.get('slide_purpose', 'content'),
                    "selected_reasoning": f"Error during processing: {str(e)}"
                }
                generator = create_enhanced_slide_generator(
                    slide_outline=slide_def,
                    global_theme=global_theme,
                    selected_template_html="<!-- Error during template selection, generate from scratch -->",
                    idx=idx
                )
                return (idx, generator, "error", events)
        
        # Process all slides in parallel
        tasks = [process_slide_template(idx, slide_def) for idx, slide_def in enumerate(slide_outline)]
        results = await asyncio.gather(*tasks)
        
        # Sort results by index to maintain order
        results.sort(key=lambda x: x[0])
        
        # Build slide_generators list in order and yield events
        for idx, generator, status, slide_events in results:
            # Yield events collected during template selection
            for ev in slide_events:
                yield ev
            
            slide_generators.append(generator)
            
            if status == "success":
                yield Event(
                    author=self.name,
                    content=types.Content(parts=[
                        types.Part(text=get_status_msg("slide_ready", idx=idx+1, total=total_slides))
                    ]),
                )
            elif status == "fallback":
                yield Event(
                    author=self.name,
                    content=types.Content(parts=[
                        types.Part(text=get_status_msg("slide_fallback", idx=idx+1, total=total_slides))
                    ]),
                )
        
        logger.info(f"✅ Template selection complete for all {total_slides} slides")
        
        # Yield summary event
        # yield Event(
        #     author=self.name,
        #     content=types.Content(parts=[
        #         types.Part(text=f"✅ All {total_slides} templates selected. Starting slide generation...")
        #     ]),
        # )
        
        # STEP 4: Generate Slides in Controlled Batches to Avoid Connection Overload
        # Windows has issues with too many concurrent connections (WinError 64)
        # So we generate slides in small batches with delays between batches
        BATCH_SIZE = int(os.getenv("SLIDE_BATCH_SIZE", "3"))  # Generate slides in batches (default: 3)
        BATCH_DELAY = float(os.getenv("SLIDE_BATCH_DELAY", "0.5"))  # Delay between batches in seconds (default: 1.5)
        
        slide_count = 0
        generated_slides = []
        processed_generators = set()
        
        # Store slides with metadata for parallel validation
        slides_for_validation = []  # List of dicts: {index, html, outline, template_info, thinking_text}
        validation_tasks = {}  # Dict to track validation tasks: {index: task}
        
        # Process slides in batches
        for batch_start in range(0, len(slide_generators), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(slide_generators))
            batch_generators = slide_generators[batch_start:batch_end]
            batch_num = (batch_start // BATCH_SIZE) + 1
            total_batches = (len(slide_generators) + BATCH_SIZE - 1) // BATCH_SIZE
            
            logger.info(f"📦 Starting batch {batch_num}/{total_batches}: slides {batch_start+1}-{batch_end}")
            
            # yield Event(
            #     author=self.name,
            #     content=types.Content(parts=[
            #         types.Part(text=f"📦 Processing batch {batch_num}/{total_batches}: slides {batch_start+1}-{batch_end}...")
            #     ]),
            # )
            
            # Add delay between batches to avoid overwhelming APIs (except for first batch)
            if batch_start > 0:
                logger.info(f"⏸️ Waiting {BATCH_DELAY}s before next batch...")
                await asyncio.sleep(BATCH_DELAY)
            
            # Generate this batch in parallel
            batch_parallel = ParallelAgent(
                name=f"batch_generation_{batch_start}",
                sub_agents=batch_generators,
                description=f"Generating slides {batch_start+1} to {batch_end}"
            )
            
            async for ev in batch_parallel.run_async(ctx):
                # Extract token counts from usage_metadata if available (before processing)
                #COMPLETED: HAVE ALREADY ADDED THIS TO THE LIGHTWEIGHT SLIDE GENERATION AGENT
                # if hasattr(ev, 'usage_metadata') and ev.usage_metadata:
                #     usage = ev.usage_metadata
                #     input_tokens = 0
                #     output_tokens = 0
                #     thoughts_tokens = 0
                    
                #     if hasattr(usage, 'prompt_token_count') and usage.prompt_token_count:
                #         input_tokens = usage.prompt_token_count
                    
                #     if hasattr(usage, 'candidates_token_count') and usage.candidates_token_count:
                #         output_tokens = usage.candidates_token_count
                    
                #     # Store tokens in session state for later aggregation
                #     if input_tokens > 0 or output_tokens > 0:
                #         p_id = ctx.session.state.get('p_id', 'unknown')
                #         state_exists = 'token_counts' in ctx.session.state
                        
                #         if not state_exists:
                #             msg = f"Initializing token_counts in ctx.session.state for p_id: {p_id}"
                #             logger.info(f"📝 {msg}")
                #             ctx.session.state['token_counts'] = {
                #                 'input_tokens': 0,
                #                 'output_tokens': 0,
                #                 'thoughts_tokens': 0
                #             }
                #             # Log initialization to token auditor
                #             log_token_usage(p_id=p_id, author=self.name, message=msg)
                        
                #         # Update session state
                #         msg=f"Updating token_counts in ctx.session.state for slide generation p_id: {p_id}"
                #         ctx.session.state['token_counts']['input_tokens'] += input_tokens
                #         ctx.session.state['token_counts']['output_tokens'] += output_tokens
                #         ctx.session.state['token_counts']['thoughts_tokens'] += thoughts_tokens
                        
                #         # Log to centralized token auditor (file)
                #         log_token_usage(
                #             p_id=p_id,
                #             author=self.name,
                #             input_tokens=input_tokens,
                #             output_tokens=output_tokens,
                #             thoughts_tokens=thoughts_tokens,
                #             message=msg
                #         )
                        
                #         logger.debug(f"📊 Pipeline (Refining) Token Usage ({self.name}): In={input_tokens}, Out={output_tokens}, StateExists={state_exists}")
                    
                    # Track slide generation
                if ev.content and ev.content.parts:
                    # Extract HTML content and thinking text separately
                    html_text = None
                    thinking_text = None
                    has_html = False
                    
                    for part in ev.content.parts:
                        # Check if this part is thinking
                        if hasattr(part, 'thought') and part.thought:
                            thinking_text = part.text if hasattr(part, 'text') else None
                        # Check if this is HTML content
                        elif hasattr(part, 'text') and part.text:
                            text = part.text
                            # Check if this is HTML output from a slide generator
                            if ('<!DOCTYPE html>' in text or '<html' in text or '```html' in text):
                                html_text = text
                                has_html = True
                    
                    # Process HTML output if found
                    if html_text and ev.author.startswith('enhanced_slide_generator'):
                        # Extract generator index from author name
                        try:
                            generator_idx = int(ev.author.split('_')[-1])
                            if generator_idx not in processed_generators:
                                slide_count += 1
                                generated_slides.append(html_text)
                                processed_generators.add(generator_idx)
                                
                                # Save slide to database
                                p_id = ctx.session.state.get("p_id")
                                user_id = ctx.session.state.get("user_id", "unknown")
                                
                                # Get slide outline and template info
                                slide_def = slide_outline[generator_idx] if generator_idx < len(slide_outline) else {}
                                template_info = slide_template_info.get(generator_idx, {
                                    "template_id": None,
                                    "template_category": "unknown",
                                    "template_type": "unknown",
                                    "selected_reasoning": "No template info available"
                                })
                                
                                # Store slide data with original HTML
                                slide_data = {
                                    "index": generator_idx,
                                    "html": html_text,
                                    "original_html": html_text,  # Store original before validation
                                    "outline": slide_def,
                                    "template_info": template_info,
                                    "thinking_text": thinking_text,
                                    "screenshot_path": None,
                                    "original_screenshot_path": None,
                                    "validation_iterations_count": 0  # Will be updated after validation
                                }
                                slides_for_validation.append(slide_data)
                                
                                # Take screenshot of original HTML before validation
                                if ENABLE_SLIDE_VALIDATION:
                                    from .slide_validation_agent import take_slide_screenshot as take_screenshot
                                    original_screenshot = await take_screenshot(html_text, p_id, generator_idx)
                                    if original_screenshot:
                                        # Store with _original suffix to distinguish
                                        from pathlib import Path
                                        screenshots_dir = Path(os.getenv("SCREENSHOTS_DIR", "screenshots"))
                                        original_screenshot_path = screenshots_dir / p_id / f"slide_{generator_idx}_original.png"
                                        # Rename the screenshot to include _original
                                        if Path(original_screenshot).exists():
                                            import shutil
                                            shutil.move(original_screenshot, str(original_screenshot_path))
                                            slide_data["original_screenshot_path"] = str(original_screenshot_path)
                                    else:
                                        slide_data["original_screenshot_path"] = original_screenshot
                                
                                # If validation is enabled, only yield thinking_text (not HTML)
                                # HTML will be yielded after validation completes
                                if ENABLE_SLIDE_VALIDATION:
                                    # Only yield thinking_text if it exists
                                    if thinking_text:
                                        yield Event(
                                            author=ev.author,
                                            content=types.Content(parts=[
                                                types.Part(text=thinking_text, thought=True)
                                            ]),
                                        )
                                    # Don't yield the HTML event - it will be yielded after validation
                                else:
                                    # If validation is disabled, yield the original event as-is
                                    yield ev
                                
                                # Yield progress
                                # yield Event(
                                #     author=self.name,
                                #     content=types.Content(parts=[
                                #         types.Part(text=f"✅ Generated slide {slide_count}/{total_slides}")
                                #     ]),
                                # )
                                logger.info(f"✅ Generated slide {slide_count}/{total_slides} from generator {generator_idx}")
                        except (ValueError, IndexError):
                            # Fallback: if we can't extract index, still count it
                            if len(generated_slides) < total_slides and html_text:
                                # Use current slide count as fallback index
                                fallback_idx = len(generated_slides)
                                slide_count += 1
                                generated_slides.append(html_text)
                                
                                # Get slide outline and template info with fallback
                                slide_def = slide_outline[fallback_idx] if fallback_idx < len(slide_outline) else {}
                                template_info = slide_template_info.get(fallback_idx, {
                                    "template_id": None,
                                    "template_category": "unknown",
                                    "template_type": "unknown",
                                    "selected_reasoning": "Fallback save - index not extracted"
                                })
                                
                                # Store slide data with original HTML
                                slide_data = {
                                    "index": fallback_idx,
                                    "html": html_text,
                                    "original_html": html_text,  # Store original before validation
                                    "outline": slide_def,
                                    "template_info": template_info,
                                    "thinking_text": thinking_text,
                                    "screenshot_path": None,
                                    "original_screenshot_path": None,
                                    "validation_iterations_count": 0  # Will be updated after validation
                                }
                                slides_for_validation.append(slide_data)
                                
                                # Take screenshot of original HTML (validation will start after all slides are generated)
                                if ENABLE_SLIDE_VALIDATION:
                                    from .slide_validation_agent import take_slide_screenshot as take_screenshot
                                    original_screenshot = await take_screenshot(html_text, p_id, fallback_idx)
                                    if original_screenshot:
                                        # Store with _original suffix to distinguish
                                        from pathlib import Path
                                        screenshots_dir = Path(os.getenv("SCREENSHOTS_DIR", "screenshots"))
                                        original_screenshot_path = screenshots_dir / p_id / f"slide_{fallback_idx}_original.png"
                                        # Rename the screenshot to include _original
                                        if Path(original_screenshot).exists():
                                            import shutil
                                            shutil.move(original_screenshot, str(original_screenshot_path))
                                            slide_data["original_screenshot_path"] = str(original_screenshot_path)
                                        else:
                                            slide_data["original_screenshot_path"] = original_screenshot
                                
                                # If validation is enabled, only yield thinking_text (not HTML)
                                # HTML will be yielded after validation completes
                                if ENABLE_SLIDE_VALIDATION:
                                    # Only yield thinking_text if it exists
                                    if thinking_text:
                                        yield Event(
                                            author=ev.author,
                                            content=types.Content(parts=[
                                                types.Part(text=thinking_text, thought=True)
                                            ]),
                                        )
                                    # Don't yield the HTML event - it will be yielded after validation
                                else:
                                    # If validation is disabled, yield the original event as-is
                                    yield ev
                                
                                # Yield progress
                                yield Event(
                                    author=self.name,
                                    content=types.Content(parts=[
                                        types.Part(text=get_status_msg("slide_generated", count=slide_count, total=total_slides))
                                    ]),
                                )
                                logger.info(f"✅ Generated slide {slide_count}/{total_slides} (fallback detection)")
                    else:
                        # For non-HTML events or non-enhanced_slide_generator events, yield as normal
                        yield ev
        
        # STEP 5: Start validation after all slides are generated (if validation was enabled)
        if ENABLE_SLIDE_VALIDATION and slides_for_validation:
            logger.info(f"✅ All {len(slides_for_validation)} slides generated. Starting validation...")
            
            yield Event(
                author=self.name,
                content=types.Content(parts=[
                    types.Part(text=get_status_msg("validating_slides", count=len(slides_for_validation)))
                ]),
            )
            
            # Extract session state from ctx to pass to worker processes
            session_state = ctx.session.state.copy() if hasattr(ctx, 'session') and hasattr(ctx.session, 'state') else {"p_id": p_id}
            
            # Prepare validation arguments for multiprocessing
            validation_args = []
            slide_data_map = {}
            for slide_data in slides_for_validation:
                slide_idx = slide_data["index"]
                html_text = slide_data["html"]
                
                # Store slide_data for later use
                slide_data_map[slide_idx] = slide_data
                
                # Prepare arguments for worker function (including session_state from ctx)
                validation_args.append((
                    html_text,
                    p_id,
                    slide_idx,
                    SLIDE_VALIDATION_RETRIES,
                    SLIDE_VALIDATION_MAX_ITERATIONS,
                    session_state  # Pass session state from ctx
                ))
            
            # Use multiprocessing to validate slides in parallel
            total_validation_tasks = len(validation_args)
            completed_count = 0
            
            # Determine number of worker processes (use CPU count, but cap at number of slides and MAX_VALIDATION_WORKERS)
            num_workers = min(multiprocessing.cpu_count(), total_validation_tasks, MAX_VALIDATION_WORKERS)
            
            logger.info(f"🚀 Starting multiprocessing validation with {num_workers} workers (max {MAX_VALIDATION_WORKERS}) for {total_validation_tasks} slides")
            
            # Use ProcessPoolExecutor for parallel processing
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                # Submit all validation tasks
                future_to_slide_idx = {
                    executor.submit(_validate_slide_worker, args): args[2]  # args[2] is slide_idx (6th element is session_state)
                    for args in validation_args
                }
                
                # Process results as they complete (yield immediately when each slide is done)
                for future in as_completed(future_to_slide_idx):
                    completed_count += 1
                    slide_idx = future_to_slide_idx[future]
                    slide_data = slide_data_map[slide_idx]
                    
                    try:
                        result = future.result()
                        slide_idx_result, fixed_html, validation_thinking, validation_iterations_count, error = result
                        
                        # Update slide data
                        original_html = slide_data["html"]
                        # Ensure we have valid HTML - use original if fixed_html is None or empty
                        final_html = fixed_html if fixed_html and fixed_html.strip() else original_html
                        slide_data["html"] = final_html
                        slide_data["validation_iterations_count"] = validation_iterations_count
                        
                        if validation_thinking:
                            if slide_data["thinking_text"]:
                                slide_data["thinking_text"] = f"{slide_data['thinking_text']}\n\n--- Validation ---\n{validation_thinking}"
                            else:
                                slide_data["thinking_text"] = validation_thinking
                        
                        # Get screenshot path
                        from pathlib import Path
                        screenshots_dir = Path(os.getenv("SCREENSHOTS_DIR", "screenshots"))
                        screenshot_path = screenshots_dir / p_id / f"slide_{slide_data['index']}.png"
                        screenshot_path = str(screenshot_path) if screenshot_path.exists() else None
                        slide_data["screenshot_path"] = screenshot_path
                        
                        # Yield event based on result - ALWAYS yield HTML to ensure all slides are sent
                        if error:
                            yield Event(
                                author=self.name,
                                content=types.Content(parts=[
                                    types.Part(text=get_status_msg("validation_failed", idx=slide_idx + 1, done=completed_count, total=total_validation_tasks))
                                ]),
                            )
                            # Still yield the original HTML (even if validation failed)
                            yield Event(
                                author=f"enhanced_slide_generator_{slide_data['index']}",
                                content=types.Content(parts=[
                                    types.Part(text=original_html)
                                ]),
                            )
                        elif fixed_html and fixed_html.strip() and fixed_html != original_html:
                            # Yield status message
                            yield Event(
                                author=self.name,
                                content=types.Content(parts=[
                                    types.Part(text=get_status_msg("validation_fixed", idx=slide_data['index'] + 1, done=completed_count, total=total_validation_tasks)),
                                    types.Part(text=validation_thinking if validation_thinking else "", thought=True)
                                ]),
                            )
                            # Yield validated HTML through socket (using enhanced_slide_generator author with index so it gets processed correctly)
                            yield Event(
                                author=f"enhanced_slide_generator_{slide_data['index']}",
                                content=types.Content(parts=[
                                    types.Part(text=fixed_html)
                                ]),
                            )
                        else:
                            # Case: fixed_html is None/empty OR fixed_html == original_html
                            # Always yield HTML (use final_html which falls back to original_html if needed)
                            # Yield status message
                            # yield Event(
                            #     author=self.name,
                            #     content=types.Content(parts=[
                            #         types.Part(text=f"✅ Slide {slide_data['index'] + 1} validation complete (no changes needed) ({completed_count}/{total_validation_tasks})")
                            #     ]),
                            # )
                            # Still yield the HTML (even if no changes, to ensure it's sent through socket)
                            yield Event(
                                author=f"enhanced_slide_generator_{slide_data['index']}",
                                content=types.Content(parts=[
                                    types.Part(text=final_html)  # Use final_html which ensures we always have valid HTML
                                ]),
                            )
                        
                        logger.info(f"✅ Slide {slide_data['index'] + 1} validation completed ({completed_count}/{total_validation_tasks})")
                        
                    except Exception as e:
                        logger.error(f"❌ Error processing validation result for slide {slide_idx + 1}: {e}")
                        # Use original HTML on error
                        slide_data["validation_iterations_count"] = 0
                        yield Event(
                            author=self.name,
                            content=types.Content(parts=[
                                types.Part(text=get_status_msg("validation_error", idx=slide_idx + 1, done=completed_count, total=total_validation_tasks))
                            ]),
                        )
                        # Still yield the original HTML (even if validation error occurred)
                        yield Event(
                            author=f"enhanced_slide_generator_{slide_data['index']}",
                            content=types.Content(parts=[
                                types.Part(text=slide_data["html"])  # Use original HTML from slide_data
                            ]),
                        )
                        continue
            
            logger.info(f"✅ All {total_validation_tasks} validation tasks completed using multiprocessing")
        
        # STEP 6: Save all slides to database
        p_id = ctx.session.state.get("p_id")
        user_id = ctx.session.state.get("user_id", "unknown")
        
        for slide_data in slides_for_validation:
            save_success = save_slide_to_database(
                p_id=p_id,
                user_id=user_id,
                slide_index=slide_data["index"],
                slide_html=slide_data["html"],  # Validated HTML
                slide_outline=slide_data["outline"],
                template_info=slide_data["template_info"],
                global_theme=global_theme,
                thinking_text=slide_data["thinking_text"],
                screenshot_path=slide_data.get("screenshot_path"),  # Validated screenshot
                original_html=slide_data.get("original_html"),  # Original HTML before validation
                original_screenshot_path=slide_data.get("original_screenshot_path"),  # Original screenshot
                validation_iterations_count=slide_data.get("validation_iterations_count", 0)  # Val692689135d3a7c4d54922b4didation iteration count
            )
            
            if save_success:
                logger.info(f"💾 Saved slide {slide_data['index'] + 1}/{total_slides} to database")
            else:
                logger.warning(f"⚠️ Failed to save slide {slide_data['index'] + 1}/{total_slides} to database")
        
        # Update generated_slides list with validated HTML
        generated_slides = [slide_data["html"] for slide_data in sorted(slides_for_validation, key=lambda x: x["index"])]
        
        # STEP 7: Final Check and Store Generated Slides
        if len(generated_slides) < total_slides:
            missing_slides = total_slides - len(generated_slides)
            logger.warning(f"⚠️ Missing {missing_slides} slides. Expected {total_slides}, got {len(generated_slides)}")
            
            # Try to find any remaining HTML content in the events
            yield Event(
                author=self.name,
                content=types.Content(parts=[
                    types.Part(text=get_status_msg("missing_slides", count=len(generated_slides), total=total_slides))
                ]),
            )
        
        ctx.session.state["generated_slides"] = generated_slides
        ctx.session.state["total_slides"] = len(generated_slides)
        
        logger.info(f"🎉 Completed: {len(generated_slides)} slides generated")
        
        # STEP 8: Cleanup - Remove all screenshots for this p_id after all tasks are completed
        # Safe cleanup for multi-worker environments (Gunicorn)
        try:
            import shutil
            import time
            screenshots_dir = Path(os.getenv("SCREENSHOTS_DIR", "screenshots"))
            p_id_screenshot_dir = screenshots_dir / p_id
            
            if p_id_screenshot_dir.exists() and p_id_screenshot_dir.is_dir():
                # Wait a short time to ensure no other processes are still writing files
                # This helps avoid race conditions in multi-worker environments
                await asyncio.sleep(1)
                
                # Check if any files were modified in the last 3 seconds (safety check)
                current_time = time.time()
                recent_files = []
                try:
                    for file_path in p_id_screenshot_dir.iterdir():
                        if file_path.is_file():
                            file_mtime = file_path.stat().st_mtime
                            # If file was modified in last 3 seconds, it might still be in use
                            if current_time - file_mtime < 3:
                                recent_files.append(file_path.name)
                except (OSError, PermissionError):
                    # If we can't check files, skip cleanup to be safe
                    logger.warning(f"⚠️ Cannot access screenshot directory for p_id {p_id}, skipping cleanup")
                    print(f"⚠️ Cannot access screenshot directory, skipping cleanup")
                    recent_files = ["unknown"]  # Force skip
                
                # Only delete if no recent file activity (safe to delete)
                if not recent_files:
                    try:
                        # Remove entire directory with all screenshots
                        shutil.rmtree(p_id_screenshot_dir, ignore_errors=True)
                        logger.info(f"🗑️ Cleaned up all screenshots for p_id: {p_id}")
                        print(f"🗑️ Cleaned up all screenshots for p_id: {p_id}")
                    except (OSError, PermissionError) as delete_error:
                        # Another process might be using files, that's okay
                        logger.warning(f"⚠️ Could not delete screenshot directory for p_id {p_id}: {delete_error}")
                        print(f"⚠️ Could not delete screenshots (may be in use by another process)")
                else:
                    logger.info(f"ℹ️ Skipping cleanup for p_id {p_id}: recent file activity detected ({len(recent_files)} files)")
                    print(f"ℹ️ Skipping cleanup: files may still be in use")
            else:
                logger.info(f"ℹ️ No screenshot directory found for p_id: {p_id} (nothing to clean up)")
        except Exception as e:
            logger.error(f"⚠️ Error cleaning up screenshots for p_id {p_id}: {e}")
            # Don't fail the entire process if cleanup fails
            print(f"⚠️ Warning: Could not clean up screenshots: {e}")
        
        # Final status
        if len(generated_slides) == total_slides:
            yield Event(
                author=self.name,
                content=types.Content(parts=[
                    types.Part(text=get_status_msg("complete_success", count=len(generated_slides)))
                ]),
            )
        else:
            yield Event(
                author=self.name,
                content=types.Content(parts=[
                    types.Part(text=get_status_msg("partial_success", count=len(generated_slides), total=total_slides))
                ]),
            )


def create_lightweight_slide_generation_agent():
    """Factory function to create the lightweight slide generation agent"""
    return LightweightSlideGenerationAgent(name="LightweightSlideGeneration")

