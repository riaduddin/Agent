"""
Slide Insertion Orchestrator
Main coordination agent for inserting slides between existing slides
"""
from datetime import datetime, timezone
from typing import Optional
from google.adk.agents import LlmAgent
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools import FunctionTool
from google.adk.sessions import InMemorySessionService, DatabaseSessionService
from google.adk.runners import Runner
from google.genai import types
import asyncio
from .insertion_request_parser import create_insertion_request_parser
from .database_tools import (
    validate_insertion_position_tool,
    fetch_presentation_context_tool,
    fetch_neighboring_slides_tool,
    save_presentation_outline_tool,
    save_global_theme_to_presentation_tool
)
# Reuse slide read operation from modification module for inspection of existing slides
from root_agent.slide_modification_agent_v2.database_tools import fetch_slide_data_tool
from .slide_renumbering_tool import renumber_slides_after_insertion_tool
from .content_generator import create_content_generator, generate_slide_plan_tool
import json
import sys
from pathlib import Path
import os
import logging
from ..sub_agents import get_model_with_fallback

logger = logging.getLogger(__name__)

def get_utc_timestamp_iso() -> str:
    """
    Get current UTC timestamp in ISO 8601 format with timezone info.
    Returns format like: '2024-01-15T10:30:45.123456+00:00'
    """
    return datetime.now(timezone.utc).isoformat()

# Add root directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
# Import template selector and slide generator (use root_agent prefix)
from root_agent.slide_creation_agent.sub_agents.template_selector_agent import create_template_selector_agent
from root_agent.slide_creation_agent.sub_agents.enhanced_slide_generator import create_enhanced_slide_generator
# Import slide validation agent
from root_agent.slide_creation_agent.sub_agents.slide_validation_agent import validate_and_fix_slide
# Import template retrieval tool
from tools.template_retrieval import retrieve_html_templates
# Note: Google search and Qdrant tools are now handled by the research agent sub-agent


# Database connection - import from db.py
try:
    from core.database import get_mongo_client
    client = get_mongo_client()
    db = client["slide_creator_db"]
except ImportError:
    # Fallback if db.py import fails
    import pymongo
    mongo_uri = os.getenv("MONGODB_URL")
    if not mongo_uri:
        raise ValueError("MONGODB_URL environment variable not set")
    client = pymongo.MongoClient(mongo_uri)
    db = client["slide_creator_db"]


def save_slide_to_database(
    p_id: str,
    user_id: str,
    slide_index: int,
    slide_html: str,
    slide_outline: dict,
    template_info: dict,
    global_theme: dict,
    thinking_text: str
    ) -> str:
    """
    Save generated slide to database. 
    
    **This is the FINAL and MANDATORY step after generate_slide_html_sync.**
    
    You MUST call this function after generate_slide_html_sync to complete the slide insertion task.
    The slide is NOT saved until this function is called successfully.
    
    Parameters:
    - p_id: Presentation ID
    - user_id: User ID
    - slide_index: 0-based slide index (from insert_position)
    - slide_html: HTML content from generate_slide_html_sync result (extract "generated_html" field)
    - slide_outline: Slide plan/outline (from generate_slide_plan)
    - template_info: Template information (from select_template_for_slide result)
    - global_theme: Global theme (from fetch_presentation_context)
    - thinking_text: Thinking/reasoning text from generate_slide_html_sync result (extract "thinking" field, or None)
    
    Returns JSON string with success status and slide_id.
    """
    print(f"💾 STEP 7: Saving slide to database at position {slide_index}")
    try:
        from bs4 import BeautifulSoup
        from datetime import datetime, timezone
        
        # Parse HTML content to remove markdown code blocks
        if slide_html.startswith("```html"):
            html_content = slide_html.replace("```html", "").replace("```", "").strip()
        else:
            html_content = slide_html
        
        # Parse with BeautifulSoup for content extraction
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract content metadata
        extracted_title = ""
        title_element = soup.find(['h1', 'h2'])
        if title_element:
            extracted_title = title_element.get_text(strip=True)
        
        extracted_headings = [h.get_text(strip=True) for h in soup.find_all(['h1', 'h2', 'h3', 'h4'])]
        extracted_paragraphs = [p.get_text(strip=True) for p in soup.find_all('p') if p.get_text(strip=True)]
        
        # Check for charts and lists
        has_charts = bool(soup.find('canvas'))
        has_lists = bool(soup.find(['ul', 'ol']))
        
        # Extract icon names
        icon_names = []
        for img in soup.find_all('img'):
            src = img.get('src', '')
            if 'icon' in src.lower() or 'logo' in src.lower():
                icon_names.append(src)
        
        # Calculate text length
        text_length = len(soup.get_text())
        
        # Create content metadata
        content_metadata = {
            "extracted_title": extracted_title,
            "extracted_headings": extracted_headings,
            "extracted_paragraphs": extracted_paragraphs,
            "has_charts": has_charts,
            "has_lists": has_lists,
            "icon_names": list(set(icon_names)),
            "text_length": text_length
        }
        
        # Compute final insertion position based on current numbering (ensure gap fill after renumbering)
        try:
            existing_numbers = set()
            for s in db.slide_html.find({"p_id": p_id}, {"slide_number": 1, "_id": 0}):
                num = s.get("slide_number")
                if isinstance(num, int):
                    existing_numbers.add(num)
            # Find smallest missing positive integer position (gap created by renumbering)
            computed_slide_number = 1
            while computed_slide_number in existing_numbers:
                computed_slide_number += 1
            computed_slide_index = computed_slide_number - 1
        except Exception:
            # Fallback to caller-provided index
            computed_slide_index = slide_index
            computed_slide_number = slide_index + 1

        # Create slide document
        slide_doc = {
            "p_id": p_id,
            "user_id": user_id,
            "slide_index": computed_slide_index,
            "slide_number": computed_slide_number,
            "body": html_content,
            "thinking": thinking_text if thinking_text else None,
            "slide_plan": slide_outline,
            "template_info": template_info,
            "content_metadata": content_metadata,
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
            "timestamp": datetime.now(timezone.utc)
        }
        
        # Insert into database
        result = db.slide_html.insert_one(slide_doc)
        
        if result.inserted_id:
            # Also update presentation outline and global theme
            try:
                print(f"📝 STEP 8: Updating presentation outline and global theme")
                
                # Update presentation outline
                from .database_tools import save_presentation_outline, save_global_theme_to_presentation
                
                # Get current slides to build outline
                slides = list(db.slide_html.find({"p_id": p_id}).sort("slide_number", 1))
                outline = []
                for slide in slides:
                    outline.append({
                        "slide_number": slide.get("slide_number"),
                        "slide_title": slide.get("slide_title", ""),
                        "slide_purpose": slide.get("slide_plan", {}).get("slide_purpose", "content")
                    })
                
                # Save outline
                outline_result = save_presentation_outline(p_id, json.dumps(outline))
                print(f"✅ Presentation outline updated")
                
                # Save global theme
                theme_result = save_global_theme_to_presentation(p_id, json.dumps(global_theme))
                print(f"✅ Global theme updated")
                
            except Exception as e:
                print(f"⚠️ Warning: Could not update outline/theme: {e}")
            
            result_data = json.dumps({
                "success": True,
                "slide_id": str(result.inserted_id),
                "slide_number": computed_slide_number
            })
            print(f"✅ Slide saved successfully to database")
            return result_data
        else:
            print(f"❌ Failed to insert slide into database")
            return json.dumps({
                "success": False,
                "error": "Failed to insert slide into database"
            })
            
    except Exception as e:
        print(f"❌ Error saving slide: {e}")
        return json.dumps({
            "success": False,
            "error": f"Error saving slide: {str(e)}"
        })


# Create function tool
save_slide_to_database_tool = FunctionTool(save_slide_to_database)

async def _select_template_for_slide_async(slide_plan_json: str, presentation_type: str, p_id: Optional[str] = None) -> str:
    import json, re

    slide_plan = json.loads(slide_plan_json)
    slide_purpose = slide_plan.get("slide_purpose", "content")

    all_templates_text = retrieve_html_templates(
        slide_purpose=slide_purpose,
        presentation_type=presentation_type
    )

    if "Failed to retrieve templates" in all_templates_text or "No templates found" in all_templates_text:
        return json.dumps({
            "error": f"Template retrieval failed: {all_templates_text}",
            "selected_template_number": None,
            "template_html": None
        })

    # Create your template selector agent
    template_selector_agent = create_template_selector_agent(
        slide_outline=slide_plan,
        all_templates=all_templates_text
    )

    # Create session infrastructure
    session_service = InMemorySessionService()
    user_id = "template_selector_user"   # you may vary
    session_id = "session_template_select"
    session = await session_service.create_session(
        app_name="template_selector_app",
        user_id=user_id,
        session_id=session_id
    )

    runner = Runner(
        agent=template_selector_agent,
        app_name="template_selector_app",
        session_service=session_service
    )

    # Prepare user message to kick off agent
    # The content should be of type google.genai.types.Content etc.
    content = types.Content(role="user", parts=[ types.Part(text="Select the best template based on slide plan.") ])

    selection_result_text = None
    template_selector_input_tokens = 0
    template_selector_output_tokens = 0
    template_selector_thoughts_tokens = 0
    
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=content
        ):
        # Extract token counts from event usage_metadata
        if hasattr(event, 'usage_metadata') and event.usage_metadata:
            input_tokens = getattr(event.usage_metadata, 'prompt_token_count', 0) or 0
            output_tokens = getattr(event.usage_metadata, 'candidates_token_count', 0) or 0
            thoughts_tokens = getattr(event.usage_metadata, 'thoughts_token_count', 0) or 0
            
            if input_tokens > 0 or output_tokens > 0:
                template_selector_input_tokens += input_tokens
                template_selector_output_tokens += output_tokens
                template_selector_thoughts_tokens += thoughts_tokens
        
        if event.is_final_response() and event.content and event.content.parts:
            selection_result_text = event.content.parts[0].text
            break
    
    # Store tokens in db.presentations (use p_id parameter or extract from slide_plan)
    if template_selector_input_tokens > 0 or template_selector_output_tokens > 0:
        try:
            # Use p_id parameter if provided, otherwise try to extract from slide_plan
            if not p_id:
                try:
                    slide_plan_dict = json.loads(slide_plan_json) if isinstance(slide_plan_json, str) else slide_plan_json
                    p_id = slide_plan_dict.get("p_id")
                except:
                    pass
            
            if p_id:
                # Get current token counts from presentations collection
                presentation = db.presentations.find_one({"p_id": p_id})
                if presentation:
                    current_token_counts = presentation.get("token_counts", {})
                    current_input = current_token_counts.get("input_tokens", 0)
                    current_output = current_token_counts.get("output_tokens", 0)
                    current_thoughts = current_token_counts.get("thoughts_tokens", 0)
                    
                    # Calculate new totals
                    new_input = current_input + template_selector_input_tokens
                    new_output = current_output + template_selector_output_tokens
                    new_thoughts = current_thoughts + template_selector_thoughts_tokens
                    
                    # Update presentations collection with new token counts
                    db.presentations.update_one(
                        {"p_id": p_id},
                        {"$set": {
                            "token_counts": {
                                "input_tokens": new_input,
                                "output_tokens": new_output,
                                "thoughts_tokens": new_thoughts
                            },
                            "updated_at": get_utc_timestamp_iso()
                        }}
                    )
                    logger.debug(f"📊 Stored template selector tokens: {template_selector_input_tokens} input, {template_selector_output_tokens} output in db.presentations for p_id={p_id}")
                else:
                    logger.warning(f"⚠️ Presentation not found for p_id={p_id}, cannot store template selector tokens")
            else:
                logger.debug(f"⚠️ p_id not found in slide_plan, skipping template selector token storage")
        except Exception as e:
            logger.warning(f"⚠️ Could not store template selector tokens in db.presentations: {e}")

    if selection_result_text:
        cleaned = re.sub(r'^```json\s*', '', selection_result_text, flags=re.MULTILINE)
        cleaned = re.sub(r'```\s*$', '', cleaned, flags=re.MULTILINE)
        cleaned = cleaned.strip()
        try:
            selection = json.loads(cleaned)
            selected_template_num = selection.get("selected_template_number")
            selected_template_id = selection.get("selected_template_id")
            selection_reasoning = selection.get("selection_reasoning", "N/A")

            template_pattern = f"## TEMPLATE {selected_template_num}:.*?\\*\\*Complete HTML Code:\\*\\*\\n```html\\n(.*?)\\n```"
            match = re.search(template_pattern, all_templates_text, re.DOTALL)
            if match:
                selected_template_html = match.group(1)
                return json.dumps({
                    "success": True,
                    "selected_template_number": selected_template_num,
                    "selected_template_id": selected_template_id or f"template_{selected_template_num}",
                    "template_html": selected_template_html,
                    "reasoning": selection_reasoning
                })
        except json.JSONDecodeError:
            # fallback to first template
            pass

    # fallback extraction of first template
    match = re.search(
        r"## TEMPLATE 1:.*?\*\*Complete HTML Code:\*\*\n```html\n(.*?)\n```",
        all_templates_text,
        re.DOTALL
    )
    if match:
        selected_template_html = match.group(1)
        id_match = re.search(r"## TEMPLATE 1:.*?\*\*ID:\*\* (.+?)\n", all_templates_text, re.DOTALL)
        template_id = id_match.group(1).strip() if id_match else "template_1"
        return json.dumps({
            "success": True,
            "selected_template_number": 1,
            "selected_template_id": template_id,
            "template_html": selected_template_html,
            "reasoning": "Selected first available template as fallback"
        })
    else:
        return json.dumps({
            "error": "Could not extract template HTML from retrieved templates",
            "selected_template_number": None,
            "template_html": None
        })


async def generate_slide_html(slide_plan_json: str, global_theme_json: str, template_html: str, p_id: str, user_id: str) -> str:
    """
    Generate HTML for a slide using the enhanced slide generator
    This runs the slide generator agent to produce actual HTML
    """
    print(f"⚡ STEP 6: Generating slide HTML")
    try:
        import asyncio
        print("slide_plan_json:", slide_plan_json)
        print("global_them_json:", global_theme_json)
        # Parse inputs
        slide_plan = json.loads(slide_plan_json)
        global_theme = json.loads(global_theme_json)
        
        # Create slide generator agent
        generator = create_enhanced_slide_generator(
            slide_outline=slide_plan,
            global_theme=global_theme,
            selected_template_html=template_html,
            idx=0
        )
        
        # Use the same Runner and session pattern as select_template_for_slide
        session_service = InMemorySessionService()
        session_id = "session_slide_generate"
        session = await session_service.create_session(
            app_name="slide_generator_app",
            user_id=user_id,
            session_id=session_id,
            state={
                "user_id": user_id,
                "p_id": p_id,
                "enhanced_query": f"Generate HTML for slide: {slide_plan.get('slide_title', 'New Slide')}"
            }
        )
        
        runner = Runner(
            agent=generator,
            app_name="slide_generator_app",
            session_service=session_service
        )
        
        # Prepare user message
        content = types.Content(role="user", parts=[types.Part(text=f"Generate HTML for slide: {slide_plan.get('slide_title', 'New Slide')}")])
        
        # Run the generator and collect both thinking text and HTML content
        # We'll collect ALL parts and separate them into thinking and HTML
        generated_html = ""
        thinking_text = ""
        
        # Token tracking for slide generator
        slide_gen_input_tokens = 0
        slide_gen_output_tokens = 0
        slide_gen_thoughts_tokens = 0
        
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=content
        ):
            # Extract token counts from usage_metadata if available
            if hasattr(event, 'usage_metadata') and event.usage_metadata:
                usage = event.usage_metadata
                if hasattr(usage, 'prompt_token_count') and usage.prompt_token_count:
                    slide_gen_input_tokens += usage.prompt_token_count
                if hasattr(usage, 'candidates_token_count') and usage.candidates_token_count:
                    slide_gen_output_tokens += usage.candidates_token_count
                if hasattr(usage, 'thoughts_token_count') and usage.thoughts_token_count:
                    slide_gen_thoughts_tokens += usage.thoughts_token_count
            
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, 'text') and part.text:
                        text = part.text
                        print(f"📝 Part text received (length: {len(text)}): {text[:200]}...")  # Print first 200 chars
                        
                        # Check if it's HTML content
                        text_stripped = text.strip()
                        is_html = (
                            text_stripped.startswith("<!DOCTYPE html") or 
                            text_stripped.startswith("<!DOCTYPE HTML") or
                            text_stripped.startswith("<!doctype html") or
                            text_stripped.startswith("<html") or
                            text_stripped.startswith("<HTML") or
                            '```html' in text
                        )
                        
                        if is_html:
                            # This is HTML content - extract and keep (overwrite if multiple HTML parts, keep the last complete one)
                            if '```html' in text:
                                # Extract HTML from markdown code block
                                start_idx = text.find('```html') + 7
                                end_idx = text.find('```', start_idx)
                                if end_idx != -1:
                                    html_extracted = text[start_idx:end_idx].strip()
                                else:
                                    html_extracted = text[start_idx:].strip()
                            else:
                                html_extracted = text
                            
                            # Keep the HTML (if multiple HTML parts, the last one is usually the complete/final version)
                            generated_html = html_extracted
                            print(f"✅ HTML content detected and collected (length: {len(generated_html)})")
                        else:
                            # This is thinking/reasoning text - accumulate all thinking parts
                            if thinking_text:
                                thinking_text += "\n\n" + text
                            else:
                                thinking_text = text
                            print(f"💭 Thinking text collected (total length: {len(thinking_text)})")
        
        # Build result with both thinking and HTML - ensure BOTH are included if they exist
        result_dict = {}
        
        # Always include thinking if it exists
        if thinking_text:
            result_dict["thinking"] = thinking_text
            print(f"📤 Thinking text added to result (length: {len(thinking_text)})")
        
        # Always include HTML - use generated if available, otherwise fallback
        if not generated_html:
            # Fallback to basic HTML if nothing generated
            generated_html = f"<div class='slide'><h1>{slide_plan.get('slide_title', 'New Slide')}</h1></div>"
            print(f"⚠️ No HTML generated, using fallback HTML")
        
        # STEP 7: Validate and fix the generated HTML using slide validation agent
        print(f"🔍 STEP 7: Validating and fixing slide HTML")
        validated_html = generated_html
        validation_thinking = None
        
        # Initialize validation token counters
        validation_input_tokens = 0
        validation_output_tokens = 0
        validation_thoughts_tokens = 0
        
        try:
            # Create InvocationContext for validation (similar to lightweight_slide_pipeline)
            from google.adk.agents.invocation_context import InvocationContext, new_invocation_context_id
            from google.adk.sessions import Session
            from google.adk.agents.base_agent import BaseAgent
            
            # Create a dummy agent for the context (validation agent will be created inside validate_and_fix_slide)
            class DummyAgent(BaseAgent):
                def __init__(self):
                    super().__init__(name="dummy_validation_agent")
            
            dummy_agent = DummyAgent()
            
            # Create session for validation context
            validation_session = Session(
                id=f"validation_{p_id}_0",  # Use slide index 0 for insertion
                app_name="slide_generator_app",
                user_id=user_id,
                state={
                    "user_id": user_id,
                    "p_id": p_id,
                    "app_name": "slide_generator_app"
                }
            )
            
            # Create InvocationContext
            validation_ctx = InvocationContext(
                session=validation_session,
                agent=dummy_agent,
                session_service=session_service,
                invocation_id=new_invocation_context_id()
            )
            
            # Initialize token counts in validation session state (validation agent will add to this)
            validation_session.state['token_counts'] = {
                'input_tokens': 0,
                'output_tokens': 0,
                'thoughts_tokens': 0
            }
            
            # Call validation function (slide_index=0 for insertion)
            fixed_html, validation_thinking_text, validation_iterations_count = await validate_and_fix_slide(
                html_content=generated_html,
                p_id=p_id,
                slide_index=0,  # Use 0 for insertion slides
                ctx=validation_ctx,
                max_retries=2,
                max_validation_iterations=3
            )
            
            # Get validation tokens from session state (if validation ran successfully)
            if validation_session.state.get('token_counts'):
                validation_input_tokens = validation_session.state['token_counts'].get('input_tokens', 0)
                validation_output_tokens = validation_session.state['token_counts'].get('output_tokens', 0)
                validation_thoughts_tokens = validation_session.state['token_counts'].get('thoughts_tokens', 0)
            
            # Use validated HTML if available, otherwise use original
            if fixed_html and fixed_html.strip():
                validated_html = fixed_html
                print(f"✅ Slide HTML validated and fixed (length: {len(validated_html)})")
            else:
                print(f"⚠️ Validation did not return fixed HTML, using original")
            
            # Combine validation thinking with generator thinking
            if validation_thinking_text:
                validation_thinking = validation_thinking_text
                if thinking_text:
                    thinking_text = f"{thinking_text}\n\n--- Validation ---\n{validation_thinking}"
                else:
                    thinking_text = validation_thinking
                print(f"💭 Validation thinking text collected (length: {len(validation_thinking)})")
            
        except Exception as e:
            logger.error(f"Error in slide validation: {e}")
            print(f"⚠️ Error during validation, using original HTML: {e}")
            # Continue with original HTML if validation fails
        
        # Build result with both thinking and HTML - use validated HTML
        result_dict = {}
        
        # Always include thinking if it exists
        if thinking_text:
            result_dict["thinking"] = thinking_text
            print(f"📤 Thinking text added to result (length: {len(thinking_text)})")
        
        # Always include validated HTML
        result_dict["generated_html"] = validated_html
        print(f"📤 Validated HTML content added to result (length: {len(validated_html)})")
        
        # Aggregate all tokens and store in MongoDB presentations collection
        total_input_tokens = slide_gen_input_tokens + validation_input_tokens
        total_output_tokens = slide_gen_output_tokens + validation_output_tokens
        total_thoughts_tokens = slide_gen_thoughts_tokens + validation_thoughts_tokens
        
        # Store tokens in MongoDB presentations collection (for aggregation with main execution)
        try:
            logger.info(f"📊 Attempting to store tokens: {total_input_tokens} input, {total_output_tokens} output (including {total_thoughts_tokens} thoughts)")
            
            # Get current token counts from presentations collection
            presentation = db.presentations.find_one({"p_id": p_id})
            if presentation:
                # Get current token counts or initialize
                current_token_counts = presentation.get("token_counts", {})
                current_input = current_token_counts.get("input_tokens", 0)
                current_output = current_token_counts.get("output_tokens", 0)
                current_thoughts = current_token_counts.get("thoughts_tokens", 0)
                
                logger.debug(f"📊 Current token counts in presentations: {current_input} input, {current_output} output, {current_thoughts} thoughts")
                
                # Calculate new totals
                new_input = current_input + total_input_tokens
                new_output = current_output + total_output_tokens
                new_thoughts = current_thoughts + total_thoughts_tokens
                
                # Update presentations collection with new token counts
                update_result = db.presentations.update_one(
                    {"p_id": p_id},
                    {"$set": {
                        "token_counts": {
                            "input_tokens": new_input,
                            "output_tokens": new_output,
                            "thoughts_tokens": new_thoughts
                        },
                        "updated_at": get_utc_timestamp_iso()
                    }}
                )
                
                if update_result.modified_count > 0:
                    logger.info(f"✅ TOKEN UPDATE SUCCESS: Database now has {new_input} input, {new_output} output, {new_thoughts} thoughts tokens")
                    logger.info(f"📊 Successfully stored {total_input_tokens} input and {total_output_tokens} output tokens from slide insertion in presentations collection")
                else:
                    logger.warning(f"⚠️ Token update did not modify any documents (p_id={p_id} may not exist)")
            else:
                logger.warning(f"⚠️ Presentation not found for p_id={p_id}, cannot update token counts")
                # Don't raise - token tracking is non-critical, continue execution
                
        except Exception as e:
            # If we can't update presentations collection, log but don't fail
            logger.warning(f"⚠️ Could not update presentations collection with tokens from slide insertion: {e}")
            # Don't raise - token tracking is non-critical, continue execution
        
        # Print summary of what we're returning
        print(f"✅ Slide generation and validation complete")
        print(f"   - Has thinking: {bool(thinking_text)}")
        print(f"   - Has HTML: {bool(validated_html)}")
        print(f"   - Result keys: {list(result_dict.keys())}")
        print(f"   - Tokens: {total_input_tokens} input, {total_output_tokens} output (including {total_thoughts_tokens} thoughts)")
        
        # Cleanup: Remove all screenshots for this p_id after slide generation and validation
        # Safe cleanup for multi-worker environments (Gunicorn)
        try:
            import shutil
            import time
            screenshots_dir = Path(os.getenv("SCREENSHOTS_DIR", "screenshots"))
            p_id_screenshot_dir = screenshots_dir/p_id
            
            if p_id_screenshot_dir.exists() and p_id_screenshot_dir.is_dir():
                # Wait a short time to ensure no other processes are still writing files
                # This helps avoid race conditions in multi-worker environments
                # time.sleep(1)
                
                # # Check if any files were modified in the last 3 seconds (safety check)
                # current_time = time.time()
                # recent_files = []
                # try:
                #     for file_path in p_id_screenshot_dir.iterdir():
                #         if file_path.is_file():
                #             file_mtime = file_path.stat().st_mtime
                #             # If file was modified in last 3 seconds, it might still be in use
                #             if current_time - file_mtime < 3:
                #                 recent_files.append(file_path.name)
                # except (OSError, PermissionError):
                #     # If we can't check files, skip cleanup to be safe
                #     logger.warning(f"⚠️ Cannot access screenshot directory for p_id {p_id}, skipping cleanup")
                #     print(f"⚠️ Cannot access screenshot directory, skipping cleanup")
                #     recent_files = ["unknown"]  # Force skip
                
                # Only delete if no recent file activity (safe to delete)
                # if not recent_files:
                try:
                    # Remove entire directory with all screenshots
                    shutil.rmtree(p_id_screenshot_dir, ignore_errors=True)
                    logger.info(f"🗑️ Cleaned up all screenshots for p_id: {p_id}")
                    print(f"🗑️ Cleaned up all screenshots for p_id: {p_id}")
                except (OSError, PermissionError) as delete_error:
                    # Another process might be using files, that's okay
                    logger.warning(f"⚠️ Could not delete screenshot directory for p_id {p_id}: {delete_error}")
                    print(f"⚠️ Could not delete screenshots (may be in use by another process)")
                # else:
                #     logger.info(f"ℹ️ Skipping cleanup for p_id {p_id}: recent file activity detected ({len(recent_files)} files)")
                #     print(f"ℹ️ Skipping cleanup: files may still be in use")
            else:
                logger.info(f"ℹ️ No screenshot directory found for p_id: {p_id} (nothing to clean up)")
        except Exception as e:
            logger.error(f"⚠️ Error cleaning up screenshots for p_id {p_id}: {e}")
            # Don't fail the entire process if cleanup fails
            print(f"⚠️ Warning: Could not clean up screenshots: {e}")
        
        result = json.dumps(result_dict)
        return result
        
    except Exception as e:
        logger.error(f"Error in generate_slide_html: {e}")
        print(f"❌ Error generating slide HTML: {e}")
        return json.dumps({
            "error": f"Error generating slide HTML: {str(e)}"
        })

# Create synchronous wrapper for async generate_slide_html
def generate_slide_html_sync(slide_plan_json: str, global_theme_json: str, template_html: str, p_id: str, user_id: str) -> str:
    """
    Generate HTML for a slide. 
    
    **CRITICAL: This is NOT the final step. You MUST call save_slide_to_database immediately after this function.**
    
    Returns a JSON string with:
    - "generated_html": The HTML content for the slide (always present)
    - "thinking": The reasoning/thinking text from the agent (optional, only if present)
    
    **MANDATORY NEXT STEP**: After receiving the result, you MUST:
    1. Parse the JSON string result using json.loads()
    2. Extract "generated_html" field
    3. Extract "thinking" field (if present, otherwise use None)
    4. Immediately call save_slide_to_database with:
       - slide_html=generated_html
       - thinking_text=thinking (or None)
       - All other required parameters (p_id, user_id, slide_index, slide_outline, template_info, global_theme)
    
    **DO NOT STOP OR COMPLETE** after calling this function. The slide is NOT saved until you call save_slide_to_database.
    """
    import asyncio
    import concurrent.futures
    
    def run_in_thread():
        """Run the async function in a new event loop"""
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        try:
            result = new_loop.run_until_complete(generate_slide_html(slide_plan_json, global_theme_json, template_html, p_id, user_id))
            return result
        finally:
            new_loop.close()
    
    try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_in_thread)
            result = future.result(timeout=120)  # 2 minute timeout for slide generation
            print(f"✅ Slide HTML generated successfully")
            return result
    except Exception as e:
        logger.error(f"Error in generate_slide_html_sync: {e}")
        print(f"❌ Error generating slide HTML: {e}")
        import json
        return json.dumps({
            "success": False,
            "error": f"Error generating slide HTML: {str(e)}",
            "generated_html": None
        })


# Create wrapper for async select_template_for_slide (use same name so orchestrator can call it)
def select_template_for_slide(slide_plan_json: str, presentation_type: str, p_id: Optional[str] = None) -> str:
    """
    Synchronous wrapper for async select_template_for_slide
    Runs in a separate thread with its own event loop to avoid "event loop already running" error
    """
    print(f"🎨 STEP 5: Selecting template for slide")
    import asyncio
    import concurrent.futures
    
    def run_in_thread():
        """Run the async function in a new event loop"""
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        try:
            result = new_loop.run_until_complete(_select_template_for_slide_async(slide_plan_json, presentation_type, p_id))
            return result
        finally:
            new_loop.close()
    
    try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_in_thread)
            result = future.result(timeout=120)  # 2 minute timeout for template selection
            print(f"✅ Template selected successfully")
            return result
    except Exception as e:
        logger.error(f"Error in select_template_for_slide_sync: {e}")
        print(f"❌ Error selecting template: {e}")
        import json
        return json.dumps({
            "success": False,
            "error": f"Error selecting template: {str(e)}",
            "selected_template_number": None,
            "template_html": None
        })

# Create function tools (functions are already named correctly)
select_template_tool = FunctionTool(select_template_for_slide)
generate_slide_html_tool = FunctionTool(generate_slide_html_sync)

def create_data_fetcher_agent() -> LlmAgent:
    """
    Sub-agent to fetch presentation data
    """
    return LlmAgent(
        name="data_fetcher",
        model=get_model_with_fallback(),  # Use PRO for function calling support
        description="Fetches presentation context and neighboring slide data",
        tools=[
            validate_insertion_position_tool,
            fetch_presentation_context_tool,
            fetch_neighboring_slides_tool
        ],
        instruction="""
    You are a data fetching agent. When called:
    1. Validate the insertion position
    2. Fetch presentation context (global_theme, title, metadata)
    3. Fetch neighboring slides context

    Return all data in a json format.
    """
    )





def create_slide_insertion_orchestrator() -> LlmAgent:
    """
    Create a simple slide insertion orchestrator with direct tools
    """
    return LlmAgent(
        name="slide_insertion_orchestrator",
        model=get_model_with_fallback(),  # Use PRO for function calling support
        description="Inserts new slides between existing slides with automatic renumbering and content generation",
        tools=[
            # Core database operations (3 tools)
            validate_insertion_position_tool,
            fetch_presentation_context_tool,
            fetch_neighboring_slides_tool,
            # Read-only slide fetch (needed when user asks to modify or inspect an existing slide before insertion)
            fetch_slide_data_tool,
            
            # Essential slide creation (4 tools)
            renumber_slides_after_insertion_tool,
            select_template_tool,
            generate_slide_html_tool,
            save_slide_to_database_tool,
            
            # Content generation (1 tool)
            generate_slide_plan_tool
        ],
        instruction="""
 You are a slide insertion agent. When a user requests to add a slide between existing slides:
 
 ## MANDATORY SEQUENTIAL WORKFLOW (YOU MUST COMPLETE ALL STEPS):
 
 **STEP 1: Parse Request**
 - Extract insertion position (e.g., "after slide 2" = position 2, "between 2 and 3" = position 2)
 - Extract topic/content from user message:
   * If user says "about X" or "on X", topic is X
   * If user says "add slide for X", topic is X
   * If user provides content directly, use that as topic
   * If topic is unclear, infer from context or use a generic topic based on insertion position
 - Extract slide type if mentioned (comparison, timeline, data, etc.), otherwise use "content"
 
 **STEP 2: Validate Position** 
 - Call validate_insertion_position with p_id and insert_after_slide
 - DO NOT proceed if validation fails
 
 **STEP 3: Fetch Context**
 - Call fetch_presentation_context to get global_theme, title, metadata
 - Call fetch_neighboring_slides to get context around insertion point
 - Save both results as JSON strings for use in STEP 4
 - **CRITICAL: DO NOT stop here - you MUST continue to STEP 4 (Generate Slide Plan)**
 - **The workflow is NOT complete after fetching context - you MUST generate the slide plan next**
 
 **STEP 4: Generate Slide Plan** (MANDATORY - DO NOT SKIP)
 - You MUST call generate_slide_plan - this step is REQUIRED and cannot be skipped
 - Extract topic from user message:
   * Look for phrases like "about X", "on X", "for X", "regarding X"
   * If user says "add slide between 2 and 3", infer topic from neighboring slides or use a generic topic
   * If topic is not clear, use "New Slide" or infer from presentation context
 - Call generate_slide_plan with:
   - topic: The topic extracted from user message (required, cannot be empty)
   - slide_type: Type if mentioned (comparison, timeline, data, etc.), otherwise "content"
   - presentation_context: Result from fetch_presentation_context (pass as JSON string)
   - neighboring_context: Result from fetch_neighboring_slides (pass as JSON string)
 - Save the slide_plan result (as JSON string) - you will need it for STEP 7
 - DO NOT stop here - you MUST continue to STEP 5
 - If you are unsure about the topic, use a reasonable default based on the insertion position and context
 
 **STEP 5: Renumber Slides**
 - Call renumber_slides_after_insertion with p_id and insert_after_slide
 - This creates a gap for the new slide
 - DO NOT stop here - you MUST continue to STEP 6
 
 **STEP 6: Select Template**
 - Call select_template_for_slide with:
   - slide_plan_json: The slide plan from STEP 4 (as JSON string)
   - presentation_type: "regular_presentation" (or extract from context)
   - p_id: Presentation ID
 - Save the template_html from the result
 - DO NOT stop here - you MUST continue to STEP 7
 
 **STEP 7: Generate HTML**
 - Call generate_slide_html_sync with:
   - slide_plan_json: Slide plan from STEP 4 (as JSON string)
   - global_theme_json: Global theme from STEP 3 (as JSON string)
   - template_html: Template HTML from STEP 6
   - p_id: Presentation ID
   - user_id: User ID
 - The result is a JSON string with "generated_html" and optionally "thinking"
 - Parse the JSON result immediately
 - Extract "generated_html" and "thinking" fields
 - DO NOT stop here - you MUST continue to STEP 8
 
 **STEP 8: Save to Database** (MANDATORY FINAL STEP)
 - Call save_slide_to_database with:
   - p_id: Presentation ID
   - user_id: User ID
   - slide_index: insert_position from STEP 2
   - slide_html: generated_html from STEP 7
   - slide_outline: slide_plan from STEP 4 (as dict, not JSON string)
   - template_info: Template info from STEP 6 (as dict)
   - global_theme: Global theme from STEP 3 (as dict, not JSON string)
   - thinking_text: thinking from STEP 7 (or None if not present)
 - This is the FINAL step - the slide is NOT saved until this is called
 - DO NOT complete the task without calling this function
 
 ## CRITICAL RULES:
 - You MUST complete ALL 8 steps in order
 - DO NOT stop after fetching context (STEP 3) - you MUST generate slide plan (STEP 4)
 - DO NOT stop after generating HTML (STEP 7) - you MUST save to database (STEP 8)
 - If you skip any step, the task is INCOMPLETE and FAILED
 - The task is only complete when save_slide_to_database is called successfully
 
 ## FILE-CONTEXT PRIORITY AND DEDUPLICATION (CRITICAL)

 - If the user mentions adding content "from the file", "from uploaded document", or similar, you MUST use the session `file_context` as the PRIMARY source of information for this slide (do NOT rely on web search for core content).
 - Extract succinct, slide-ready content from `file_context` first.
 - Avoid repeating content that already appears in existing slides:
   - Use `fetch_neighboring_slides` and/or `fetch_slide_data` to inspect nearby or specific slides.
   - Compare your intended bullet points/phrases against existing slide text; if a point already exists, choose a different point from `file_context`.
   - Keep bullets unique and non-overlapping with adjacent slides.

 ## EXAMPLE EXECUTION:
 User: "add a new slide between 2 and 3 which should be about tesla marketing vs traditional marketing"
 
 Execution sequence (you MUST follow this):
 1. Parse: insert_after_slide=2, topic="tesla marketing vs traditional marketing", slide_type="comparison"
 2. Call: validate_insertion_position(p_id="...", insert_after_slide=2)
 3. Call: fetch_presentation_context(p_id="...") → save as presentation_context_json
 4. Call: fetch_neighboring_slides(p_id="...", insert_after_slide=2) → save as neighboring_context_json
 5. Call: generate_slide_plan(topic="tesla marketing vs traditional marketing", slide_type="comparison", presentation_context=presentation_context_json, neighboring_context=neighboring_context_json) → save as slide_plan_json
 6. Call: renumber_slides_after_insertion(p_id="...", insert_after_slide=2)
 7. Call: select_template_for_slide(slide_plan_json=slide_plan_json, presentation_type="regular_presentation", p_id="...") → extract template_html from result
 8. Call: generate_slide_html_sync(slide_plan_json=slide_plan_json, global_theme_json=presentation_context_json, template_html=template_html, p_id="...", user_id="...") → parse JSON, extract generated_html and thinking
 9. Call: save_slide_to_database(p_id="...", user_id="...", slide_index=2, slide_html=generated_html, slide_outline=slide_plan_dict, template_info=template_info_dict, global_theme=global_theme_dict, thinking_text=thinking)
 
 ## ADDITIONAL RULES:
 - Use tools directly - no transfers needed
 - If you need to inspect an EXISTING slide, call fetch_slide_data before starting the insertion workflow
 - If user mentions "from file" or "from document", prioritize file_context from session state
 - Handle errors gracefully but DO NOT stop the workflow unless validation fails
 - Return success message ONLY after save_slide_to_database completes successfully
 
 ## PARAMETERS:
 - p_id: Presentation ID from context
 - user_id: User ID from context  
 - enhanced_query: User's request message
         """
        )


# Create the main orchestrator instance
slide_insertion_orchestrator = create_slide_insertion_orchestrator()
