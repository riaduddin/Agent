"""
Content Generator Sub-Agent
Generates slide content when user only provides topic or type
"""

from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool
from google.adk.tools.agent_tool import AgentTool
import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add root directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
# Import research agent
from .research_agent import create_research_agent

load_dotenv()
GEMINI_MODEL = os.getenv("GEMINI_MODEL_FLASH", "gemini-2.5-flash")
GEMINI_MODEL_PRO = os.getenv("GEMINI_MODEL_PRO", "gemini-2.5-pro")


def generate_slide_plan(topic: str, slide_type: str, presentation_context: str, neighboring_context: str) -> str:
    """
    Generate a slide plan based on topic, type, and context
    
    Args:
        topic: Topic for the slide
        slide_type: Type of slide (comparison, timeline, data, etc.)
        presentation_context: JSON string with presentation context
        neighboring_context: JSON string with neighboring slide context
    
    Returns:
        JSON string with generated slide plan
    """
    print(f"📝 STEP 4: Generating slide plan for topic: {topic}")
    try:
        # Parse context with error handling for malformed JSON
        pres_context = {}
        neighbor_context = {}
        
        if presentation_context:
            try:
                # Try to parse JSON
                pres_context = json.loads(presentation_context)
            except json.JSONDecodeError as json_err:
                print(f"⚠️ Warning: Failed to parse presentation_context JSON: {json_err}")
                print(f"⚠️ presentation_context preview: {presentation_context[:200] if len(presentation_context) > 200 else presentation_context}...")
                # Try to fix invalid escape sequences more carefully
                try:
                    import re
                    # First, try to decode if it's a raw string with escape sequences
                    # Fix invalid escape sequences (like \x, \u without proper format, or standalone \)
                    # Pattern: backslash not followed by valid escape character
                    # Valid escapes in JSON: \\, \/, \", \b, \f, \n, \r, \t, \uXXXX
                    fixed_context = re.sub(r'\\(?![\\"/bfnrt]|u[0-9a-fA-F]{4})', r'\\\\', presentation_context)
                    # Also handle cases where backslash is at end of string or followed by invalid char
                    # Try parsing again
                    pres_context = json.loads(fixed_context)
                    print(f"✅ Fixed and parsed presentation_context")
                except Exception as fix_err:
                    # If regex fix fails, try a more aggressive approach: encode/decode to handle escapes
                    try:
                        # Try encoding as bytes and decoding to handle escape sequences
                        import codecs
                        # Try to decode as if it's a string literal
                        try:
                            decoded = codecs.decode(presentation_context, 'unicode_escape')
                            pres_context = json.loads(decoded)
                            print(f"✅ Fixed and parsed presentation_context using unicode_escape")
                        except (UnicodeDecodeError, json.JSONDecodeError, Exception):
                            # If still fails, try using ast.literal_eval as fallback for dict-like strings
                            try:
                                import ast
                                pres_context = ast.literal_eval(presentation_context)
                                if not isinstance(pres_context, dict):
                                    pres_context = {}
                                print(f"✅ Parsed presentation_context using ast.literal_eval")
                            except (ValueError, SyntaxError):
                                # If all fails, use empty dict
                                pres_context = {}
                                print(f"⚠️ Using empty presentation_context due to parsing error: {fix_err}")
                    except Exception as final_err:
                        # Last resort: use empty dict
                        pres_context = {}
                        print(f"⚠️ Using empty presentation_context - all parsing attempts failed: {final_err}")
        
        if neighboring_context:
            try:
                # Try to parse JSON
                neighbor_context = json.loads(neighboring_context)
            except json.JSONDecodeError as json_err:
                print(f"⚠️ Warning: Failed to parse neighboring_context JSON: {json_err}")
                print(f"⚠️ neighboring_context preview: {neighboring_context[:200] if len(neighboring_context) > 200 else neighboring_context}...")
                # Try to fix invalid escape sequences more carefully
                try:
                    import re
                    # First, try to decode if it's a raw string with escape sequences
                    # Fix invalid escape sequences (like \x, \u without proper format, or standalone \)
                    # Pattern: backslash not followed by valid escape character
                    # Valid escapes in JSON: \\, \/, \", \b, \f, \n, \r, \t, \uXXXX
                    fixed_context = re.sub(r'\\(?![\\"/bfnrt]|u[0-9a-fA-F]{4})', r'\\\\', neighboring_context)
                    # Try parsing again
                    neighbor_context = json.loads(fixed_context)
                    print(f"✅ Fixed and parsed neighboring_context")
                except Exception as fix_err:
                    # If regex fix fails, try a more aggressive approach: encode/decode to handle escapes
                    try:
                        # Try encoding as bytes and decoding to handle escape sequences
                        import codecs
                        # Try to decode as if it's a string literal
                        try:
                            decoded = codecs.decode(neighboring_context, 'unicode_escape')
                            neighbor_context = json.loads(decoded)
                            print(f"✅ Fixed and parsed neighboring_context using unicode_escape")
                        except:
                            # If still fails, try using ast.literal_eval as fallback for dict-like strings
                            try:
                                import ast
                                neighbor_context = ast.literal_eval(neighboring_context)
                                if not isinstance(neighbor_context, dict):
                                    neighbor_context = {}
                                print(f"✅ Parsed neighboring_context using ast.literal_eval")
                            except:
                                # If all fails, use empty dict
                                neighbor_context = {}
                                print(f"⚠️ Using empty neighboring_context due to parsing error: {fix_err}")
                    except Exception as final_err:
                        # Last resort: use empty dict
                        neighbor_context = {}
                        print(f"⚠️ Using empty neighboring_context - all parsing attempts failed: {final_err}")
        
        # Extract relevant information
        presentation_title = pres_context.get("title", "")
        global_theme = pres_context.get("global_theme", {})
        
        # Determine slide purpose based on type
        purpose_mapping = {
            "comparison": "comparison_analysis",
            "timeline": "process_flow",
            "data": "data_visualization",
            "list": "content_summary",
            "overview": "introduction",
            "conclusion": "summary"
        }
        
        slide_purpose = purpose_mapping.get(slide_type, "content")
        
        # Generate content guidance based on topic and type
        content_guidance = f"Create a {slide_type} slide about {topic}"
        if presentation_title:
            content_guidance += f" related to {presentation_title}"
        
        # Determine required elements based on slide type
        element_mapping = {
            "comparison": ["headline", "body_content", "comparison_table", "visual_icon"],
            "timeline": ["headline", "body_content", "timeline", "visual_icon"],
            "data": ["headline", "body_content", "chart_data", "metrics"],
            "list": ["headline", "body_content", "visual_icon"],
            "overview": ["headline", "subheading", "body_content"],
            "conclusion": ["headline", "body_content", "action_items"]
        }
        
        required_elements = element_mapping.get(slide_type, ["headline", "body_content"])
        
        # Generate search query for research
        search_query = f"{topic} {slide_type} {presentation_title}".strip()
        
        # Create slide plan
        slide_plan = {
            "slide_purpose": slide_purpose,
            "slide_title": f"{topic.title()} - {slide_type.title()}",
            "suggested_type": slide_type,
            "search_query": search_query,
            "content_guidance": content_guidance,
            "required_elements": required_elements,
            "fallback_keywords": [topic, slide_type, presentation_title]
        }
        
        result = json.dumps({
            "success": True,
            "slide_plan": slide_plan
        })
        print(f"✅ Slide plan generated successfully")
        return result
        
    except Exception as e:
        print(f"❌ Error generating slide plan: {e}")
        return json.dumps({
            "success": False,
            "error": f"Error generating slide plan: {str(e)}"
        })


# Create function tools
generate_slide_plan_tool = FunctionTool(generate_slide_plan)
from google.adk.models.google_llm import Gemini
from google.genai import types

def create_content_generator() -> LlmAgent:
    """
    Create the content generator agent
    """
    return LlmAgent(
        name="content_generator",
        model=Gemini(
            model=GEMINI_MODEL,
            retry_options=types.HttpRetryOptions(initial_delay=30, attempts=3,exp_base=2.0,jitter=0.3,http_status_codes=[429, 500, 502, 503, 504])
        ),
        description="Generate slide content and plans when user provides only topic or type",
        tools=[generate_slide_plan_tool],
        sub_agents=[create_research_agent()],
        instruction="""
You are an expert at generating slide content and plans for presentations.

Your task is to:
1. Generate comprehensive slide plans based on topics and slide types
2. Use research agent when additional information is needed
3. Ensure content fits the presentation context and theme

When generating slide plans, consider:
- The presentation's overall theme and context
- The type of slide requested (comparison, timeline, data, etc.)
- Required elements for the slide type
- Appropriate content guidance for the topic

## Workflow:

1. **Generate initial slide plan** using generate_slide_plan tool
2. **If more research is needed**, transfer to research_agent sub-agent
   - The research agent will search Qdrant database and Google
   - It will provide comprehensive research findings
3. **Enhance slide plan** with research findings
4. **Return final slide plan** with all necessary information

## Research Agent Usage:
Transfer to: research_agent

Provide the research agent with:
- topic: The main topic to research
- search_query: Specific search query
- user_id: From presentation context
- p_id: From presentation context

The research agent will return structured research results that you can use to enhance the slide plan.

Always return structured JSON responses with success status and generated content.
        """
    )
