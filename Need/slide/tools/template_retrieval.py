"""
Template Retrieval Tool - Fetches HTML templates from the template service
Returns all templates for LLM agent to analyze and select from
"""
from google.adk.tools import FunctionTool
import requests
import logging
import os
from typing import List, Dict
from dotenv import load_dotenv
from datetime import datetime
from core.database import get_mongo_client

load_dotenv()
logger = logging.getLogger(__name__)

# Template service configuration
TEMPLATE_SERVICE_URL = os.getenv("TEMPLATE_SERVICE_URL", "http://163.172.181.252:8010/items")

def track_missing_template(category: str, template_type: str, slide_purpose: str, presentation_type: str):
    """
    Track missing template in the database for analytics and template creation prioritization.
    """
    try:
        client = get_mongo_client()
        db = client["slide_creator_db"]
        
        # Create tracking document
        missing_template_record = {
            "category": category,
            "type": template_type,
            "slide_purpose": slide_purpose,
            "presentation_type": presentation_type,
            "timestamp": datetime.utcnow(),
            "count": 1
        }
        
        # Check if this combination already exists
        existing = db.template_missing.find_one({
            "category": category,
            "type": template_type,
            "slide_purpose": slide_purpose,
            "presentation_type": presentation_type
        })
        
        if existing:
            # Increment count and update timestamp
            db.template_missing.update_one(
                {"_id": existing["_id"]},
                {
                    "$inc": {"count": 1},
                    "$set": {"last_requested": datetime.utcnow()},
                    "$push": {"request_timestamps": datetime.utcnow()}
                }
            )
            logger.info(f"📊 Updated missing template count for category='{category}', type='{template_type}' (total: {existing.get('count', 0) + 1})")
        else:
            # Create new record
            missing_template_record["request_timestamps"] = [datetime.utcnow()]
            db.template_missing.insert_one(missing_template_record)
            logger.info(f"📊 Tracked new missing template: category='{category}', type='{template_type}'")
            
    except Exception as e:
        logger.error(f"❌ Failed to track missing template: {str(e)}")

# Mapping from slide purposes to template types
SLIDE_PURPOSE_TO_TYPE = {
    "title": "title",
    "problem_statement": "problem",
    "solution_intro": "solution",
    "core_content": "content",
    "data_visualization": "data",
    "case_study": "testimonial",
    "comparison": "comparison",
    "benefits": "benefits",
    "implementation": "process",
    "key_takeaways": "overview",
    "call_to_action": "demo",
    "timeline": "timeline",
    "features": "features",
    "objectives": "objectives",
    "quote": "Quote",
    "contact": "contact",
    "analysis": "analysis",
    "thank_you": "thank you"
}

# Mapping from presentation types to categories
PRESENTATION_TYPE_TO_CATEGORY = {
    "pitch_deck": "business",
    "sales_deck": "business",
    "regular_presentation": "business",
    "data_report": "technical",
    "keynote_deck": "business",
    "academic_presentation": "academic",
    "research_presentation": "academic",
    "technical_presentation": "technical"
}


def retrieve_html_templates(
    slide_purpose: str,
    presentation_type: str = "regular_presentation"
) -> str:
    """
    Retrieve HTML templates from the template service.
    Returns all matching templates for the LLM agent to analyze and select from.
    
    Args:
        slide_purpose: Purpose of the slide (e.g., 'title', 'problem_statement', 'core_content')
        presentation_type: Type of presentation (e.g., 'pitch_deck', 'sales_deck', 'regular_presentation')
        
    Returns:
        A formatted string containing all template options with full HTML code.
        The LLM will analyze these and select the most appropriate one.
    """
    try:
        # Map slide purpose to template type
        template_type = SLIDE_PURPOSE_TO_TYPE.get(slide_purpose, "content")
        
        # Map presentation type to category
        category = PRESENTATION_TYPE_TO_CATEGORY.get(presentation_type, "business")
        
        logger.info(f"🔍 Fetching templates: category={category}, type={template_type}")
        
        # Make API request
        response = requests.get(
            TEMPLATE_SERVICE_URL,
            params={
                "category": category,
                "type": template_type
            },
            timeout=15
        )
        
        if response.status_code != 200:
            error_msg = f"Template API returned status {response.status_code}"
            logger.error(f"❌ {error_msg}")
            return f"Failed to retrieve templates: {error_msg}"
        
        templates = response.json()
        
        if not templates:
            logger.warning(f"⚠️ No templates found for category='{category}' and type='{template_type}'. Trying fallback to 'content' type...")
            
            # Track missing template in database
            track_missing_template(category, template_type, slide_purpose, presentation_type)
            
            # Fallback: Try with type='content' (generic content template)
            fallback_response = requests.get(
                TEMPLATE_SERVICE_URL,
                params={
                    # "category": category,
                    "type": template_type  # Generic content template
                },
                timeout=15
            )
            
            if fallback_response.status_code == 200:
                fallback_templates = fallback_response.json()
                if fallback_templates:
                    logger.info(f"✅ Found {len(fallback_templates)} fallback templates using type='{template_type}'")                    
                    templates = fallback_templates
                    template_type = template_type  #"content"  # Update the type for display
                else:
                    # Second fallback: Try with category='business' and type='content'
                    logger.warning(f"⚠️ No fallback templates found. Trying second fallback with category='business' and type='content'...")
                    
                    second_fallback_response = requests.get(
                        TEMPLATE_SERVICE_URL,
                        params={
                            "type": template_type       # Generic content type
                        },
                        timeout=15
                    )
                    
                    if second_fallback_response.status_code == 200:
                        second_fallback_templates = second_fallback_response.json()
                        if second_fallback_templates:
                            logger.info(f"✅ Found {len(second_fallback_templates)} second fallback templates using category='business' and type='content'")
                            templates = second_fallback_templates
                            template_type = template_type
                            return f"No templates found for category='{category}' and type='{template_type}'. No fallback templates available (tried 'content' type and 'business' category)."
                    else:
                        return f"No templates found for category='{category}' and type='{template_type}'. All fallback attempts failed."
            else:
                return f"No templates found for category='{category}' and type='{template_type}'. Fallback request failed with status {fallback_response.status_code}."
        
        original_count = len(templates)
        if original_count > 10:
            logger.info(f"🔢 Limiting templates from {original_count} to 10 for LLM consumption")
            templates = templates[:10]        
        # Format ALL templates for LLM analysis
        formatted_output = f"**Retrieved {len(templates)} HTML Templates (requested: {original_count}):**\n\n"
        formatted_output += f"Category: {category} | Type: {template_type}\n"
        formatted_output += f"For slide purpose: {slide_purpose}\n"
        
        # Add fallback indicator if we used fallback templates
        # if template_type == "content" and category == "business":
        #     formatted_output += f"**Note: Using fallback templates (generic content templates) as specific templates were not available.**\n"
        # elif template_type == "content":
        #     formatted_output += f"**Note: Using fallback templates (content type) as specific type was not available.**\n"
        
        formatted_output += "\n"
        formatted_output += "**IMPORTANT**: Analyze ALL templates below and SELECT THE BEST ONE that matches your slide requirements (layout, elements, content structure).\n\n"
        formatted_output += "---\n\n"
        
        for i, template in enumerate(templates, 1):
            template_id = template.get("id", "Unknown ID")
            description = template.get("description", "No description available")
            html_code = template.get("html_code", "")
            
            # Note: We don't escape braces here because:
            # 1. Python f-strings will include the literal string value (including any {angle})
            # 2. Escaping will be done in the agent instruction where ADK processes it
            # If html_code contains {angle}, Python f-string will try to interpolate it and fail.
            # So we need to escape for Python f-string safety, but not double-escape.
            # We'll escape braces to make them safe for Python f-strings: { becomes {{
            html_code_safe = html_code.replace("{", "{{").replace("}", "}}")
            
            formatted_output += f"## TEMPLATE {i}:\n"
            formatted_output += f"**ID:** {template_id}\n"
            formatted_output += f"**Description:** {description}\n"
            formatted_output += f"**HTML Length:** {len(html_code)} characters\n\n"
            
            # Show HTML structure preview (first 800 chars to understand structure)
            #preview = html_code_safe[:800] if len(html_code_safe) > 800 else html_code_safe
            #formatted_output += f"**HTML Structure Preview:**\n```html\n{preview}...\n```\n\n"
            
            # Full HTML code (escaped for Python f-string safety only)
            formatted_output += f"**Complete HTML Code:**\n```html\n{html_code_safe}\n```\n\n"
            formatted_output += "---\n\n"
        
        logger.info(f"✅ Retrieved {len(templates)} templates for LLM selection")
        return formatted_output
        
    except requests.exceptions.Timeout:
        error_msg = "Template API request timed out"
        logger.error(f"❌ {error_msg}")
        return f"Failed to retrieve templates: {error_msg}"
    except requests.exceptions.ConnectionError:
        error_msg = f"Cannot connect to template service at {TEMPLATE_SERVICE_URL}"
        logger.error(f"❌ {error_msg}")
        return f"Failed to retrieve templates: {error_msg}. Make sure the service is running."
    except Exception as e:
        error_msg = f"Template retrieval error: {str(e)}"
        logger.error(f"❌ {error_msg}")
        return f"Failed to retrieve templates: {error_msg}"


# Create the tool
retrieve_html_templates_tool = FunctionTool(
    func=retrieve_html_templates
)

