"""
Database tools for slide modification
"""

from google.adk.tools import FunctionTool
from datetime import datetime, timezone
from bs4 import BeautifulSoup
from typing import List
import sys
from pathlib import Path
import logging

# Add root directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from core.database import get_mongo_client

logger = logging.getLogger(__name__)

# Database connection
client = get_mongo_client()
db = client["slide_creator_db"]


def fetch_slide_data(p_id: str, slide_number: int) -> str:
    """
    Fetch slide HTML, plan, and theme from database
    
    Args:
        p_id: Presentation ID
        slide_number: Slide number (1-based, user-facing)
    
    Returns:
        JSON string containing slide data or error
    """
    import json
    
    try:
        slide_index = slide_number - 1
        
        # Fetch slide document
        slide_doc = db.slide_html.find_one({"p_id": p_id, "slide_index": slide_index})
        if not slide_doc:
            return json.dumps({
                "error": f"Slide {slide_number} not found for presentation {p_id}"
            })
        
        # Fetch presentation document for global theme
        presentation = db.presentations.find_one({"p_id": p_id})
        if not presentation:
            return json.dumps({
                "error": f"Presentation {p_id} not found"
            })
        
        # Return all relevant data
        return json.dumps({
            "html": slide_doc.get("body", ""),
            "slide_plan": slide_doc.get("slide_plan", {}),
            "template_info": slide_doc.get("template_info", {}),
            "content_metadata": slide_doc.get("content_metadata", {}),
            "global_theme": presentation.get("global_theme", {}),
            "presentation_metadata": presentation.get("presentation_metadata", {})
        })
        
    except Exception as e:
        logger.error(f"Error fetching slide data: {e}")
        return json.dumps({"error": f"Database error: {str(e)}"})


def update_slide_html(p_id: str, slide_number: int, modified_html: str) -> str:
    """
    Update slide HTML in database
    
    Args:
        p_id: Presentation ID
        slide_number: Slide number (1-based)
        modified_html: Modified HTML content
    
    Returns:
        JSON string with success/error message
    """
    import json
    
    try:
        slide_index = slide_number - 1
        
        # Update content metadata after modification
        soup = BeautifulSoup(modified_html, 'html.parser')
        
        content_metadata = {
            "extracted_title": soup.find('h1').get_text(strip=True) if soup.find('h1') else "",
            "extracted_headings": [h.get_text(strip=True) for h in soup.find_all(['h1', 'h2', 'h3', 'h4'])],
            "extracted_paragraphs": [p.get_text(strip=True) for p in soup.find_all('p') if p.get_text(strip=True)],
            "has_charts": bool(soup.find('canvas')),
            "has_lists": bool(soup.find(['ul', 'ol'])),
            "text_length": len(soup.get_text())
        }
        
        # Update slide in database
        result = db.slide_html.update_one(
            {"p_id": p_id, "slide_index": slide_index},
            {
                "$set": {
                    "body": modified_html,  # Store raw HTML (no prettify)
                    "content_metadata": content_metadata,
                    "updated_at": datetime.now(timezone.utc)
                }
            }
        )
        
        if result.modified_count > 0:
            logger.info(f"✅ Successfully updated slide {slide_number}")
            return json.dumps({
                "success": True,
                "message": f"Successfully updated slide {slide_number}"
            })
        else:
            logger.warning(f"⚠️ No changes made to slide {slide_number}")
            return json.dumps({
                "success": False,
                "message": f"Slide {slide_number} was not updated (no changes detected)"
            })
            
    except Exception as e:
        logger.error(f"Error updating slide: {e}")
        return json.dumps({
            "success": False,
            "message": f"Database update failed: {str(e)}"
        })


def validate_presentation_slides(p_id: str, slide_numbers: List[int]) -> str:
    """
    Validate that all requested slide numbers exist in the presentation
    
    Args:
        p_id: Presentation ID
        slide_numbers: List of slide numbers to validate
    
    Returns:
        JSON string with validation result
    """
    import json
    
    try:
        # Count actual slides in slide_html collection (more reliable than total_slides field)
        total_slides = db.slide_html.count_documents({"p_id": p_id})
        
        if total_slides == 0:
            # Check if presentation exists at all
            presentation = db.presentations.find_one({"p_id": p_id})
            if not presentation:
                return json.dumps({
                    "valid": False,
                    "error": f"Presentation {p_id} not found"
                })
            else:
                return json.dumps({
                    "valid": False,
                    "error": f"Presentation {p_id} exists but has no slides generated yet"
                })
        
        # Check if all requested slide numbers exist
        invalid_slides = []
        for slide_num in slide_numbers:
            # Check if this specific slide exists
            slide_exists = db.slide_html.find_one({
                "p_id": p_id,
                "slide_number": slide_num
            })
            if not slide_exists:
                invalid_slides.append(slide_num)
        
        if invalid_slides:
            return json.dumps({
                "valid": False,
                "error": f"Slide(s) {invalid_slides} not found. Presentation has {total_slides} slides (1-{total_slides})."
            })
        
        # Success - only return valid flag (orchestrator doesn't need total_slides)
        return json.dumps({
            "valid": True
        })
        
    except Exception as e:
        logger.error(f"Error validating slides: {e}")
        return json.dumps({
            "valid": False,
            "error": f"Validation error: {str(e)}"
        })


# Create function tools
fetch_slide_data_tool = FunctionTool(func=fetch_slide_data)
update_slide_html_tool = FunctionTool(func=update_slide_html)
validate_presentation_slides_tool = FunctionTool(func=validate_presentation_slides)


