"""
Database tools for slide insertion
"""

from google.adk.tools import FunctionTool
from datetime import datetime, timezone
import json
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


def validate_insertion_position(p_id: str, insert_after_slide: int) -> str:
    """
    Validate that the insertion position is valid for the presentation
    
    Args:
        p_id: Presentation ID
        insert_after_slide: Slide number to insert after (0-based for end)
    
    Returns:
        JSON string with validation result
    """
    print(f"🔍 STEP 1: Validating insertion position {insert_after_slide} for presentation {p_id}")
    try:
        # Get total slide count
        total_slides = db.slide_html.count_documents({"p_id": p_id})
        
        if total_slides == 0:
            return json.dumps({
                "valid": False,
                "error": f"Presentation {p_id} has no slides"
            })
        
        # Validate position
        if insert_after_slide is None:
            # Insert at end
            return json.dumps({
                "valid": True,
                "total_slides": total_slides,
                "insert_position": total_slides
            })
        
        if insert_after_slide < 0:
            return json.dumps({
                "valid": False,
                "error": "Cannot insert before slide 1"
            })
        
        if insert_after_slide > total_slides:
            return json.dumps({
                "valid": False,
                "error": f"Cannot insert after slide {insert_after_slide}. Presentation has only {total_slides} slides"
            })
        
        result = json.dumps({
            "valid": True,
            "total_slides": total_slides,
            "insert_position": insert_after_slide
        })
        print(f"✅ Position validation result: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Error validating insertion position: {e}")
        print(f"❌ Error validating insertion position: {e}")
        return json.dumps({
            "valid": False,
            "error": f"Database error: {str(e)}"
        })


def fetch_presentation_context(p_id: str) -> str:
    """
    Fetch presentation context including global theme and metadata
    
    Args:
        p_id: Presentation ID
    
    Returns:
        JSON string with presentation context
    """
    print(f"📊 STEP 2: Fetching presentation context for {p_id}")
    try:
        # Fetch presentation document
        presentation = db.presentations.find_one({"p_id": p_id})
        if not presentation:
            return json.dumps({
                "error": f"Presentation {p_id} not found"
            })
        
        # Get total slides count
        total_slides = db.slide_html.count_documents({"p_id": p_id})
        
        result = json.dumps({
            "p_id": p_id,
            "title": presentation.get("title", ""),
            "global_theme": presentation.get("global_theme", {}),
            "presentation_metadata": presentation.get("presentation_metadata", {}),
            "total_slides": total_slides
        })
        print(f"✅ Presentation context fetched successfully")
        return result
        
    except Exception as e:
        logger.error(f"Error fetching presentation context: {e}")
        print(f"❌ Error fetching presentation context: {e}")
        return json.dumps({
            "error": f"Database error: {str(e)}"
        })


def fetch_neighboring_slides(p_id: str, slide_number: int) -> str:
    """
    Fetch slides before and after the insertion point for context
    
    Args:
        p_id: Presentation ID
        slide_number: Slide number to insert after
    
    Returns:
        JSON string with neighboring slide context
    """
    print(f"🔍 STEP 3: Fetching neighboring slides around position {slide_number}")
    try:
        # Fetch slide before insertion point
        slide_before = None
        if slide_number > 0:
            slide_before = db.slide_html.find_one({
                "p_id": p_id,
                "slide_number": slide_number
            })
        
        # Fetch slide after insertion point (will become slide_number + 2)
        slide_after = db.slide_html.find_one({
            "p_id": p_id,
            "slide_number": slide_number + 1
        })
        
        # Extract relevant context
        context = {
            "slide_before": None,
            "slide_after": None
        }
        
        if slide_before:
            context["slide_before"] = {
                "slide_number": slide_before.get("slide_number"),
                "slide_plan": slide_before.get("slide_plan", {}),
                "template_info": slide_before.get("template_info", {}),
                "content_metadata": slide_before.get("content_metadata", {})
            }
        
        if slide_after:
            context["slide_after"] = {
                "slide_number": slide_after.get("slide_number"),
                "slide_plan": slide_after.get("slide_plan", {}),
                "template_info": slide_after.get("template_info", {}),
                "content_metadata": slide_after.get("content_metadata", {})
            }
        
        result = json.dumps(context)
        print(f"✅ Neighboring slides context fetched successfully")
        return result
        
    except Exception as e:
        logger.error(f"Error fetching neighboring slides: {e}")
        print(f"❌ Error fetching neighboring slides: {e}")
        return json.dumps({
            "error": f"Database error: {str(e)}"
        })


def save_presentation_outline(p_id: str, outline: str) -> str:
    """
    Save presentation outline after slide insertion
    
    Args:
        p_id: Presentation ID
        outline: JSON string containing the updated outline (can be dict or list)
    
    Returns:
        JSON string with save result
    """
    try:
        outline_data = json.loads(outline)
        
        # Handle both dict and list formats
        if isinstance(outline_data, list):
            # If outline is just a list of slides, wrap it in a dict
            outline_data = {
                "slide_outline": outline_data,
                "total_slides": len(outline_data)
            }
        
        # Create or update presentation outline document
        outline_doc = {
            "p_id": p_id,
            "user_id": outline_data.get("user_id", ""),
            "slide_outline": outline_data.get("slide_outline", []),
            "total_slides": outline_data.get("total_slides", 0),
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc)
        }
        
        # Upsert the outline
        result = db.presentation_outlines.update_one(
            {"p_id": p_id},
            {"$set": outline_doc},
            upsert=True
        )
        
        return json.dumps({
            "success": True,
            "upserted_id": str(result.upserted_id) if result.upserted_id else None,
            "modified_count": result.modified_count
        })
        
    except Exception as e:
        logger.error(f"Error saving presentation outline: {e}")
        return json.dumps({
            "success": False,
            "error": f"Database error: {str(e)}"
        })


def save_global_theme_to_presentation(p_id: str, global_theme_json: str) -> str:
    """
    Save global_theme to presentations collection
    
    Args:
        p_id: Presentation ID
        global_theme_json: JSON string containing the global_theme
    
    Returns:
        JSON string with save result
    """
    try:
        global_theme = json.loads(global_theme_json)
        
        # Update presentations collection with global_theme
        result = db.presentations.update_one(
            {"p_id": p_id},
            {"$set": {"global_theme": global_theme, "updated_at": datetime.now(timezone.utc)}}
        )
        
        if result.modified_count > 0:
            return json.dumps({
                "success": True,
                "message": "global_theme saved to presentations collection"
            })
        else:
            return json.dumps({
                "success": False,
                "message": "No presentation found or no changes made"
            })
        
    except Exception as e:
        logger.error(f"Error saving global_theme: {e}")
        return json.dumps({
            "success": False,
            "error": f"Database error: {str(e)}"
        })


# Create function tools
validate_insertion_position_tool = FunctionTool(validate_insertion_position)

fetch_presentation_context_tool = FunctionTool(fetch_presentation_context)

fetch_neighboring_slides_tool = FunctionTool(fetch_neighboring_slides)

save_presentation_outline_tool = FunctionTool(save_presentation_outline)

save_global_theme_to_presentation_tool = FunctionTool(save_global_theme_to_presentation)
