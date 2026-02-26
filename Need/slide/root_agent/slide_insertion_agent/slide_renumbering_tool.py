"""
Slide Renumbering Tool
Handles renumbering slides after insertion
"""

from google.adk.tools import FunctionTool
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


def renumber_slides_after_insertion(p_id: str, insert_after_slide: int) -> str:
    """
    Renumber slides after insertion point
    
    Args:
        p_id: Presentation ID
        insert_after_slide: Slide number to insert after (slides after this will be renumbered)
    
    Returns:
        JSON string with renumbering result
    """
    print(f"🔄 STEP 4: Renumbering slides after position {insert_after_slide}")
    try:
        # If inserting at end, no renumbering needed
        if insert_after_slide is None:
            return json.dumps({
                "success": True,
                "renumbered_count": 0,
                "message": "No renumbering needed - inserting at end"
            })
        
        # Update all slides where slide_number > insert_after_slide
        # Increment both slide_number and slide_index by 1
        result = db.slide_html.update_many(
            {
                "p_id": p_id,
                "slide_number": {"$gt": insert_after_slide}
            },
            {
                "$inc": {
                    "slide_number": 1,
                    "slide_index": 1
                }
            }
        )
        
        logger.info(f"Renumbered {result.modified_count} slides for p_id {p_id}")
        
        result_data = json.dumps({
            "success": True,
            "renumbered_count": result.modified_count,
            "message": f"Successfully renumbered {result.modified_count} slides"
        })
        print(f"✅ Slides renumbered successfully: {result.modified_count} slides updated")
        return result_data
        
    except Exception as e:
        logger.error(f"Error renumbering slides: {e}")
        print(f"❌ Error renumbering slides: {e}")
        return json.dumps({
            "success": False,
            "error": f"Database error: {str(e)}"
        })


def get_slide_numbers(p_id: str) -> str:
    """
    Get all slide numbers for a presentation (for debugging)
    
    Args:
        p_id: Presentation ID
    
    Returns:
        JSON string with slide numbers
    """
    try:
        slides = list(db.slide_html.find(
            {"p_id": p_id},
            {"slide_number": 1, "slide_index": 1, "_id": 0}
        ).sort("slide_number", 1))
        
        return json.dumps({
            "success": True,
            "slides": slides
        })
        
    except Exception as e:
        logger.error(f"Error getting slide numbers: {e}")
        return json.dumps({
            "success": False,
            "error": f"Database error: {str(e)}"
        })


# Create function tools
renumber_slides_after_insertion_tool = FunctionTool(renumber_slides_after_insertion)

get_slide_numbers_tool = FunctionTool(get_slide_numbers)
