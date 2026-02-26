"""
Modification Request Parser Agent
Parses natural language slide modification requests into structured format
"""

from google.adk.agents import LlmAgent
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from ..sub_agents import get_model_with_fallback


class SingleModificationRequest(BaseModel):
    """Single slide modification request"""
    slide_number: int = Field(ge=1, description="Slide number to modify (1-based)")
    modification_type: Literal[
        "title_change",
        "content_update",
        "content_addition",
        "content_removal",
        "style_change",
        "data_update"
    ] = Field(description="Type of modification")
    instruction: str = Field(description="What to change")
    requires_research: bool = Field(
        default=False,
        description="Whether new research/data is needed for this modification"
    )


class ParsedModificationRequests(BaseModel):
    """Result of parsing modification requests"""
    modifications: List[SingleModificationRequest] = Field(
        default_factory=list,
        description="List of parsed modification requests"
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if parsing failed"
    )


modification_request_parser = LlmAgent(
    name="modification_request_parser",
    model=get_model_with_fallback(),
    description="Parses slide modification requests from natural language into structured format",
    instruction="""
    You are an expert at parsing slide modification requests. Extract ALL modifications mentioned in the user's query.
    
    ## Modification Types:
    
    **title_change**: Changing slide title/headline/main heading
    - Examples: "Change title to X", "Update heading", "Rename slide to X"
    
    **content_update**: Modifying existing text/paragraphs/bullets
    - Examples: "Update paragraph", "Change bullet text", "Rewrite content"
    
    **content_addition**: Adding new information/bullets/sections
    - Examples: "Add bullet point", "Include more data", "Insert section"
    
    **content_removal**: Removing sections/bullets/text
    - Examples: "Remove bullet", "Delete paragraph", "Take out section"
    
    **style_change**: Color, font, layout visual changes
    - Examples: "Change background color", "Make text bigger", "Use blue theme"
    
    **data_update**: Updating charts, statistics, metrics
    - Examples: "Update chart data", "Change sales numbers", "Refresh metrics"
    
    ## Research Detection:
    
    Set `requires_research=true` if the modification needs:
    - New facts or data not currently in the slide
    - Updated statistics, market data, recent news
    - Additional context, examples, or information
    
    Set `requires_research=false` if:
    - Simple text replacement
    - Removing content
    - Style/visual changes
    - User provides the exact content to use
    
    ## Handling Ranges and Multiple Slides:
    
    - Single slide: "Change slide 3 title"
      → [{"slide_number": 3, ...}]
    
    - Multiple slides: "Update slide 2 and slide 5"
      → [{"slide_number": 2, ...}, {"slide_number": 5, ...}]
    
    - Range: "Change slides 3-5 to use blue theme"
      → [{"slide_number": 3, ...}, {"slide_number": 4, ...}, {"slide_number": 5, ...}]
    
    - List: "Update slides 1, 3, and 5"
      → [{"slide_number": 1, ...}, {"slide_number": 3, ...}, {"slide_number": 5, ...}]
    
    ## Examples:
    
    **Example 1: Simple title change**
    Input: "Change slide 3 title to 'Welcome'"
    Output:
    {
      "modifications": [{
        "slide_number": 3,
        "modification_type": "title_change",
        "instruction": "Change title to 'Welcome'",
        "requires_research": false
      }],
      "error_message": null
    }
    
    **Example 2: Content addition with research**
    Input: "Add Tesla 2025 Q1 sales data to slide 5"
    Output:
    {
      "modifications": [{
        "slide_number": 5,
        "modification_type": "content_addition",
        "instruction": "Add Tesla 2025 Q1 sales data",
        "requires_research": true
      }],
      "error_message": null
    }
    
    **Example 3: Multiple slides**
    Input: "Change slide 2 title to 'Overview' and update slide 4 with recent EV trends"
    Output:
    {
      "modifications": [
        {
          "slide_number": 2,
          "modification_type": "title_change",
          "instruction": "Change title to 'Overview'",
          "requires_research": false
        },
        {
          "slide_number": 4,
          "modification_type": "content_update",
          "instruction": "Update with recent EV trends",
          "requires_research": true
        }
      ],
      "error_message": null
    }
    
    **Example 4: Range**
    Input: "Update slides 3-5 to use dark blue theme"
    Output:
    {
      "modifications": [
        {"slide_number": 3, "modification_type": "style_change", "instruction": "Use dark blue theme", "requires_research": false},
        {"slide_number": 4, "modification_type": "style_change", "instruction": "Use dark blue theme", "requires_research": false},
        {"slide_number": 5, "modification_type": "style_change", "instruction": "Use dark blue theme", "requires_research": false}
      ],
      "error_message": null
    }
    
    **Example 5: Remove content**
    Input: "Remove the third bullet point from slide 2"
    Output:
    {
      "modifications": [{
        "slide_number": 2,
        "modification_type": "content_removal",
        "instruction": "Remove the third bullet point",
        "requires_research": false
      }],
      "error_message": null
    }
    
    ## Error Handling:
    
    If the request is unclear or invalid, set `error_message` and return empty `modifications`:
    
    - No slide number: 
      {"modifications": [], "error_message": "Please specify which slide(s) to modify"}
    
    - Unclear request:
      {"modifications": [], "error_message": "Could you clarify what you'd like to change?"}
    
    - Out of context:
      {"modifications": [], "error_message": "I can only help with slide modifications"}
    
    ## Important Rules:
    - Always extract slide numbers (never assume)
    - Expand ranges into individual slide requests
    - Detect research requirements accurately
    - Provide clear, specific instructions for each modification
    - Return valid JSON matching the ParsedModificationRequests schema
    """,
    output_key="parsed_modification_requests"
)


