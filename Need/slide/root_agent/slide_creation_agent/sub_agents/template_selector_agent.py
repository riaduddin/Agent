"""
Template Selector Agent - Intelligently selects the best HTML template
"""
from google.adk.agents import LlmAgent
import logging
import os
from root_agent.sub_agents import get_native_gemini_model
import json

def create_template_selector_agent(slide_outline: dict, all_templates: str) -> LlmAgent:
    """
    Creates an agent that analyzes all available templates and selects the best one.
    
    Args:
        slide_outline: Slide definition with requirements
        all_templates: String containing all template options from API
    
    Returns:
        LlmAgent configured to select the best template
    """
    
    # Prepare inputs and escape braces to avoid ADK session-state interpolation on raw content
    slide_requirements = json.dumps({
        "slide_purpose": slide_outline.get("slide_purpose", "content"),
        "slide_title": slide_outline.get("slide_title", ""),
        "suggested_type": slide_outline.get("suggested_type", "centered"),
        "content_guidance": slide_outline.get("content_guidance", ""),
        "required_elements": slide_outline.get("required_elements", [])
    }, indent=2)

    # Escape braces in JSON and template corpus so ADK doesn't treat {angle}, {x}, etc. as context variables
    # ADK uses regex {+[^{}]*}+ to match variables, so we need to escape braces to prevent matching
    # Since instructions are Python f-strings, we need to account for that too
    def _escape_braces_for_adk(text: str) -> str:
        """
        Neutralize braces so ADK does not treat {angle} patterns as context variables.
        Use HTML entities to avoid matching ADK's regex while preserving visual braces for the LLM.
        """
        if not isinstance(text, str):
            return text
        return text.replace("{", "&#123;").replace("}", "&#125;")

    slide_requirements_escaped = _escape_braces_for_adk(slide_requirements)
    all_templates_escaped = _escape_braces_for_adk(all_templates)
    
    # Build instruction text separately so we can log it for debugging
    instruction_text = f"""
    You are an expert HTML template analyst. Your job is to SELECT THE BEST template from multiple options.

    ## SLIDE REQUIREMENTS:
    {slide_requirements_escaped}

    ## AVAILABLE TEMPLATES:
    {all_templates_escaped}

    ## YOUR TASK:

    **Step 1: Analyze Each Template**
    For EACH template provided, evaluate:

    1. **Layout Compatibility** (35% importance)
    - Does the HTML structure match `suggested_type: {slide_outline.get('suggested_type')}`?
    - Layout mapping:
        * hero_title → centered, large heading, minimal elements
        * two_column_split → grid/flex with 2 main sections
        * three_column_grid → 3-column layout
        * four_quadrant_grid → 2x2 or 4-column grid
        * timeline_flow → sequential, numbered steps
        * data_dashboard → metrics, stats, numbers prominently displayed
        * comparison_matrix → side-by-side comparison structure
        * centered_focus → single centered content area
        * icon_grid → grid of icons with text
    
    2. **Required Elements Match** (30% importance)
    - Required elements: {slide_outline.get('required_elements', [])}
    - Check if template HTML contains structures for these:
        * headline → <h1> or main title element
        * subheading → <h2> or subtitle element
        * body_content → <p> or content sections
        * metrics → stat/number display elements
        * timeline → sequential step containers
        * visual_icon → icon/image placeholders
        * chart_data → chart/graph containers
        * comparison_table → table or comparison grid
        * action_items → button/CTA elements
        * sources → footer or citation area

    3. **Content Structure Fit** (20% importance)
    - Content guidance: "{slide_outline.get('content_guidance', '')}"
    - Can this template accommodate this type of content?
    - Does it have appropriate sections and containers?

    4. **Complexity Appropriateness** (15% importance)
    - Slide purpose: {slide_outline.get('slide_purpose')}
    - Simple purposes (title, quote, thank_you) → prefer simpler templates
    - Complex purposes (data_visualization, comparison, case_study) → prefer detailed templates
    - Look at number of sections, divs, and content areas

    **Step 2: Select Best Template**
    After analyzing ALL templates, select the ONE that best matches requirements.

    **Step 3: Output Your Decision**

    Return ONLY a JSON object with this EXACT structure (no markdown, no explanation before or after):

    {{{{
    "selected_template_number": <the sequential number 1, 2, 3, etc.>,
    "selected_template_id": "<the actual template ID from the **ID:** field, NOT the number>",
    "selection_reasoning": "Brief explanation of why this template was chosen (2-3 sentences)",
    "layout_match_score": <1-10>,
    "elements_match_score": <1-10>,
    "content_fit_score": <1-10>,
    "overall_confidence": <1-10>
    }}}}

    **CRITICAL**:
    - `selected_template_number`: Use the sequential number (1, 2, 3, etc. from "TEMPLATE 1", "TEMPLATE 2")
    - `selected_template_id`: Use the ACTUAL template ID shown in the "**ID:**" field for that template
    - Example: If you select "TEMPLATE 2" which has "**ID:** template_abc123", then:
    * selected_template_number: 2
    * selected_template_id: "template_abc123"
    - Return ONLY the JSON object
    - No markdown code blocks (no ```json or ```)
    - No additional text or explanations
    - Just the raw JSON starting with {{ and ending with }}

    Analyze the templates and make your selection now.
    """

    # Log full instruction for debugging
    # try:
    #     print("📄 Template Selector Instruction (BEGIN)\n%s\n📄 Template Selector Instruction (END)", instruction_text)
    # except Exception:
    #     pass

    return LlmAgent(
        name=f"template_selector",
        model=get_native_gemini_model(),
        description="""
        Expert template analyst that selects the best HTML template based on slide requirements.
        Analyzes layout structure, element availability, and content fit.
        """,
        instruction=instruction_text,
        tools=[],
        output_key="template_selection"
    )

