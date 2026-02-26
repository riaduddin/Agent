"""
Content Modifier Agent
LLM-powered agent that modifies slide content while preserving HTML structure
"""

from google.adk.agents import LlmAgent
import os
from dotenv import load_dotenv
import sys
from pathlib import Path

# Add root directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from tools.qdrant_retrieval import retrieve_research_tool
from tools.image_search import search_images_tool

load_dotenv()
GEMINI_MODEL = os.getenv("GEMINI_MODEL_FLASH", "gemini-2.5-flash")

from google.adk.models.google_llm import Gemini
from google.genai import types
def create_content_modifier_agent(
    current_html: str,
    extracted_content: dict,
    modification_request: dict,
    global_theme: dict,
    user_id: str,
    p_id: str
) -> LlmAgent:
    """
    Creates a content modifier agent for a specific modification
    
    Args:
        current_html: Current slide HTML
        extracted_content: Extracted content structure (title, headings, etc.)
        modification_request: What to modify
        global_theme: Presentation theme
        user_id: User ID for research
        p_id: Presentation ID for research
    
    Returns:
        LlmAgent configured for content modification
    """
    
    # Format inputs as JSON for instruction
    import json
    context_json = json.dumps({
        "extracted_content": extracted_content,
        "modification_request": modification_request,
        "global_theme": global_theme
    }, indent=2)
    
    return LlmAgent(
        name="content_modifier_agent",
        model=Gemini(
            model=GEMINI_MODEL,
            retry_options=types.HttpRetryOptions(initial_delay=30, attempts=3,exp_base=2.0,jitter=0.3,http_status_codes=[429, 500, 502, 503, 504])
        ),
        description="Modifies slide content while strictly preserving HTML structure",
        tools=[retrieve_research_tool, search_images_tool],
        instruction=f"""
You are a slide content modification expert. Your ONLY job is to modify CONTENT while preserving HTML structure.

## CRITICAL RULES (MANDATORY):

### ❌ NEVER DO THESE:
1. ❌ DO NOT add or remove HTML tags (div, span, section, etc.)
2. ❌ DO NOT change HTML tag names
3. ❌ DO NOT modify classes or IDs
4. ❌ DO NOT change tag attributes (except style values if style_change)
5. ❌ DO NOT rearrange HTML elements
6. ❌ DO NOT change HTML structure or hierarchy
7. ❌ DO NOT add new HTML elements

### ✅ ONLY DO THESE:
1. ✅ DO change text content inside existing tags
2. ✅ DO update numbers, statistics, data values
3. ✅ DO modify list item text (not <li> tags)
4. ✅ DO update paragraph text (not <p> tags)
5. ✅ DO change heading text (not <h1>, <h2> tags)
6. ✅ DO update CSS style VALUES (not selectors) if style_change
7. ✅ DO update Chart.js data arrays if data_update

## Current Slide Context:
<context>
{context_json}
</context>

## Current HTML (READ-ONLY for structure reference):
<current_html>
{current_html}
</current_html>

## Your Task:

**Modification Type:** {modification_request.get('modification_type', 'content_update')}
**Instruction:** {modification_request.get('instruction', 'Update content')}
**Requires Research:** {modification_request.get('requires_research', False)}

### Step-by-Step Process:

**STEP 1: Understand What to Modify**

Based on modification_type:

- **title_change**: 
  - Find the <h1> tag in the HTML
  - Replace ONLY the text inside <h1>...</h1>
  - Keep <h1> tag and all attributes unchanged

- **content_update**:
  - Identify which paragraph or section to update
  - Replace ONLY the text content
  - Keep all HTML tags unchanged

- **content_addition**:
  - Find appropriate existing element to add content to
  - Add text to existing <p>, <li>, or other content tags
  - DO NOT create new structural elements

- **content_removal**:
  - Identify what to remove
  - Remove text or list items
  - If removing <li>, you can remove the entire <li>tag</li>
  - But preserve the parent <ul> or <ol> structure

- **style_change**:
  - Find the <style> section or inline style attributes
  - Modify CSS VALUES only (colors, sizes, etc.)
  - Keep CSS selectors and structure unchanged

- **data_update**:
  - Find Chart.js data arrays or numbers in text
  - Update values only
  - Keep chart configuration structure

**STEP 2: Gather Research (if needed)**

If requires_research is true:
  - Derive search query from the instruction
  - Call retrieve_research_context tool
  - Use: search_query, user_id="{user_id}", p_id="{p_id}", limit=6
  - Extract relevant facts/data from results

**STEP 3: Apply Modification**

Locate the exact content to modify in the HTML.

Based on the modification:
1. Find the target element (h1, p, li, etc.)
2. Replace ONLY the text content
3. Preserve ALL HTML tags, attributes, structure

**Example (title_change):**
```
Current: <h1 class="title" style="color: #CC0000">Old Title</h1>
Target: Change to "New Title"
Result: <h1 class="title" style="color: #CC0000">New Title</h1>
          ✅ class, style, tag preserved - ONLY text changed
```

**Example (content_addition with research):**
```
Current: 
<ul class="points">
  <li>Point 1</li>
  <li>Point 2</li>
</ul>

Target: Add "Tesla 2025 sales data"
Research: Call retrieve_research_context("Tesla 2025 Q1 sales data", "{user_id}", "{p_id}", 6)
Result: 
<ul class="points">
  <li>Point 1</li>
  <li>Point 2</li>
  <li>Tesla delivered 495,000 units in Q1 2025</li>
</ul>
          ✅ ul tag, class preserved - ONLY new <li> added with research data
```

**Example (style_change):**
```
Current: 
<style>
  h1 {{ color: #CC0000; }}
</style>

Target: Change heading color to blue
Result:
<style>
  h1 {{ color: #0000FF; }}
</style>
          ✅ CSS structure preserved - ONLY color value changed
```

**STEP 4: Output Complete Modified HTML**

Return the ENTIRE HTML document with ONLY the requested content changes applied.

- All HTML tags must be identical to original
- All classes, IDs, attributes must be unchanged
- Only text content or CSS values should be different
- Maintain exact same HTML structure

## Validation Reminder:

Your output will be validated to ensure structure is unchanged. If you modify:
- Tag names
- Classes or IDs
- HTML hierarchy
- Add/remove elements

The modification will be REJECTED and the user will be asked to clarify.

## Output Format:

Return ONLY the complete modified HTML (no explanations, no markdown).
Start with <!DOCTYPE html> and end with </html>.
""",
    output_key="modified_html_content"
    )


# Generic, reusable agent registered as a tool. This variant expects inputs to be
# passed as tool-call arguments: current_html, extracted_content, modification_request,
# global_theme, user_id, p_id. It does not bake inputs into the instruction so it can
# be safely wrapped by AgentTool and invoked dynamically by other agents.
content_modifier_agent = LlmAgent(
    name="content_modifier_agent",
    model=GEMINI_MODEL,
    description="Modifies slide content while strictly preserving HTML structure",
    tools=[retrieve_research_tool, search_images_tool],
    instruction="""
You are a slide content modification expert. Your ONLY job is to modify CONTENT while preserving HTML structure.

Inputs are provided as arguments when this tool is called by the orchestrator:
- current_html: the full HTML of the slide (read-only for structure reference)
- extracted_content: structured content extracted from the current slide (JSON)
- modification_request: the requested change including modification_type, instruction, requires_research (JSON)
- global_theme: presentation theme context (JSON)
- user_id: the user identifier for research retrieval
- p_id: the presentation identifier for research retrieval

Use ONLY the provided arguments; do not assume any hidden context.

CRITICAL RULES (MANDATORY):

1) Do NOT change HTML structure (no adding/removing/reordering of tags; keep attributes/classes/ids identical).
2) Only change text content within existing elements, update numbers, list item text, or CSS value literals if style_change.
3) If requires_research is true, derive a concise search query from the instruction and call retrieve_research_context with
   search_query, user_id, p_id, limit=6, then use only factual results to enrich the content.
4) If the modification involves adding/replacing an image or logo placeholder, call search_images with:
   - search_queries: 1–3 precise terms (e.g., "[Brand] official logo transparent PNG")
   - count_per_query: 3–5
   Use the best-quality URL in the existing <img> tag's src without changing structure.

Process:
1. Read modification_request.modification_type and modification_request.instruction
2. Locate the exact target in current_html using extracted_content as guidance
3. Apply ONLY content-level changes as allowed by rules
4. Return the COMPLETE modified HTML as plain text

Output: Only the full modified HTML document. No explanations or markdown.
""",
    output_key="modified_html_content"
)

