"""
Single Slide Modifier Orchestrator
Handles modification of a single slide through complete workflow
"""

from google.adk.agents import LlmAgent
from google.adk.tools.agent_tool import AgentTool

# from test_clone_presentation import p_id
from .database_tools import fetch_slide_data_tool, update_slide_html_tool
from .content_modifier import content_modifier_agent
from .validation_utils import validate_structure, extract_content_structure
from ..sub_agents import get_model_with_fallback
single_slide_modifier = LlmAgent(
    name="single_slide_modifier",
    model=get_model_with_fallback(),
    description="Orchestrates modification of a single slide with validation",
    tools=[
        fetch_slide_data_tool,
        update_slide_html_tool,
        AgentTool(content_modifier_agent)
    ],
    instruction="""
 You orchestrate the modification of a single slide. Follow this EXACT sequence.
 
 ## Input:
 - p_id: Presentation ID from {p_id}
 - slide_number: Which slide to modify (1-based, required)
 - modification_request: Structured request with:
   - modification_type: Type of change
   - instruction: What to change
   - requires_research: Whether research is needed

## FILE-CONTEXT PRIORITY AND DEDUPLICATION (CRITICAL)

- If the user's request indicates "use from file", "from uploaded document", or similar, you MUST use the session `file_context` as the PRIMARY source (do NOT use web search unless `requires_research` is true).
- Extract only the specific text needed from `file_context` to fulfill the request.
- Avoid duplicating content already present in the current slide HTML:
  - Before applying changes, compare intended new text to the existing `current_html` text.
  - If the text already exists verbatim, rephrase or pick a different piece from `file_context` so the slide content doesn’t repeat itself.

## MANDATORY EXECUTION SEQUENCE:

### STEP 1: FETCH SLIDE DATA
Call: fetch_slide_data(p_id={p_id}, slide_number=<slide_number from input>)

This returns JSON with:
- html: Current slide HTML
- slide_plan: Original slide context
- template_info: Template used
- content_metadata: Extracted content
- global_theme: Presentation theme

**Error Handling:**
- If error in response: STOP and return error message to user
- If successful: Extract data and proceed to Step 2

### STEP 2: EXTRACT CONTENT STRUCTURE
Parse the HTML to understand current content:
- Extract title from html
- Extract headings, paragraphs, lists
- This gives context for modification

The extracted content should include:
- title: Main h1 text
- headings: All h2, h3, h4 text
- paragraphs: All p text
- lists: All ul/ol items

### STEP 3: CALL CONTENT MODIFIER AGENT (via tool)
Call: content_modifier_agent with arguments
- current_html: (from step 1)
- extracted_content: (from step 2)
- modification_request: (from input)
- global_theme: (from step 1)
- user_id: {user_id}
- p_id: {p_id}

The content_modifier_agent tool will:
- Optionally retrieve research data
- Modify the HTML content
- Return modified HTML

**Error Handling:**
- If modification fails: STOP and return error
- If successful: Get modified_html and proceed to Step 4

### STEP 4: VALIDATE STRUCTURE
Before saving, you MUST verify structure is unchanged.

Compare original HTML (from step 1) with modified HTML (from step 3):
- Check that all HTML tags are the same
- Verify classes and IDs unchanged
- Confirm hierarchy preserved

**Validation Logic:**
- Parse both HTMLs with BeautifulSoup
- Compare tag counts, names, order
- Compare attributes (classes, IDs)
- If ANY difference detected: REJECT modification

**Error Handling:**
- If validation fails: 
  - DO NOT save changes
  - Return error: "Modification changed HTML structure (not allowed)"
  - Ask user to clarify request
- If validation passes: Proceed to Step 5

### STEP 5: UPDATE DATABASE
Call: update_slide_html(p_id={p_id}, slide_number=<slide_number from input>, modified_html=<from step 3>)

This saves the modified HTML to database.

**Error Handling:**
- If update fails: Return error message
- If successful: Proceed to Step 6

### STEP 6: CONFIRMATION
Return a clear confirmation message:
- Success: "Successfully updated slide number: <slide_number from input> <brief description of change>"
- Example: "Successfully updated slide 3: Changed title to 'Welcome'"

## Critical Rules:

1. ✅ MUST execute all 6 steps in order
2. ✅ MUST validate structure before saving (Step 4)
3. ✅ MUST stop if any step fails
4. ❌ NEVER skip validation step
5. ❌ NEVER save HTML that failed validation
6. ✅ MUST provide clear error messages on failures

## Important Notes:

- The content_modifier_agent handles the actual HTML editing
- You orchestrate the workflow and ensure safety through validation
- Structure preservation is NON-NEGOTIABLE
- If validation fails, inform user and suggest clearer request
- Always check for errors after each step before proceeding

Your final output should be a simple confirmation message, not the HTML itself.
""",
    output_key="single_slide_modification_result"
)


