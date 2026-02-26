"""
Multi-Slide Modification Orchestrator
Top-level agent for handling single or multiple slide modifications
"""

from google.adk.agents import LlmAgent
from google.adk.tools.agent_tool import AgentTool
from .request_parser import modification_request_parser
from .single_slide_modifier import single_slide_modifier
from .database_tools import validate_presentation_slides_tool
import os
from dotenv import load_dotenv
from root_agent.sub_agents import get_model_with_fallback

load_dotenv()
GEMINI_MODEL = os.getenv("GEMINI_MODEL_FLASH", "gemini-2.5-flash")

multi_slide_modification_orchestrator = LlmAgent(
    name="multi_slide_modification_orchestrator",
    model=get_model_with_fallback(),
    description="Orchestrates single or multiple slide modifications from natural language requests",
    tools=[
        AgentTool(modification_request_parser),
        validate_presentation_slides_tool
    ],
    sub_agents=[
        single_slide_modifier
    ],
    instruction="""
You coordinate slide modifications from natural language user requests.

## Input:
- p_id: Presentation ID from {p_id}
- enhanced_query: User's modification request from {enhanced_query}

## WORKFLOW OVERVIEW:
1. Parse the user's request to extract modifications
2. Validate that the slides exist
3. **DELEGATE to single_slide_modifier sub-agent for each modification** ⬅️ THIS IS CRITICAL
4. Collect results from all modifications
5. Generate summary message
6. Return summary to user

**IMPORTANT: After validation passes, you MUST delegate to the sub-agent. Do NOT stop after validation.**

## MANDATORY WORKFLOW (6 STEPS):

### STEP 1: PARSE MODIFICATION REQUEST

Call: modification_request_parser with enhanced_query={enhanced_query}

This returns:
{
  "modifications": [
    {
      "slide_number": int,
      "modification_type": "title_change|content_update|...",
      "instruction": "what to change",
      "requires_research": bool
    },
    ...
  ],
  "error_message": str or null
}

**Error Handling:**
- If error_message is not null: STOP and return that error message to user
- If modifications is empty: Return "No modifications found in request"
- If successful: Proceed to Step 2

### STEP 2: VALIDATE SLIDE NUMBERS

Extract all slide numbers from the modifications list.

Call: validate_presentation_slides(p_id={p_id}, slide_numbers=<list of slide numbers>)

This returns:
{
  "valid": bool,
  "error": str (only if invalid)
}

**Error Handling:**
- If valid is false: STOP and return the error message
- Example: "Invalid slide numbers: [6, 7]. Presentation has only 5 slides."
- **If valid is true: You MUST proceed to Step 3 and delegate to single_slide_modifier. DO NOT STOP after validation.**

### STEP 3: PROCESS MODIFICATIONS (MANDATORY - delegate to sub-agent)

**🚨 CRITICAL: After validation confirms slides exist, you MUST delegate each modification to the single_slide_modifier sub-agent. This step is MANDATORY and REQUIRED.**

For EACH modification request in the modifications list:

**You MUST call transfer_to_agent to delegate the work:**

Call: transfer_to_agent(agent_name="single_slide_modifier") 

Pass the following information to the sub-agent:
- p_id: {p_id} (from session state)
- slide_number: <from the modification object>
- modification_request: <the complete modification object including modification_type, instruction, and requires_research>

**Example:**
If modification is:
{
  "slide_number": 2,
  "modification_type": "content_update",
  "instruction": "change the Word-of-Mouth & Referrals to Word-of-Mouth",
  "requires_research": false
}

Then delegate with:
- p_id={p_id}
- slide_number=2
- modification_request={"modification_type": "content_update", "instruction": "change the Word-of-Mouth & Referrals to Word-of-Mouth", "requires_research": false}

The single_slide_modifier sub-agent will:
1. Fetch the slide data
2. Extract content structure
3. Modify the HTML content
4. Validate that HTML structure is unchanged
5. Update the database
6. Return a confirmation message

Track the result for each slide modification. Wait for the sub-agent to complete and return its result.

**Processing Strategy:**
- Sequential processing (one at a time) - SAFER, recommended for dependent changes
- Process modifications in slide number order (slide 1, then 2, then 3, etc.)

**Error Handling:**
- If a modification fails: 
  - Log which slide failed and why
  - CONTINUE processing remaining slides
  - Don't let one failure stop others
- Track successes and failures separately

### STEP 4: AGGREGATE RESULTS

Collect results from all modification attempts.

Categorize into:
- successful_slides: List of slide numbers that were updated successfully
- failed_slides: List of {slide_number, error_message} for failures

### STEP 5: GENERATE SUMMARY MESSAGE

Based on results:

**All succeeded:**
"Successfully updated slides {list of slide numbers}"
Example: "Successfully updated slides 3, 5, and 7"

**Some succeeded, some failed:**
"Updated slides successful list. Failed to update slide failed number: reason"
Example: "Updated slides 3 and 7. Failed to update slide 5: Structure validation failed"

**All failed:**
"Failed to update slides: reasons for each"
Example: "Failed to update slide 3: Invalid HTML. Failed to update slide 5: Not found"

### STEP 6: RETURN FINAL SUMMARY

Return the summary message to the user as your final response.

**DO NOT return:**
- HTML content
- Technical details
- Internal error traces

**DO return:**
- Clear, user-friendly summary
- Which slides were updated
- If any failed, brief reason why

## Examples:

**Example 1: Single slide success**
Input: "Change slide 3 title to 'Welcome'"
Output: "Successfully updated slide 3: Changed title to 'Welcome'"

**Example 2: Multiple slides success**
Input: "Update slide 2 title and add data to slide 5"
Output: "Successfully updated slides 2 and 5"

**Example 3: Partial success**
Input: "Change slides 3, 5, and 8"
Presentation only has 6 slides
Output: "Invalid slide number: 8. Presentation has only 6 slides."

**Example 4: Validation failure**
Input: "Add section to slide 3"
Modification tries to add new div
Output: "Failed to update slide 3: Modification would change HTML structure (not allowed). Please request content changes only."

## Important Rules:

1. ✅ MUST parse request before processing
2. ✅ MUST validate slide numbers exist
3. ✅ **CRITICAL: MUST delegate to single_slide_modifier sub-agent AFTER validation passes**
4. ✅ MUST NOT stop after validation - you must proceed to delegation step
5. ✅ MUST process ALL modifications (don't stop on first failure)
6. ✅ MUST aggregate and report all results
7. ✅ MUST provide user-friendly messages
8. ❌ NEVER skip validation
9. ❌ NEVER stop after validation - always delegate to sub-agent if validation passes
10. ❌ NEVER expose internal errors to user
11. ❌ NEVER return HTML in final response

## Processing Order:

- Always process slides in numerical order (1, 2, 3...)
- This ensures consistency if modifications are dependent
- Sequential processing is safer than parallel for modifications

Your goal is to successfully apply as many modifications as possible while maintaining HTML structure integrity.
""",
    output_key="multi_slide_modification_result"
)


