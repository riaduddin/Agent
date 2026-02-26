from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool
from google.adk.tools.agent_tool import AgentTool
from .planning_agent import plan_modifier_agent
from .slide_generator_agent import slide_generator_agent
import json
from dotenv import load_dotenv
from datetime import datetime, timezone
import os
from bs4 import BeautifulSoup
from .request_parser_agent import request_parser_agent
from .status_announcer_agent import status_announcer_agent
from .database_connection import db
from ..sub_agents import get_model_with_fallback
load_dotenv()
GEMINI_MODEL=os.getenv("GEMINI_MODEL_FLASH","gemini-2.0-flash")

def slide_data_fetcher_tool(p_id: str, slide_number: int) -> str:
    """
    Fetches the original slide plan and the presentation's global theme from the database.

    Args:
        p_id: The unique ID of the presentation.
        slide_number: The user-facing slide number (e.g., 1, 2, 3).

    Returns:
        A JSON string containing the 'original_plan' and 'global_theme'.
    """
    try:
        # Fetch the presentation document to get the global theme
        presentation_doc = db.presentations.find_one({"p_id": p_id})
        if not presentation_doc:
            return json.dumps({"error": f"Presentation with p_id '{p_id}' not found."})
        global_theme = presentation_doc.get("global_theme")

        # Fetch the specific slide document to get its plan
        slide_index = slide_number - 1
        slide_doc = db.slide_html.find_one({"p_id": p_id, "slide_index": slide_index})
        if not slide_doc:
            return json.dumps({"error": f"Slide number {slide_number} not found for p_id '{p_id}'."})
        original_plan = slide_doc.get("slide_plan")

        if not original_plan or not global_theme:
            return json.dumps({"error": "Required data (plan or theme) is missing from the database."})

        # Return both pieces of data in a single, structured response
        return json.dumps({
            "original_plan": original_plan,
            "global_theme": global_theme
        })

    except Exception as e:
        # It's good practice to handle potential errors
        return json.dumps({"error": f"An unexpected database error occurred: {str(e)}"})


def database_updater_tool(p_id: str, slide_number: int, modified_plan: dict, new_html: str) -> str:
    """
    Updates a slide's HTML and its plan in the database to keep them synchronized.

    Args:
        p_id: The unique ID of the presentation.
        slide_number: The slide number that was edited.
        modified_plan: The updated JSON plan object for the slide.
        new_html: The new, complete HTML string for the slide.

    Returns:
        A JSON string indicating success or failure.
    """
    try:
        slide_index = slide_number - 1

        # Clean and pretty-print the HTML
        soup = BeautifulSoup(new_html.replace("\\", ""), "html.parser")
        pretty_html = soup.prettify()

        result = db.slide_html.update_one(
            {"p_id": p_id, "slide_index": slide_index},
            {
                "$set": {
                    "slide_plan": modified_plan,
                    "body": pretty_html,
                    "timestamp": datetime.now(timezone.utc)
                }
            }
        )

        if result.modified_count == 0:
            return json.dumps({"status": "error", "message": f"Failed to find and update slide {slide_number}."})

        return json.dumps({"status": "success", "message": f"Successfully updated slide {slide_number}."})

    except Exception as e:
        return json.dumps({"status": "error", "message": f"Database update failed: {str(e)}"})


def log_event_to_db(event, db,type=None,session_id=None, user_id=None,output=None,reference=False,final_content_summary=None):
    if reference:
        db.references.insert_one({
            "event_id": event.id,
            "name": getattr(event, "name", None),  # Agent or tool name
            "author": getattr(event, "author", None),
            "timestamp": str(getattr(event, "timestamp", datetime.utcnow())),
            "reference": final_content_summary,  # Reference content
            "session_id": session_id,
            "user_id": user_id
        })
    else:   
        log_entry = {
            "event_id": event.id,
            "name": getattr(event, "name", None),  # Agent or tool name
            "author": getattr(event, "author", None),
            "timestamp": str(getattr(event, "timestamp", datetime.utcnow())),
            "type": type if type else "event",  # Event type (e.g., 'thinking', 'response')
            "session_id": session_id,
            "user_id": user_id,
            "parts": [],
            "function_calls": [],
            "function_responses": []
        }

        # ✅ Extract content parts, function calls, and responses
        if event.content and event.content.parts:
            for part in event.content.parts:
                part_entry = {
                    "thought": getattr(part, "thought", False),
                    "text": getattr(part, "text", None)
                }

                # Check for function_call
                if getattr(part, "function_call", None):
                    fc = part.function_call
                    log_entry["function_calls"].append({
                        "name": fc.name,
                        "arguments": fc.args
                    })

                # Check for function_response
                if getattr(part, "function_response", None):
                    fr = part.function_response
                    log_entry["function_responses"].append({
                        "name": fr.name,
                        "response": fr.response
                    })

                log_entry["parts"].append(part_entry)

        db.system_logs.insert_one(log_entry)


import time
class LoggingSlideModificationAgent(LlmAgent):
    async def _run_async_impl(self, ctx):
        user_id = ctx.session.state.get("user_id")
        p_id = ctx.session.state.get("p_id")

        async for event in super()._run_async_impl(ctx):
            # Yield as normal so execution continues
            # print("event_for_modification: ", event)
            # time.sleep(10)

            # Extract event data
            try:
                # content_text = getattr(event, "content", None)
                # if isinstance(content_text, list):
                #     # Convert Gemini structured parts to plain text
                #     content_text = "\n".join(
                #         part.text for part in content_text if hasattr(part, "text")
                #     )
                log_event_to_db(event,db,session_id=p_id,user_id=user_id)
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        text = getattr(part, "text", None)
                        if text:
                            # try:
                            #     output = generate_user_facing_message(text)
                            # except Exception as e:
                            #     output = "An error occurred while processing your request."
                            db.agent_outputs_2.insert_one({
                                "user_id": user_id,
                                "session_id": p_id,
                                "role": "agent",
                                "agent_name": "unknown_agent",
                                "parsed_output": part.text,
                                "p_id": p_id,
                                "timestamp": datetime.utcnow()
                            })
                
            except Exception as e:
                print(f"[Logging Error] {e}")
            yield event



slide_modification_agent = LlmAgent(
    name="slide_modification_agent",
    model=get_model_with_fallback(),
    description="Orchestrates a full slide edit from a single natural language command.",
    instruction="""
    You are the master AI coordinator for editing presentation slides. Your job is to manage the entire workflow from a user's command to the final database update.

    🚨 **CRITICAL: You MUST execute ALL 6 steps in the exact order specified below. DO NOT skip any step. DO NOT deviate from this sequence. Each step is MANDATORY and REQUIRED.**

    ---
    🎯 **Input:** `p_id` (string) and `enhanced_query` (string)
    ---
    
    ⚙️ **MANDATORY EXECUTION SEQUENCE - FOLLOW EXACTLY:**

    **STEP 1 - PARSE REQUEST (REQUIRED):**
    - MUST call `request_parser_agent` with the `enhanced_query`
    - This will return `slide_number` and `edit_request`
    - If it returns an error: STOP IMMEDIATELY and return the error message to user
    - DO NOT proceed to step 2 until this step completes successfully

    **STEP 2 - FETCH DATA (REQUIRED):**
    - MUST call `slide_data_fetcher_tool` with  {p_id}  and `slide_number`
    - This will return `original_plan` and `global_theme`
    - If it returns an error: STOP IMMEDIATELY and return the error message to user
    - DO NOT proceed to step 3 until this step completes successfully

    **STEP 3 - MODIFY PLAN (REQUIRED):**
    - MUST call `plan_modifier_agent` with `original_plan` and `edit_request`
    - This will return `modified_plan` and `error_message`
    - If `error_message` is not null: STOP IMMEDIATELY and return that exact error message to user
    - DO NOT proceed to step 4 until this step completes successfully with no errors

    **STEP 4 - GENERATE HTML (REQUIRED):**
    - MUST call `slide_generator_agent` with `modified_plan` and `global_theme`
    - This will return `new_html`
    - If it returns an error: STOP IMMEDIATELY and return the error message to user
    - DO NOT proceed to step 5 until this step completes successfully

    **STEP 5 - SAVE TO DATABASE (REQUIRED):**
    - MUST call `database_updater_tool` as your FINAL action
    - MUST provide:  {p_id} , `slide_number`, `modified_plan`, and `new_html`
    - If it returns an error: STOP and return the error message to user
    - DO NOT proceed to step 6 until this step completes successfully

    **STEP 6 - FINAL CONFIRMATION (REQUIRED):**
    - After database tool confirms success, provide a simple confirmation message
    - Example: "Successfully edited and saved slide [number]." 
    - OR return the success message from the database tool itself
    - DO NOT return HTML content in your final response

    ---
    ⚠️ **STRICT RULES - NO EXCEPTIONS:**
    - You MUST execute steps 1-6 in exact sequential order
    - You CANNOT skip any step under any circumstances
    - You MUST stop immediately if any step returns an error
    - You MUST check for errors after each step before proceeding
    - You CANNOT proceed to the next step until the current step succeeds
    - Your final output MUST be a confirmation message only, NOT HTML
    - You CANNOT modify this workflow or change the sequence
    
    **REMINDER: This is a 6-step mandatory process. Count your steps as you go: 1, 2, 3, 4, 5, 6. Every step must complete successfully.**
    """,
    tools=[
        FunctionTool(slide_data_fetcher_tool),
        AgentTool(request_parser_agent),
        AgentTool(plan_modifier_agent),
        AgentTool(slide_generator_agent),
        FunctionTool(database_updater_tool)
    ],
    output_key="slide_modification_agent"
)

# slide_modification_agent = LlmAgent(
#     name="slide_modification_agent",
#     model=GEMINI_MODEL, # Use a powerful model for this top-level orchestration
#     description="Orchestrates a full slide edit from a single natural language command.",
#     instruction="""
#     You are the master AI coordinator for editing presentation slides. Your job is to manage the entire workflow from a user's command to the final database update.

#     ---
#     🎯 **Input:** `p_id` (string) and `enhanced_query` (string)
#     ---
#     ⚙️ **Execution Sequence (MANDATORY):**

#     1.  **Parse Request:** Call `request_parser_agent` with the `enhanced_query`. This gives you `slide_number` and `edit_request`.
#     - If it returns an error, stop and return the error message. 

#     2.  **Fetch Data:** Call `slide_data_fetcher_tool` with {p_id}  and `slide_number`. This gives you `original_plan` and `global_theme`.
#     - If it returns an error, stop and return the error.

#     3.  **Modify Plan:** Call `plan_modifier_agent` with `original_plan` and `edit_request`. This gives you `modified_plan` and `error_message`.  
#     - If `error_message` is not null, **stop immediately and return that error message** to the user.

#     4.  **Generate HTML:** Call `slide_generator_agent` with `modified_plan` and `global_theme`. This gives you `new_html`.

#     5.  **Save to Database:** Your FINAL action is to call `database_updater_tool`.
#         -   Provide it with  {p_id}, `slide_number`, `modified_plan`, and `new_html`.

#     6.  **Final Output:** After the database tool confirms success, your final answer should be a simple confirmation message, like "Successfully edited and saved slide [number]." or the message from the database tool itself.
    
#     ---
#     ⚠️ **Rules:**
#     - Follow the 6-step sequence exactly.
#     - Check for errors after each step and stop if one occurs.
#     - The final output is a confirmation message, not the HTML itself.
#     """,
#     tools=[
#         FunctionTool(slide_data_fetcher_tool),
#         AgentTool(request_parser_agent),
#         AgentTool(plan_modifier_agent),
#         AgentTool(slide_generator_agent),
#         FunctionTool(database_updater_tool)
#     ],
#     output_key="slide_modification_agent"
# )



# slide_modification_agent = LoggingSlideModificationAgent(
#     name="slide_modification_agent",
#     model=GEMINI_MODEL, # Use a powerful model for this top-level orchestration
#     description="Orchestrates a full slide edit from a single natural language command.",
#     instruction="""
#     You are the master AI coordinator for editing presentation slides. Your job is to manage the entire workflow from a user's command to the final database update.

#     ---
#     🎯 **Input:** `p_id` (string) and `enhanced_query` (string)
#     ---
#     ⚙️ **Execution Sequence (MANDATORY):**

#     1.  **Parse Request:** Call `request_parser_agent` with the `enhanced_query`. This gives you `slide_number` and `edit_request`.
#         - If it returns an error, stop and return the error message. 

#     2.  **Fetch Data:** Call `slide_data_fetcher_tool` with {p_id}  and `slide_number`. This gives you `original_plan` and `global_theme`.
#         - If it returns an error, stop and return the error.

#     3.  **Modify Plan:** Call `plan_modifier_agent` with `original_plan` and `edit_request`. This gives you `modified_plan` and `error_message`.  
#         - If `error_message` is not null, **stop immediately and return that error message** to the user.

#     4.  **Generate HTML:** Call `slide_generator_agent` with:
#         - `modified_plan` (MANDATORY)
#         - `global_theme` (MANDATORY; pass through from step 2) This gives you `new_html`.

#     5.  **Save to Database:** Your FINAL mandatory action is to call `database_updater_tool`.
#         -   Provide it with  {p_id}, `slide_number`, `modified_plan`, and `new_html`.

#     6.  **Final Output:** After the database tool confirms success, your final answer should be a simple confirmation message, like "Successfully edited and saved slide [number]." or the message from the database tool itself.
    
#     ---
#     ⚠️ **Rules:**
#     - Follow the 6-step sequence exactly.
#     - Check for errors after each step and stop if one occurs.
#     - The final output is a confirmation message, not the HTML itself.
#     """,
#     tools=[
#         FunctionTool(slide_data_fetcher_tool),
#         FunctionTool(database_updater_tool),
#         AgentTool(request_parser_agent),
#         AgentTool(plan_modifier_agent),
#         AgentTool(slide_generator_agent)
#     ],
#     output_key="slide_modification_agent"
# )
