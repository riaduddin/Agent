import json
from database_connection import db # Your MongoDB connection
from planning_agent import plan_modifier_agent # Import your agents
from slide_generator_agent import slide_generator_agent
import logging
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
logger = logging.getLogger(__name__)

def edit_slide_orchestrator(p_id: str, slide_number: int, edit_request: str):
    """
    Orchestrates the editing of a single slide.
    
    Args:
        p_id: The ID of the presentation.
        user_id: The ID of the user (for security/scoping).
        slide_number: The user-facing slide number (e.g., 1, 2, 3).
        edit_request: The user's plain text instruction (e.g., "change the headline").
    """
    print(f"Starting edit for slide {slide_number} of presentation {p_id}...")
    
    # 1. FETCH DATA
    # --------------------------------
    # Convert user-facing number (1-based) to 0-based index
    slide_index = slide_number - 1

    # Fetch the specific slide to be edited
    #slide_doc = db.slides.find_one({"p_id": p_id, "slide_index": slide_index})
    slide_doc = db.slide_html.find_one({"p_id": p_id, "slide_index": slide_index})
    if not slide_doc:
        raise ValueError("Slide not found.")

    # Add this for debugging:
    logger.debug(f"Fetched slide document:\n{json.dumps(slide_doc, default=str, indent=2)}")

    if 'slide_plan' not in slide_doc:
        raise KeyError(f"'slide_plan' key is missing in slide document: {slide_doc}")

    # Fetch the presentation's global theme
    presentation_doc = db.presentations.find_one({"p_id": p_id})
    if not presentation_doc:
        raise ValueError("Presentation not found.")
        
    original_plan = slide_doc.get('slide_plan')
    if not original_plan:
        raise ValueError(f"Missing 'slide_plan' for p_id: {p_id}, slide_index: {slide_index}")

    global_theme = presentation_doc['global_theme']
    # 2. MODIFY THE PLAN
    # --------------------------------
    # Use the new agent to turn the text request into a new JSON plan
    print("Modifying slide plan with AI...")
    # The 'plan_modifier_agent' is invoked with the original plan and the user's text
    # We assume the agent returns a JSON string in its 'modified_plan_json' output key
    #logger.debug("testing the output from the planning agent")
    APP_NAME = "Customer Support"
    runner = Runner(
    session_service=InMemorySessionService(),
    agent=plan_modifier_agent,
    app_name=APP_NAME
    )

    # Run your agent via the runner
    response_events = runner.run(
        user_id="user123",
        session_id="sess1",
        new_message={"original_json": original_plan, "user_request": edit_request}
    )

    # Extract the final answer from events
    for event in response_events:
        if event.is_final_response():
            modified_plan_str = event.content.parts[-1].text
            break
    
    print(f"modified_plan: {modified_plan_str}")
    modified_plan = json.loads(modified_plan_str)

    print(f"modified_plan: {modified_plan}")

    logger.info("Entering into the planning agent")
    modified_plan_str = plan_modifier_agent.run({
        "original_json": original_plan,
        "user_request": edit_request
    })["modified_plan_json"]
    logger.info("Completed the planning agent")
    
    modified_plan = json.loads(modified_plan_str) # Convert the JSON string to a Python dict

    # 3. RE-GENERATE THE SLIDE HTML
    # --------------------------------
    # Reuse your existing slide_generator_agent
    print("Re-generating slide HTML...")
    generator_payload = {
        "request": {
            **modified_plan,  # Unpack all keys from the modified plan
            "global_theme": global_theme  # Add the global theme
        }
    }
    
    # The 'slide_generator_agent' is invoked with the complete, updated payload
    new_html_output = slide_generator_agent.invoke(generator_payload)["slide_generator_agent"]

    # 4. UPDATE THE DATABASE
    # --------------------------------
    print("Updating database with new slide content...")
    db.slide_html.update_one(
        {"_id": slide_doc["_id"]},
        {
            "$set": {
                "slide_plan": modified_plan,  # Save the new plan
                "html_body": new_html_output, # Save the new HTML
                "timestamp": "..." # Update the timestamp
            }
        }
    )
    
    print("Edit complete!")
    return new_html_output



edit_slide_orchestrator(p_id="5bd1b7e1-a04f-460c-a037-608c27c5d9c9", slide_number=1, edit_request="change the title to Riad")
