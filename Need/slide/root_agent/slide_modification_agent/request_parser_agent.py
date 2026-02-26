from google.adk.agents import LlmAgent
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv
load_dotenv()
GEMINI_MODEL=os.getenv("GEMINI_MODEL_FLASH","gemini-2.0-flash")
# Define the structured output we want from this parser
from pydantic import BaseModel, Field
from typing import Optional, Annotated

# No need for conint anymore. We will use Annotated.

class ParsedEditRequest(BaseModel):
    # This is the correct modern syntax.
    # We are "annotating" the base type 'int' with a Field constraint.
    slide_number: Optional[Annotated[int, Field(ge=1)]] = Field(
        default=None, 
        description="The slide number the user wants to edit."
    )
    
    edit_request: Optional[str] = Field(
        default=None, 
        description="The specific change the user wants to make to the slide."
    )
    
    error_message: Optional[str] = Field(
        default=None, 
        description="An error message if the user's request is unclear or invalid."
    )

request_parser_agent = LlmAgent(
    name="request_parser_agent",
    model=GEMINI_MODEL,
    description="Parses a user's natural language sentence to extract the slide number and the specific edit request.",
    instruction="""
    You are an expert at understanding user requests for slide editing. Your task is to analyze a single sentence and extract the slide number and the specific edit.

    - The slide number must be a positive integer.
    - The edit request is the description of the change.

    **Error Handling Rules:**
    1.  If the user's request **does not contain a slide number**, you MUST set the `error_message` field to "I'm sorry, I couldn't understand which slide you want to edit. Please specify a slide number, like 'on slide 3...'." Do not attempt to guess.
    2.  If the user's request is **completely out of context** (e.g., "what's the weather like?", "write me a poem"), you MUST set the `error_message` field to "I'm sorry, I can only help with editing presentation slides. Please ask me to make a change to a specific slide."
    3.  If there is no error, the `error_message` field should be null.

    You must return a structured JSON object.

    ---
    **Example 1 (Success):**
    *   **User Input:** "On slide 3, change the headline."
    *   **Your Output:** `{"slide_number": 3, "edit_request": "change the headline", "error_message": null}`

    **Example 2 (Missing Slide Number):**
    *   **User Input:** "Change the headline to 'Welcome'."
    *   **Your Output:** `{"slide_number": null, "edit_request": null, "error_message": "I'm sorry, I couldn't understand which slide you want to edit. Please specify a slide number, like 'on slide 3...'"}`

    **Example 3 (Out of Context):**
    *   **User Input:** "Tell me a joke."
    *   **Your Output:** `{"slide_number": null, "edit_request": null, "error_message": "I'm sorry, I can only help with editing presentation slides. Please ask me to make a change to a specific slide."}`
    ---
    """,
    output_key="request_parser_agent")