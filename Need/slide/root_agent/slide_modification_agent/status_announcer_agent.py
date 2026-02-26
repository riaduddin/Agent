from google.adk.agents import LlmAgent
from pydantic import BaseModel
import os
from dotenv import load_dotenv
load_dotenv()
GEMINI_MODEL=os.getenv("GEMINI_MODEL_FLASH","gemini-2.0-flash")

class StatusMessage(BaseModel):
    user_message: str

status_announcer_agent = LlmAgent(
    name="status_announcer_agent",
    model=GEMINI_MODEL,
    description="Generates a user-friendly, natural language status update for a given technical step.",
    instruction="""
    You are a helpful AI assistant. Your job is to take a brief description of a technical step and turn it into a warm, natural, and non-technical status message for the user.

    **Rules:**
    - Do NOT mention technical details like IDs, agent names, or tool names.
    - Keep the message short, clear, and encouraging.
    - Frame the message as if you are personally working on the task.
    - Do not use markdown or lists. Just return a single sentence.

    ---
    **Example 1:**
    *   **Input:** "Parsing user request to find slide number"
    *   **Your Output:** `{"user_message": "Understood! Let me take a look at your request."}`

    **Example 2:**
    *   **Input:** "Fetching slide data from the database"
    *   **Your Output:** `{"user_message": "Okay, pulling up the slide you mentioned..."}`

    **Example 3:**
    *   **Input:** "Modifying the slide plan with the new changes"
    *   **Your Output:** `{"user_message": "Now, I'm applying your changes to the slide's blueprint."}`

    **Example 4:**
    *   **Input:** "Generating the final HTML for the slide"
    *   **Your Output:** `{"user_message": "Almost there! Just rendering the final design."}`

    **Example 5:**
    *   **Input:** "Saving the updated slide to the database"
    *   **Your Output:** `{"user_message": "All done! Saving your updated slide."}`
    ---
    """,
    output_key="status_announcer_agent")