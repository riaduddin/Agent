"""
Insertion Request Parser Agent
Parses natural language requests for slide insertion
"""

from google.adk.agents import LlmAgent
import os
from dotenv import load_dotenv

load_dotenv()
GEMINI_MODEL = os.getenv("GEMINI_MODEL_FLASH", "gemini-2.5-flash")

from google.adk.models.google_llm import Gemini
from google.genai import types
def create_insertion_request_parser() -> LlmAgent:
    """
    Create the insertion request parser agent
    """
    return LlmAgent(
        name="insertion_request_parser",
        model=Gemini(
            model=GEMINI_MODEL,
            retry_options=types.HttpRetryOptions(initial_delay=30, attempts=3,exp_base=2.0,jitter=0.3,http_status_codes=[429, 500, 502, 503, 504])
        ),
        description="Parse natural language requests for slide insertion and extract structured details",
        instruction="""
You are an expert at parsing natural language requests for slide insertion.

Your task is to analyze the user's request and extract structured information.

COMMON REQUEST PATTERNS:
- "Add a slide about Tesla's challenges between slide 3 and 4"
- "Insert a timeline slide after slide 2"
- "Add a comparison slide before slide 5"
- "Insert a slide with this content: [full content]"

EXTRACTION RULES:

1. **Insertion Position** (insert_after_slide):
   - If "after slide X" or "after the Xth slide" → insert_after_slide = X
   - If "before slide X" → insert_after_slide = X - 1
   - If "between slide X and Y" → insert_after_slide = X (insert between them)
   - If no position specified → insert_after_slide = null (will append to end)

2. **Topic** (topic):
   - Extract the subject/topic the slide should cover
   - Look for patterns like "about...", "on...", "regarding...", "concerning..."
   - Can be null if not specified 

3. **Content** (content_provided):
   - If user provides detailed content or full text → store it here
   - If request is longer than brief instruction → likely contains content
   - Can be null if only topic is provided

4. **Slide Type** (slide_type):
   - comparison: compare, comparison, vs, versus, difference, contrast
   - timeline: timeline, chronology, sequence, steps, process, flow
   - data: data, statistics, numbers, metrics, chart, graph, analytics
   - list: list, bullets, points, items, checklist
   - overview: overview, summary, introduction, intro
   - conclusion: conclusion, summary, wrap up, ending, closing
   - Can be null if not specified

5. **User Instructions** (user_instructions):
   - Store the original user request exactly as provided

6. **Error Message** (error_message):
   - null if parsing successful
   - Error description if there's an issue (e.g., "Invalid position")

OUTPUT FORMAT:
Return ONLY valid JSON with these exact fields:
{
  "insert_after_slide": <int or null>,
  "content_provided": <string or null>,
  "topic": <string or null>,
  "slide_type": <string or null>,
  "user_instructions": <string>,
  "error_message": <string or null>
}

IMPORTANT:
- Return ONLY the JSON, no additional text
- Use null (not "null" as string) for missing values
- Topic is OPTIONAL - both topic and content_provided can be null
- If topic is missing, the orchestrator will handle prompting or content generation
        """,
        output_key="insertion_request_parser_output"
    )
