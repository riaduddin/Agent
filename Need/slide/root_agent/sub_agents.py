from google.adk.agents import LlmAgent
from dotenv import load_dotenv
import os
import time
load_dotenv()

# Set LiteLLM-specific environment variables
# LiteLLM looks for GEMINI_API_KEY (not GOOGLE_API_KEY) for Gemini models
if not os.getenv("GEMINI_API_KEY") and os.getenv("GOOGLE_API_KEY"):
    os.environ["GEMINI_API_KEY"] = os.getenv("GOOGLE_API_KEY")

GEMINI_MODEL=os.getenv("GEMINI_MODEL_FLASH","gemini-2.5-flash")
GEMINI_MODEL_PRO=os.getenv("GEMINI_MODEL_PRO","gemini-2.5-flash")

# Import LiteLLM for fallback support
from google.adk.models.lite_llm import LiteLlm
from google.adk.models.google_llm import Gemini
from google.genai import types

# Create a reusable LiteLLM model configuration with DeepSeek fallback
# This ensures all sub-agents can fall back to DeepSeek if Gemini fails
def get_model_with_fallback():
    """Returns a LiteLLM model with DeepSeek fallback configured."""
    return LiteLlm(
        model=f"gemini/{GEMINI_MODEL}",
        fallbacks=[
            "deepseek/deepseek-chat",
            "zai/glm-4.6",
        ],
        num_retries=3,
    )

def get_native_gemini_model():
    """
    Returns a native Gemini model (without LiteLLM wrapper).
    Use this for agents that need Google Search tool, which is incompatible with LiteLLM.
    Note: This model does NOT have DeepSeek fallback.
    """
    return Gemini(
        model=GEMINI_MODEL,
        retry_options=types.HttpRetryOptions(
            initial_delay=30, 
            attempts=3,
            exp_base=2.0,
            jitter=0.3,
            http_status_codes=[429, 500, 502, 503, 504]
        )
    )




def create_file_extractor_agent():
    return LlmAgent(
        name="file_extractor_agent",
        model=get_model_with_fallback(),
        description="Extracts clean, readable text content from a local file path (PDF, TXT, or DOCX).",
        instruction="""
You are an expert in extracting clean, readable text from uploaded document files. 
You will be given a local file path on the system — your job is to return the meaningful human-visible text extracted from it.

---
🎯 Input: `file_path` (string path to a saved file: PDF, TXT, or DOCX)
---
✅ Output:
- A clean, readable text string.
- Only extract human-visible content (not metadata or byte strings).
- Remove headers/footers/boilerplate if repetitive.
- Do not include any error messages or explanations in your output.
        """,
        output_key="extracted_text"
    )

def create_query_enhancer_agent():
    # Import google_search tool from ADK
    try:
        from google.adk.tools import google_search as _google_search
    except Exception:
        _google_search = None

    tools_list = []
    if _google_search is not None:
        tools_list.append(_google_search)

    # Note: Using native Gemini model instead of LiteLLM because Google Search tool
    # is not compatible with LiteLLM. This agent won't have DeepSeek fallback,
    # but it's necessary for Google Search functionality.
    return LlmAgent(
        name="query_enhancer_agent",
        model=get_native_gemini_model(),
        description=(
            "Refines and expands any presentation-related user request "
            "— whether for creating a new presentation, editing an existing slide, "
            "or modifying a specific block — into a detailed, actionable query."
        ),
        instruction="""
You enhance the user's presentation-related query into a precise, detailed, and context-rich instruction.

---
🎯 Input:
- `user_query` (string)

---
✅ Output:
- A rewritten, detailed query that is unambiguous, clearly states the task, and includes inferred details such as topic, style, tone, or intended outcome when possible.
- Do not add extra comments, explanations, or formatting beyond the rewritten query.
- Return only the enhanced query text, not JSON or structured data.

**Example:**

Input: "presentation about Tesla"
Output: "Create a professional presentation about Tesla's electric vehicle innovations, recent market performance, and strategic initiatives, focusing on their latest technological advancements and competitive positioning in the EV market."

Input: "slides on AI"
Output: "Create an informative presentation on current Artificial Intelligence trends, covering recent breakthroughs in generative AI, enterprise adoption, and practical applications across industries."
        """,
        output_key="enhanced_query",
        tools=tools_list if len(tools_list) > 0 else None,
    )



def create_query_classifier_agent():
    return LlmAgent(
        name="query_classifier_agent",
        model=get_native_gemini_model(),
        description="Classifies the user query into creation or editing category.",
        instruction="""
Classify the user query into one of the following actions:

- `create_presentation` - User wants to create a new presentation
- `edit_slide` - User wants to modify existing slide content
- `insert_slide` - User wants to add a new slide between existing slides
- `other` - Query doesn't fit the above categories

Keywords for insert_slide:
- "add", "insert", "between", "after slide X", "before slide Y"
- "add a slide about", "insert a timeline", "add between slide 3 and 4"
- "put a slide", "include a slide", "add another slide"

---
🎯 Input: `query` (string)
---
✅ Output format:
```json
{"action": "create_presentation"}  // or "edit_slide", "insert_slide", or "other"
  """,
    output_key="classification"
)


def create_topic_checker_agent():
    return LlmAgent(
        name="topic_checker_agent",
        model=get_native_gemini_model(),
        description="Detects whether a clear topic is present in the user query or file_context.",
        instruction="""
Your job is to determine if a specific topic for a presentation can be identified from either the user query or the file_context.

---
🎯 Inputs:
- `query`
- `file_context`: <file_context_info>
{file_context}
</file_context_info>
  (Content extracted from files uploaded by the user - may be None, empty, or contain information)

---
✅ Output:
Return only `true` if:
- A specific topic is clearly mentioned in the query, OR
- A specific topic can be identified from the file_context (when file_context is available and not empty)

Return only `false` if:
- The query is vague, missing, or needs clarification, AND
- The file_context is None/empty OR doesn't contain a clear topic

**Logic:**
1. First check the `query` - if it has a clear, specific topic, return `true`
2. If the query doesn't have a clear topic, check `file_context`:
   - If `file_context` is None, empty, or just whitespace, return `false`
   - If `file_context` has content, analyze it to see if a clear topic can be identified
   - If a clear topic is found in file_context, return `true`
3. Only return `false` if both query and file_context fail to provide a clear topic
        """,
        output_key="has_topic"
    )