from google.adk.agents import LlmAgent
from dotenv import load_dotenv
from pymongo import MongoClient
from google.adk.tools.agent_tool import AgentTool
from root_agent.sub_agents import create_file_extractor_agent, create_query_enhancer_agent, create_query_classifier_agent
from root_agent.sub_agents import create_topic_checker_agent
from root_agent.slide_creation_agent.slide_creation_agent import pipeline
from root_agent.slide_modification_agent_v2 import multi_slide_modification_orchestrator
from root_agent.slide_insertion_agent import slide_insertion_orchestrator
from google.adk.tools import FunctionTool
import logging
logger = logging.getLogger(__name__)
import os
import time
load_dotenv()
from root_agent.sub_agents import get_native_gemini_model

# GEMINI_API_KEY is already handled by core or env configuration

# Verify DeepSeek API key is available for fallback
if os.getenv("DEEPSEEK_API_KEY"):
    logger.info("✅ DEEPSEEK_API_KEY found - fallback enabled")
else:
    logger.warning("⚠️ DEEPSEEK_API_KEY not found - fallback may not work")

GEMINI_MODEL=os.getenv("GEMINI_MODEL_FLASH","gemini-2.5-flash")
GEMINI_MODEL_PRO=os.getenv("GEMINI_MODEL_PRO","gemini-2.5-flash")

from google.adk.models.lite_llm import LiteLlm
import litellm

# Disable proxy for LiteLlm to avoid connection issues
# This must be done BEFORE any HTTP clients are initialized
# Remove all proxy environment variables
for proxy_var in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy']:
    os.environ.pop(proxy_var, None)

# Set NO_PROXY to disable proxy for all hosts (wildcard)
os.environ['NO_PROXY'] = '*'
os.environ['no_proxy'] = '*'

# Also set empty string for ALL_PROXY to explicitly disable
os.environ['ALL_PROXY'] = ''
os.environ['all_proxy'] = ''

SlideOrchestrationAgent = LlmAgent(
    name="slide_orchestration_agent",
    model=get_native_gemini_model(),
    description="Main controller for interpreting user intent and routing presentation-related tasks to the appropriate agents.",
    instruction="""
You are the central brain coordinating user requests for slide creation or editing.

---
🎯 Your Role:
- Start every response with a brief, friendly acknowledgment of the user's request (e.g., "Sure, I can help you with that!", "Absolutely, I'm on it!").
- Interpret the user's query and any extracted file content `file_context`={file_context}.
- Extract presentation content directly from user messages when available.
- Enhance vague requests using query, file context, and extracted content.
- Determine whether to create a new presentation or edit an existing one.
- Always check for topic first before proceeding with other operations.
- Maintain context across user turns using `ctx.session.state`.

---
🧠 Session State Rules (`ctx.session.state`):
- Expect {p_id} and {user_id}.
- May include {file_context} (from files) and `extracted_content` (from user message).
- Handle multi-turn topic recovery:
  - If topic is missing, set `awaiting_topic = True`.
  - When user responds, use that input as topic.

---
⚙️ Execution Logic:

1. **Check for Awaiting Topic:**
   - If `ctx.session.state.get("awaiting_topic") == True`:
     - Treat current user input as the missing topic.
     - Merge it with `original_query` in state, set as `enhanced_query`.
     - Set `ctx.session.state["awaiting_topic"] = False`.
     - Proceed directly to **Step 4 (Query Enhancement)**.

2. **Content Extraction from User Message:**
   - For all new requests, look for structured content (bullets, lists, Facts) in the user message.
   - If content is found, save as `ctx.session.state["extracted_content"]`.

3. **Topic Detection (Priority Check):**
   - Call `topic_checker_agent` with the current query.
   - If topic is missing:
     - Set `ctx.session.state["awaiting_topic"] = True`.
     - Save original query as `ctx.session.state["original_query"]`.
     - Respond: "Can you please specify the topic you'd like to use for this presentation?"
     - **STOP EXECUTION**.

4. **Query Enhancement:**
   - Call `query_enhancer_agent` with current query, `file_context`, and `extracted_content`.
   - Save the result as `enhanced_query`.

5. **Intent Classification:**
   - Call `query_classifier_agent` with `enhanced_query`.

6. **Task Routing:**
  - If `create_presentation`:
    - Call `transfer_to_agent(agent_name="SlideCreationPipeline")` with enhanced_query, file_context, and extracted_content.
  - If `edit_slide`:
    - Call `transfer_to_agent(agent_name="multi_slide_modification_orchestrator")`.
  - If `insert_slide`:
    - Call `transfer_to_agent(agent_name="slide_insertion_orchestrator")`.
  - If `other`:
    - Respond: "I can only help with creating presentations, editing slides, or inserting new slides..."

---
📌 Important Notes:
- **Topic checking is the FIRST priority** before any other operations.
- Do not extract files yourself — only use `file_context` if provided by backend.
- Distinguish between `file_context` (from files) and `extracted_content` (from user message).
    """,
    tools=[
        AgentTool(create_topic_checker_agent()), 
        AgentTool(create_query_enhancer_agent()),
        AgentTool(create_query_classifier_agent())
    ],
    sub_agents=[
        pipeline,                              
        multi_slide_modification_orchestrator, 
        slide_insertion_orchestrator
    ],
    output_key="slide_orchestration_result"
)