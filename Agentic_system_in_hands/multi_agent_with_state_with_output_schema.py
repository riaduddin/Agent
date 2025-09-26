# main.py
import asyncio
import uuid
from typing import AsyncGenerator, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from google.adk.agents import LoopAgent, LlmAgent, BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse, LlmRequest
from google.genai import types
import json

load_dotenv()

# -------------------------
# Pydantic output schema
# -------------------------
class StatusResult(BaseModel):
    status: str = Field(..., description="Process status", pattern="^(completed|pending)$")

# -------------------------
# Custom Agent Definition
# -------------------------
class ConditionChecker(BaseAgent):
    """Reads session.state['status_update'] (a dict or Pydantic-like dict) and escalates if completed."""
    name: str = "ConditionChecker"
    description: str = "Checks if a process is complete and signals the loop to stop."

    async def _run_async_impl(
        self, context: InvocationContext
    ) -> AsyncGenerator[Event, None]:

        # The result was saved by process_step under this key
        result = context.session.state.get("status_update")

        # If ADK stores a model instance, it will be dict-like; handle both safely:
        if result is None:
            status = "pending"
        elif isinstance(result, StatusResult):
            status = result.status
        elif isinstance(result, dict):
            status = result.get("status", "pending")
        else:
            # If some adapter serialized it as JSON string:
            try:
                data = json.loads(result) if isinstance(result, str) else {}
                status = data.get("status", "pending")
            except Exception:
                status = "pending"

        print("Current session state snapshot:", dict(context.session.state))
        print("Derived status:", status)

        if status == "completed":
            # Optionally mirror a flat key too
            yield Event(
                author=self.name,
                actions=EventActions(escalate=True, state_delta={"status": "completed"})
            )
        else:
            yield Event(
                author=self.name,
                content=types.Content(role="model", parts=[types.Part(text="Condition not met, continuing loop.")])
            )

# -------------------------
# Callbacks (access state via CallbackContext)
# -------------------------
def before_model_logger(callback_context: CallbackContext, llm_request: LlmRequest) -> Optional[LlmResponse]:
    """Example: read state before the model call; you could inject hints or guardrails here."""
    st = callback_context.state  # dict-like
    last_status = st.get("status_update")
    count = callback_context.state.get("checking", 0)
    iteration_count = callback_context.state.get("iterative", 0)
    callback_context.state["checking"] = count + 1
    callback_context.state["iterative"] = iteration_count + 1
    print(f"[before_model] Agent={callback_context.agent_name} | prior status_update={last_status}")
    print(f"[before_model] Agent={callback_context.agent_name} | iterative={iteration_count + 1}")
    # Return None to proceed with the normal LLM call; or return LlmResponse to short-circuit.
    return None

def after_model_logger(callback_context: CallbackContext, llm_response: LlmResponse) -> Optional[LlmResponse]:
    """Example: inspect the model response; you could normalize or enforce structured format here."""
    text = ""
    count = callback_context.state.get("checking", 0)
    iteration_count = callback_context.state.get("iterative", 0)
    
    if llm_response and llm_response.content and llm_response.content.parts:
        text = llm_response.content.parts[0].text or ""
    print(f"[after_model] Agent={callback_context.agent_name} | model_text='{text}'")
    print(f"[after_model] Agent={callback_context.agent_name} | checking={count}")
    print(f"[after_model] Agent={callback_context.agent_name} | iterative={iteration_count}")
    print(f"[after_model] Agent={callback_context.agent_name} | status_update={callback_context.state.get('status_update', {})}")
    return None  # Return a modified LlmResponse to override, or None to keep as-is.

# -------------------------
# Define Agents
# -------------------------
process_step = LlmAgent(
    name="ProcessingStep",
    model="gemini-2.0-flash-exp",
    instruction=(
        "You are a step in a longer, multi-step process. "
        "Current iteration: {iterative}. "
        "If the iteration is 3 or higher, the process status should be completed. "
        "Otherwise, it should be pending. "
        "Return ONLY JSON that matches the schema (no extra text): {\"status\": \"completed\" | \"pending\"}."
    ),
    output_schema=StatusResult,     # <— Pydantic model for validated structured output
    output_key="status_update",     # <— ADK saves the validated result in session.state under this key
    before_model_callback=before_model_logger,
    after_model_callback=after_model_logger,
)

poller = LoopAgent(
    name="StatusPoller",
    max_iterations=10,
    sub_agents=[
        process_step,
        ConditionChecker(),
    ],
)

# -------------------------
# Entry Point
# -------------------------
async def main():
    session_service = InMemorySessionService()
    SESSION_ID = str(uuid.uuid4())
    initial_state = {
        "checking": 1,
        "iterative": 0  # Initialize iteration counter
    }

    # Create a fresh session (state is shared across sub-agents)
    await session_service.create_session(
        app_name="status_app",
        user_id="user123",
        session_id=SESSION_ID,
        state=initial_state  # initial state
    )

    runner = Runner(
        app_name="status_app",
        agent=poller,
        session_service=session_service
    )

    print("🔄 Running StatusPoller...")
    msg = types.Content(role="user", parts=[types.Part(text="Start the process")])

    async for event in runner.run_async(
        user_id="user123",
        session_id=SESSION_ID,
        new_message=msg,
    ):
        if event.content:
            print(f"{event.author}: {event.content}")
        if event.actions and event.actions.escalate:
            print("✅ Loop terminated: status completed.")
            break
    updated_session = await session_service.get_session(app_name="status_app", user_id="user123", session_id=SESSION_ID)
    print(f"Updated state: {updated_session.state}")

if __name__ == "__main__":
    asyncio.run(main())
