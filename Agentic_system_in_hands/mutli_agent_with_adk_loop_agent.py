# main.py
import asyncio
import uuid
import os
from typing import AsyncGenerator
from dotenv import load_dotenv
from google.adk.agents import LoopAgent, LlmAgent, BaseAgent
from google.adk.events import Event, EventActions
from google.adk.agents.invocation_context import InvocationContext
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
import re
import json
# Load environment variables from .env file
load_dotenv()


# class ProjectStatusToState(BaseAgent):
#     name = "ProjectStatusToState"

#     async def _run_async_impl(self, context):
#         # Get last LLM message text from history (most recent event by ProcessingStep)
#         last_events = [e for e in context.session.events if e.author == "ProcessingStep" and e.content]
#         text = last_events[-1].content.parts[0].text if last_events else ""

#         # Extract the fenced JSON block ```json ... ```
#         m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
#         if m:
#             try:
#                 obj = json.loads(m.group(1))
#                 status = obj.get("status")
#                 if status:
#                     yield Event(
#                         author=self.name,
#                         content=f"status set to {status}",
#                         actions=EventActions(state_delta={"status": status})
#                     )
#                     return
#             except json.JSONDecodeError:
#                 pass

#         # Fallback: do nothing
#         yield Event(author=self.name, content="no status found")

# -------------------------
# Custom Agent Definition
# -------------------------
class ConditionChecker(BaseAgent):
    """A custom agent that checks for a 'completed' status in the session state."""

    name: str = "ConditionChecker"
    description: str = "Checks if a process is complete and signals the loop to stop."

    async def _run_async_impl(
        self, context: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """Checks state and yields an event to either continue or stop the loop."""
        state_values = context.session.state
        print("Current session state:", state_values)
        result = context.session.state.get("status_update", "pending")
        print("Result from previous agent:", result)
        result= result.strip() if isinstance(result, str) else result
        print("Stripped result:", result)
        result= json.loads(result) if isinstance(result, str) and result.startswith("{") else result
        print("Parsed result:", result)
        #status = result if isinstance(result, str) else ""
        if isinstance(result, dict) and result.get("status"):
            status = result.get("status")
        print("Checking status:", status)
        is_done = (status == "completed")

        if is_done:
            yield Event(author=self.name, actions=EventActions(escalate=True))
        else:
            yield Event(
                author=self.name,
                content=types.Content(role="model", parts=[types.Part(text="Condition not met, continuing loop.")])
            )


# -------------------------
# Define Agents
# -------------------------
process_step = LlmAgent(
    name="ProcessingStep",
    model="gemini-2.0-flash-exp",
    instruction=(
        "You are a step in a longer process. "
        "Perform your task about multiple steps and only give a json format with a 'status' key in the following not any other text. "
        " output format: {\"status\": \"completed\"}"
    ),
    output_key="status_update",
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
    # Initialize session service
    session_service = InMemorySessionService()
    SESSION_ID = str(uuid.uuid4())
    
    # Create session
    session = await session_service.create_session(
        app_name="status_app",
        user_id="user123",
        session_id=SESSION_ID,
        state={}  # optional initial state
    )
    
    # Create runner
    runner = Runner(
        app_name="status_app",
        agent=poller,
        session_service=session_service
    )
    
    print("🔄 Running StatusPoller...")
    
    # Create initial message
    msg = types.Content(role="user", parts=[types.Part(text="Start the process")])
    
    # Run the agent using the runner
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


if __name__ == "__main__":
    asyncio.run(main())
