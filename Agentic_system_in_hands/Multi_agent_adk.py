import asyncio
from typing import AsyncGenerator
from google.adk.agents import LoopAgent, LlmAgent, BaseAgent
from google.adk.events import Event, EventActions
from google.adk.agents.invocation_context import InvocationContext
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
# from google.adk.agents.invocation_context import InvocationContext

# print(InvocationContext.model_json_schema())

class ConditionChecker(BaseAgent):
    """A custom agent that checks for a 'completed' status in the session state."""
    name: str = "ConditionChecker"
    description: str = "Checks if a process is complete and signals the loop to stop."

    async def _run_async_impl(self, context: InvocationContext) -> AsyncGenerator[Event, None]:
        status= context.session.state.get("status", "")
        print("doing its task")
        is_done= (status == "completed")
        if is_done:
            yield Event(author=self.name, actions=EventActions(escalate=True))
        yield Event(
            author=self.name,
            content=types.Content(role="model", parts=[types.Part(text="Condition not met, continuing loop.")])
        )

process_step= LlmAgent(
    name="ProcessStep",
    model="gemini-2.0-flash",
    instruction="you are a step in a longer process. If you are the fourth step, update session state by setting 'status' to 'completed'."
)

pollar= LoopAgent(
    name="StatusPoller",
    max_iterations=10,
    sub_agents=[process_step, ConditionChecker()]
    )

async def main():
    session_service = InMemorySessionService()
    import uuid
    SESSION_ID = str(uuid.uuid4()) 
    # Initialize session + runner
    session = await session_service.create_session(
        app_name="status_app",
        user_id="user123",
        session_id=SESSION_ID,
        state={}   # optional initial state
    )   

    runner = Runner(
        app_name="status_app",
        agent=pollar,
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
            print(f"✅ {event.author} signaled completion, stopping loop.")


if __name__ == "__main__":
    asyncio.run(main())