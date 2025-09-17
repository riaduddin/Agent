import os
import asyncio
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.tool_context import ToolContext
from google.genai.types import Content, Part

# Define a tool function that updates state via tool_context
def log_user_login(tool_context: ToolContext) -> dict:
    """
    Tracks user login and updates session state.
    """
    state = tool_context.state
    login_count = state.get("user:login_count", 0) + 1
    state["user:login_count"] = login_count
    state["task_status"] = "active"
    state["user:last_login_ts"] = __import__("time").time()
    state["temp:validation_needed"] = True

    return {
        "status": "success",
        "message": f"User login tracked. Total logins: {login_count}.",
    }

# Another tool to show current state
def show_state(tool_context: ToolContext) -> dict:
    """
    Returns the current session state for inspection.
    """
    return {"current_state": dict(tool_context.state)}

async def main():
    agent = LlmAgent(
        name="Greeter",
        model="gemini-2.0-flash",
        instruction=(
            "When the user says hello, respond with a greeting. "
            "Also call the `log_user_login` tool to track logins, then call `show_state` "
            "to display the updated session state."
        ),
        output_key="last_greeting",
        tools=[log_user_login, show_state]
    )

    app_name, user_id, session_id = "tool_app", "user3", "session3"
    session_service = InMemorySessionService()
    runner = Runner(agent=agent, app_name=app_name, session_service=session_service)

    session = await session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        state={"user:login_count": 0, "task_status": "idle"},
    )

    print("Initial state:", session.state)

    user_message = Content(parts=[Part(text="Hello, please show me my state!")])

    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=user_message,
    ):
        # You can inspect events if the event objects have attributes for tool outputs;
        # but since you said ToolResponseEvent etc. aren't there, you might just rely
        # on the final response or the built-in logging ADK may provide.
        pass

    print("Final state:", session.state)

if __name__ == "__main__":
    asyncio.run(main())
