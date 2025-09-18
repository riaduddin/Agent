import os
import asyncio
import json
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.tool_context import ToolContext
from google.genai.types import Content, Part
from google.adk.sessions.state import State
from google.adk.events.event import Event
import uuid
import time
from google.adk.events.event_actions import EventActions

async def persist_state_from_response_tool(current_state: dict, session_service: InMemorySessionService, app_name: str, user_id: str, session_id: str):
    session = await session_service.get_session(app_name=app_name, user_id=user_id, session_id=session_id)
    actions = EventActions(state_delta={**{State.USER_PREFIX + k: v for k, v in current_state.items()}})
    # Create a dummy event with state_delta
    event = Event(
        invocation_id=str(uuid.uuid4()),
        author="system",
        actions=actions,
        timestamp=time.time()
    )
    await session_service.append_event(session, event)

# ----------------- TOOLS -----------------
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

def show_state(tool_context: ToolContext) -> dict:
    """
    Returns the current session state with updated testing_list for inspection.
    """
    tool_context.state['testing_list'] = [1, 2, 3]
    tool_context.state['global_theme'] = 'light'
    # safer: convert to dict
    return {
        "current_state": tool_context.state.to_dict(),
        "persist_state": True # Indicate that state should be persisted
    }

# ----------------- MAIN -----------------
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
        state={"user:login_count": 0, "task_status": "idle", "global_theme": "dark"},
    )

    print("Initial state:", session.state)

    user_message = Content(parts=[Part(text="Hello, please show me my state!")])

    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=user_message,
    ):
        print(f"\n📌 Event from: {event.author}")

        #Inspect parts directly
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    print("💬 Agent text:", part.text)
                if part.function_call:
                    dump = part.function_call.model_dump() if hasattr(part.function_call, "model_dump") else part.function_call.dict()
                    print("📞 Function call:", json.dumps(dump, indent=2))
                if part.function_response:
                    dump = part.function_response.model_dump() if hasattr(part.function_response, "model_dump") else part.function_response.dict()
                    print("🔧 Function response:", json.dumps(dump, indent=2))
                    response_content = dump.get("response", {})
                    if response_content.get("persist_state", False):
                        print("💾 Persisting state as requested by tool...")
                        current_state = response_content.get("current_state", {})
                        await persist_state_from_response_tool(
                            current_state=current_state,
                            session_service=session_service,
                            app_name=app_name,
                            user_id=user_id,
                            session_id=session_id
                        )
                if part.code_execution_result:
                    # code_execution_result is usually a plain string or dict already
                    print("🖥️ Code exec result:", part.code_execution_result)

        # Convenience helpers
        # for call in event.get_function_calls():
        #     print("➡️ Detected tool call:", call)

        # for resp in event.get_function_responses():
        #     print("⬅️ Detected tool response:", json.dumps(resp, indent=2))

        # if event.is_final_response():
        #     print("✅ Final response reached")

    print("Final state:", session.state)

if __name__ == "__main__":
    asyncio.run(main())
