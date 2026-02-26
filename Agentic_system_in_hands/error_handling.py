import asyncio
import uuid
import logging
from typing import AsyncGenerator, Dict, Any, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from google.adk.agents import Agent, SequentialAgent, BaseAgent, LlmAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.tool_context import ToolContext
from google.genai import types

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------
# Custom Tools Definition
# -------------------------

def get_precise_location_info(tool_context: ToolContext, address: str) -> Dict[str, Any]:
    """
    Get precise location information for a given address.
    This tool simulates a precise location lookup service that might fail.
    
    Args:
        tool_context: The ADK tool context for accessing session state
        address: The address to look up
        
    Returns:
        Dict containing location information or error details
    """
    logger.info(f"Attempting precise location lookup for: {address}")
    
    # Update session state to track tool usage
    tool_context.state["last_tool_used"] = "get_precise_location_info"
    
    # Simulate potential failures based on address content
    if "invalid" in address.lower() or "error" in address.lower():
        logger.error(f"Precise location lookup failed for: {address}")
        error_result = {
            "success": False,
            "error": "Address not found in precise location database",
            "address": address,
            "error_type": "NOT_FOUND"
        }
        # Update session state with error
        tool_context.state["primary_location_failed"] = True
        tool_context.state["errors"] = tool_context.state.get("errors", []) + [error_result]
        return error_result
    
    # Simulate network/service failures randomly
    import random
    if random.random() < 0.3:  # 30% chance of failure
        logger.error(f"Network error during precise location lookup for: {address}")
        error_result = {
            "success": False,
            "error": "Network timeout - service unavailable",
            "address": address,
            "error_type": "NETWORK_ERROR"
        }
        # Update session state with error
        tool_context.state["primary_location_failed"] = True
        tool_context.state["errors"] = tool_context.state.get("errors", []) + [error_result]
        return error_result
    
    # Successful lookup
    logger.info(f"Precise location lookup successful for: {address}")
    success_result = {
        "success": True,
        "address": address,
        "coordinates": {"lat": 40.7128, "lng": -74.0060},
        "formatted_address": f"Precise location for {address}",
        "confidence": "high",
        "source": "precise_location_service"
    }
    # Update session state with success
    tool_context.state["location_result"] = success_result
    tool_context.state["primary_location_failed"] = False
    return success_result

def get_general_area_info(tool_context: ToolContext, city: str) -> Dict[str, Any]:
    """
    Get general area information for a city as a fallback option.
    This tool is more reliable but provides less precise information.
    
    Args:
        tool_context: The ADK tool context for accessing session state
        city: The city name to look up
        
    Returns:
        Dict containing general area information
    """
    logger.info(f"Getting general area info for city: {city}")
    
    # Update session state to track tool usage
    tool_context.state["last_tool_used"] = "get_general_area_info"
    tool_context.state["fallback_used"] = True
    
    # This tool is more reliable and rarely fails
    result = {
        "success": True,
        "city": city,
        "general_info": f"General information about {city}",
        "population": "Approximately 8.4 million",
        "area_type": "Urban metropolitan area",
        "confidence": "medium",
        "source": "general_area_service"
    }
    
    # Update session state with fallback result
    tool_context.state["location_result"] = result
    return result

# -------------------------
# Error Handling Agent
# -------------------------
class ErrorHandler(BaseAgent):
    """Custom agent that handles errors and updates session state accordingly."""
    name: str = "ErrorHandler"
    description: str = "Monitors and handles errors in the agent workflow"

    async def _run_async_impl(self, context: InvocationContext) -> AsyncGenerator[Event, None]:
        """Handle errors and update session state for error tracking."""
        
        # Check for errors in the session state
        errors = context.session.state.get("errors", [])
        primary_location_failed = context.session.state.get("primary_location_failed", False)
        
        logger.info(f"ErrorHandler checking state - errors: {len(errors)}, primary_failed: {primary_location_failed}")
        
        # Update error tracking state
        if primary_location_failed:
            context.session.state["fallback_triggered"] = True
            logger.info("Fallback mechanism triggered due to primary location failure")
            
            yield Event(
                author=self.name,
                actions=EventActions(
                    state_delta={
                        "fallback_triggered": True,
                        "error_count": len(errors)
                    }
                ),
                content=types.Content(
                    role="model", 
                    parts=[types.Part(text="Error detected - fallback mechanism activated")]
                )
            )
        else:
            yield Event(
                author=self.name,
                content=types.Content(
                    role="model", 
                    parts=[types.Part(text="No errors detected - proceeding normally")]
                )
            )

# -------------------------
# Agent Definitions
# -------------------------

# Agent 1: Primary location handler with error handling
primary_handler = LlmAgent(
    name="primary_handler",
    model="gemini-2.0-flash-exp",
    instruction="""
You are a location lookup specialist. Your job is to get precise location information.

1. Use the get_precise_location_info tool with the user's provided address
2. The tool will automatically handle state updates for success/failure cases
3. Report the results clearly to the user

Always handle errors gracefully and provide clear feedback about what happened.
    """,
    tools=[get_precise_location_info]
)

# Agent 2: Fallback handler with error recovery
fallback_handler = LlmAgent(
    name="fallback_handler",
    model="gemini-2.0-flash-exp",
    instruction="""
You are a fallback location handler. Your job is to provide alternative location information when the primary lookup fails.

1. Check if the primary location lookup failed by looking at state["primary_location_failed"]
2. If it is True:
   - Extract the city from the user's original query
   - Use the get_general_area_info tool with the city name
   - The tool will automatically update the state
3. If it is False, do nothing and indicate no fallback was needed

Always log your actions and provide clear status updates.
    """,
    tools=[get_general_area_info]
)

# Agent 3: Response formatter with comprehensive error handling
response_agent = LlmAgent(
    name="response_agent",
    model="gemini-2.0-flash-exp",
    instruction="""
You are a response formatter that presents location information to the user.

1. Review the location information stored in state["location_result"]
2. Check for any errors in state["errors"]
3. Present the information clearly and concisely to the user
4. If there are errors, explain what went wrong and what was done to recover
5. If state["location_result"] does not exist or is empty:
   - Apologize that location could not be retrieved
   - Explain what errors occurred
   - Suggest alternative approaches

Always be transparent about any issues encountered and recovery attempts made.
    """,
    tools=[]  # This agent only reasons over the final state
)

# -------------------------
# Sequential Agent with Error Handling
# -------------------------
robust_location_agent = SequentialAgent(
    name="robust_location_agent",
    sub_agents=[primary_handler, ErrorHandler(), fallback_handler, ErrorHandler(), response_agent]
)

# -------------------------
# Main Execution Function
# -------------------------
async def run_location_lookup_with_error_handling(user_query: str) -> None:
    """
    Run the location lookup with comprehensive error handling and logging.
    
    Args:
        user_query: The user's location query
    """
    session_service = InMemorySessionService()
    session_id = str(uuid.uuid4())
    
    # Initialize session with error tracking state
    initial_state = {
        "errors": [],
        "primary_location_failed": False,
        "fallback_triggered": False,
        "fallback_used": False,
        "location_result": None,
        "query": user_query
    }
    
    try:
        # Create session
        await session_service.create_session(
            app_name="location_lookup_app",
            user_id="user123",
            session_id=session_id,
            state=initial_state
        )
        
        # Create runner
        runner = Runner(
            app_name="location_lookup_app",
            agent=robust_location_agent,
            session_service=session_service
        )
        
        logger.info(f"Starting location lookup for query: {user_query}")
        
        # Prepare user message
        user_message = types.Content(
            role="user", 
            parts=[types.Part(text=user_query)]
        )
        
        # Run the agent with error handling
        async for event in runner.run_async(
            user_id="user123",
            session_id=session_id,
            new_message=user_message,
        ):
            if event.content and event.content.parts:
                content_text = "".join([part.text for part in event.content.parts if part.text])
                if content_text.strip():
                    print(f"🤖 {event.author}: {content_text}")
            
            # Handle escalation events (errors or completion)
            if event.actions:
                if event.actions.escalate:
                    logger.info(f"Escalation event from {event.author}")
                if event.actions.state_delta:
                    logger.info(f"State update from {event.author}: {event.actions.state_delta}")
        
        # Get final session state for analysis
        final_session = await session_service.get_session(
            app_name="location_lookup_app",
            user_id="user123", 
            session_id=session_id
        )
        
        # Log final state
        logger.info("Final session state:")
        for key, value in final_session.state.items():
            logger.info(f"  {key}: {value}")
            
    except Exception as e:
        logger.error(f"Critical error in location lookup: {e}")
        print(f"❌ Critical error occurred: {e}")

# -------------------------
# Test Scenarios
# -------------------------
async def test_error_scenarios():
    """Test various error scenarios to demonstrate error handling."""
    
    test_cases = [
        "Find location for 123 Main Street, New York",  # Normal case
        "Find location for invalid address error",      # Simulated error case
        "Find location for 456 Oak Avenue, Los Angeles", # Normal case
        "Find location for error prone address",        # Another error case
    ]
    
    print("🧪 Testing Error Handling Scenarios\n")
    print("=" * 60)
    
    for i, test_query in enumerate(test_cases, 1):
        print(f"\n📍 Test Case {i}: {test_query}")
        print("-" * 40)
        
        try:
            await run_location_lookup_with_error_handling(test_query)
        except Exception as e:
            logger.error(f"Test case {i} failed with error: {e}")
            print(f"❌ Test case {i} failed: {e}")
        
        print("-" * 40)
        
        # Add delay between tests
        await asyncio.sleep(1)
    
    print("\n✅ Error handling test scenarios completed!")

# -------------------------
# Main Entry Point
# -------------------------
async def main():
    """Main function to demonstrate error handling in ADK agents."""
    print("🚀 ADK Error Handling Demonstration")
    print("=" * 50)
    
    try:
        # Run test scenarios
        await test_error_scenarios()
        
        # Interactive mode
        print("\n" + "=" * 50)
        print("🎯 Interactive Mode - Enter your location queries:")
        print("(Type 'quit' to exit)")
        
        while True:
            try:
                user_input = input("\n📍 Enter location query: ").strip()
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                if user_input:
                    await run_location_lookup_with_error_handling(user_input)
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error(f"Interactive mode error: {e}")
                print(f"❌ Error: {e}")
                
    except Exception as e:
        logger.error(f"Main execution error: {e}")
        print(f"❌ Fatal error: {e}")

if __name__ == "__main__":
    asyncio.run(main())