# ADK Error Handling Guide

This guide demonstrates comprehensive error handling patterns in the Google ADK (Agent Development Kit) framework using a location lookup service as an example.

## Overview

The `error_handling.py` file implements a robust error handling system that showcases:

1. **Custom Tool Error Simulation** - Tools that can fail in various ways
2. **Sequential Agent Error Handling** - Multi-agent workflows with error recovery
3. **State-Based Error Tracking** - Session state management for error monitoring
4. **Fallback Mechanisms** - Alternative approaches when primary methods fail
5. **Comprehensive Logging** - Detailed logging for debugging and monitoring

## Architecture

### Agent Structure

```
SequentialAgent
├── primary_handler (LlmAgent)
├── ErrorHandler (BaseAgent)
├── fallback_handler (LlmAgent)
├── ErrorHandler (BaseAgent)
└── response_agent (LlmAgent)
```

### Error Flow

1. **Primary Handler** attempts precise location lookup
2. **Error Handler** monitors for failures and updates state
3. **Fallback Handler** provides alternative location information
4. **Error Handler** tracks fallback usage
5. **Response Agent** formats final response with error context

## Key Components

### 1. Custom Tools with Error Simulation

```python
@tool
def get_precise_location_info(address: str) -> Dict[str, Any]:
    """Simulates a location service that can fail in various ways."""
    # Simulates different types of failures:
    # - Invalid addresses (NOT_FOUND)
    # - Network timeouts (NETWORK_ERROR)
    # - Random failures (30% chance)
```

### 2. Error Handler Agent

```python
class ErrorHandler(BaseAgent):
    """Monitors and handles errors in the agent workflow."""
    
    async def _run_async_impl(self, context: InvocationContext):
        # Checks session state for errors
        # Updates error tracking state
        # Triggers fallback mechanisms
```

### 3. State Management

The system uses session state to track:

- `errors[]` - List of all encountered errors
- `primary_location_failed` - Boolean flag for primary failure
- `fallback_triggered` - Boolean flag for fallback activation
- `fallback_used` - Boolean flag for fallback usage
- `location_result` - Final location information
- `query` - Original user query

### 4. Error Types Handled

1. **NOT_FOUND Errors** - Invalid or non-existent addresses
2. **NETWORK_ERROR** - Service timeouts and connectivity issues
3. **Random Failures** - Simulated service instability
4. **Critical Errors** - Unexpected exceptions in the workflow

## Error Handling Patterns

### Pattern 1: Tool-Level Error Handling

```python
@tool
def get_precise_location_info(address: str) -> Dict[str, Any]:
    try:
        # Simulate potential failures
        if "invalid" in address.lower():
            return {
                "success": False,
                "error": "Address not found",
                "error_type": "NOT_FOUND"
            }
        # ... rest of implementation
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": "CRITICAL_ERROR"
        }
```

### Pattern 2: Agent-Level Error Handling

```python
primary_handler = LlmAgent(
    instruction="""
    1. Use the get_precise_location_info tool
    2. If success=False, update session state:
       - Set state["primary_location_failed"] = True
       - Add error details to state["errors"]
    3. If successful, store result in state["location_result"]
    """
)
```

### Pattern 3: State-Based Error Recovery

```python
fallback_handler = LlmAgent(
    instruction="""
    1. Check state["primary_location_failed"]
    2. If True:
       - Extract city from original query
       - Use get_general_area_info tool
       - Store result in state["location_result"]
       - Set state["fallback_used"] = True
    """
)
```

### Pattern 4: Comprehensive Error Reporting

```python
response_agent = LlmAgent(
    instruction="""
    1. Review state["location_result"]
    2. Check state["errors"]
    3. Present information clearly
    4. Explain any errors and recovery attempts
    5. Suggest alternatives if needed
    """
)
```

## Usage Examples

### Basic Usage

```python
# Run a single location lookup
await run_location_lookup_with_error_handling("123 Main Street, New York")
```

### Testing Error Scenarios

```python
# Run comprehensive error tests
await test_error_scenarios()
```

### Interactive Mode

```python
# Start interactive session
await main()
```

## Error Scenarios Tested

1. **Normal Case**: Valid address lookup
2. **Invalid Address**: Address with "invalid" or "error" keywords
3. **Network Failure**: Random 30% failure simulation
4. **Fallback Recovery**: Automatic fallback to general area info

## Best Practices Demonstrated

### 1. Graceful Degradation

- Primary method fails → Fallback method activated
- Precise location unavailable → General area information provided
- Complete failure → Clear error message with suggestions

### 2. State Transparency

- All errors tracked in session state
- Clear error types and messages
- Recovery attempts documented

### 3. User Experience

- Transparent error reporting
- Helpful suggestions for alternatives
- No silent failures

### 4. Monitoring and Debugging

- Comprehensive logging at all levels
- State snapshots for debugging
- Error categorization for analysis

### 5. Resilient Architecture

- Multiple fallback mechanisms
- Error isolation between agents
- State-based error recovery

## Running the Example

### Prerequisites

```bash
pip install google-adk python-dotenv
```

### Environment Setup

Create a `.env` file with your Google AI API key:
```
GOOGLE_API_KEY=your_api_key_here
```

### Execution

```bash
python error_handling.py
```

## Expected Output

The system will:

1. Run automated test scenarios
2. Demonstrate various error conditions
3. Show fallback mechanisms in action
4. Provide interactive mode for custom testing
5. Display comprehensive logging and state information

## Key Takeaways

1. **Error Handling is Multi-Layered**: Tools, agents, and workflows all need error handling
2. **State is Critical**: Use session state to track errors and recovery attempts
3. **Fallbacks are Essential**: Always provide alternative approaches
4. **Transparency Matters**: Users should understand what went wrong
5. **Logging is Vital**: Comprehensive logging enables debugging and monitoring

This implementation serves as a comprehensive template for building robust, error-resilient agent systems using the ADK framework.
