# Code Patterns & Conventions

Best practices and common patterns used in the codebase.

---

## 🏗️ Project Structure

```
project-root/
├── main.py                       # FastAPI app entry point
├── app_sse.py                    # SSE endpoints
├── db.py                         # MongoDB operations
├── auth_middleware.py            # JWT authentication
├── qdrant_utils.py               # Qdrant management
├── *_tool.py                     # FunctionTool definitions
├── root_agent/
│   ├── agent.py                  # SlideOrchestrationAgent
│   ├── sub_agents.py             # Utility agents
│   └── slide_creation_agent/
│       ├── agent.py              # Main orchestration
│       ├── sub_agents/           # Planning, generation, quality
│       ├── browser_agent/        # Web search
│       └── keyword_research_agent/
└── memory_bank/                  # Documentation (this!)
```

---

## 🤖 Creating an Agent

### **Basic LlmAgent Pattern**

```python
from google.adk.agents import LlmAgent

def create_my_agent():
    """
    Create an agent that does X.
    
    Returns:
        LlmAgent: Configured agent instance
    """
    return LlmAgent(
        name="my_agent",                          # Unique identifier
        model="gemini-2.5-flash",                 # Model to use
        description="Does X for Y purpose",       # What it does
        instruction="""
You are an expert at X.

Your ONLY task is to Y.

You should NOT do Z.

Output Format:
Return a JSON object with the following structure:
{
  "field1": "value",
  "field2": ["list", "items"]
}

CRITICAL: Return ONLY raw JSON, no markdown, no explanations.
        """,
        tools=[tool1, tool2],                     # Optional tools
    )
```

### **Key Principles**

1. **Clear Name**: Use descriptive, unique names
2. **Explicit Instructions**: Say what to do AND what NOT to do
3. **Output Format**: Specify exact JSON structure if needed
4. **No Template Syntax**: Never use `{{variable}}` in instructions
5. **No Code Examples**: Avoid Python/CSS code blocks in instructions

---

## 🔄 Creating a Sequential Agent

```python
from google.adk.agents import SequentialAgent

def create_pipeline_agent():
    """Multi-step agent that runs sub-agents in sequence."""
    return SequentialAgent(
        name="pipeline_agent",
        description="Runs a multi-step process",
        sub_agents=[
            ("step1", create_step1_agent()),
            ("step2", create_step2_agent()),
            ("step3", create_step3_agent()),
        ]
    )
```

---

## ⚡ Creating a Parallel Agent

```python
from google.adk.agents import ParallelAgent

def create_parallel_agent(num_workers=5):
    """Run multiple agents simultaneously."""
    workers = [
        (f"worker_{i}", create_worker(i))
        for i in range(num_workers)
    ]
    
    return ParallelAgent(
        name="parallel_agent",
        description="Run workers in parallel",
        sub_agents=workers
    )
```

---

## 🛠️ Creating a FunctionTool

```python
from google.adk.agents import FunctionTool

def my_function(param1: str, param2: int) -> dict:
    """
    Function that agents can call.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        dict: Result data
    """
    # Implementation
    result = do_something(param1, param2)
    return {"result": result}

# Create tool
my_tool = FunctionTool(
    name="my_function",
    description="What this tool does and when to use it",
    func=my_function
)
```

### **Tool Best Practices**

1. **Clear Description**: Explain when to use the tool
2. **Type Hints**: Use proper type annotations
3. **Error Handling**: Catch and return errors gracefully
4. **Logging**: Log tool usage for debugging
5. **Idempotent**: Same input = same output

---

## 🔐 Authentication Pattern

### **Protecting an Endpoint**

```python
from fastapi import APIRouter, Depends
from auth_middleware import get_current_user, AuthenticatedUser

router = APIRouter()

@router.get("/my-endpoint")
async def my_endpoint(
    param: str,
    current_user: AuthenticatedUser = Depends(get_current_user),
    db = Depends(get_db)
):
    """
    Protected endpoint - requires JWT authentication.
    
    Args:
        param: Endpoint parameter
        current_user: Authenticated user (from JWT)
        db: Database connection
    """
    user_id = current_user.user_id
    # Use user_id for user-specific operations
    
    return {"data": "..."}
```

### **Optional Authentication**

```python
from auth_middleware import get_optional_user

@router.get("/public-endpoint")
async def public_endpoint(
    current_user: Optional[AuthenticatedUser] = Depends(get_optional_user)
):
    """
    Endpoint that works with or without authentication.
    """
    if current_user:
        # Authenticated behavior
        return {"user": current_user.user_id}
    else:
        # Public behavior
        return {"message": "Public data"}
```

---

## 📡 SSE Streaming Pattern

### **Basic SSE Endpoint**

```python
from sse_starlette.sse import EventSourceResponse
import asyncio

@router.post("/sse/my-stream")
async def stream_data(
    current_user: AuthenticatedUser = Depends(get_current_user)
):
    """Stream data via SSE."""
    
    async def event_generator():
        try:
            # Send progress updates
            yield {
                "event": "chunk",
                "data": json.dumps({
                    "message": "Processing...",
                    "progress": 10
                })
            }
            
            # Do work
            result = await do_work()
            
            # Send result
            yield {
                "event": "chunk",
                "data": json.dumps({
                    "message": "Complete!",
                    "result": result
                })
            }
            
            # Send done signal
            yield {
                "event": "done",
                "data": json.dumps({"status": "completed"})
            }
            
        except Exception as e:
            yield {
                "event": "error",
                "data": json.dumps({"error": str(e)})
            }
    
    return EventSourceResponse(event_generator())
```

---

## 🗄️ Database Patterns

### **MongoDB Operations**

```python
from db import get_db
from bson import ObjectId

# Insert
def create_presentation(db, user_id, p_id, data):
    """Create new presentation."""
    result = db.presentations.insert_one({
        "p_id": p_id,
        "user_id": user_id,
        **data,
        "created_at": datetime.utcnow()
    })
    return result.inserted_id

# Find one
def get_presentation(db, user_id, p_id):
    """Get presentation by p_id."""
    return db.presentations.find_one({
        "p_id": p_id,
        "user_id": user_id  # Always filter by user!
    })

# Update
def update_presentation(db, user_id, p_id, updates):
    """Update presentation."""
    return db.presentations.update_one(
        {"p_id": p_id, "user_id": user_id},
        {"$set": {
            **updates,
            "updated_at": datetime.utcnow()
        }}
    )

# Delete
def delete_presentation(db, user_id, p_id):
    """Delete presentation."""
    return db.presentations.delete_one({
        "p_id": p_id,
        "user_id": user_id
    })
```

### **Qdrant Operations**

```python
from qdrant_utils import QdrantManager

# Initialize
manager = QdrantManager()

# Store research
await manager.store_research_data(
    text="Research content...",
    user_id="user123",
    p_id="pres123",
    keyword="search query",
    source_url="https://...",
    title="Source Title"
)

# Retrieve research
results = manager.retrieve_research_data(
    query="search query",
    user_id="user123",
    p_id="pres123",
    top_k=10
)

# Delete data
manager.delete_presentation_data(
    user_id="user123",
    p_id="pres123"
)
```

---

## 🎨 Agent Instruction Templates

### **Research Agent**

```python
instruction="""
You are a research expert specializing in [DOMAIN].

Your ONLY task is to:
1. Search for comprehensive information about [TOPIC]
2. Extract key facts, data, and insights
3. Return well-structured research findings

You should NOT:
- Create presentations or slides
- Generate HTML or CSS
- Make assumptions without evidence

Tools Available:
- brave_search_tool: For web searches
- content_scrapper: For extracting content from URLs

Process:
1. Search for [TOPIC] using brave_search_tool
2. Scrape relevant URLs for detailed content
3. Synthesize findings into a comprehensive report

Output Format:
Return a detailed text report with:
- Key findings (bullet points)
- Important statistics and data
- Source citations
"""
```

### **Planning Agent**

```python
instruction="""
You are an expert presentation planner.

Your task is to create a slide outline (NOT full content).

Input:
- Presentation topic: [TOPIC]
- Audience: [AUDIENCE]
- Slide count: [COUNT]

Output Format:
Return a JSON array of slide outlines:
[
  {
    "slide_number": 1,
    "slide_purpose": "Introduce the topic",
    "slide_title": "Title Here",
    "suggested_type": "Title Slide",
    "search_query": "Query for retrieving relevant research",
    "content_guidance": "What should be on this slide",
    "required_elements": ["element1", "element2"],
    "fallback_keywords": ["keyword1", "keyword2"]
  }
]

CRITICAL OUTPUT REQUIREMENT:
Return ONLY raw JSON - no markdown code blocks, no explanations, 
no follow-up messages. Just the JSON array in your first and only response.
"""
```

### **Generation Agent**

```python
instruction="""
You are an expert HTML slide generator.

Your task is to create ONE slide based on the provided outline.

Process:
1. Use qdrant_retrieval_tool to get relevant research
2. Analyze the research and extract key information
3. (Optional) Use search_images_tool for visual elements
4. Generate complete HTML slide with inline CSS

HTML Requirements:
- Dimensions: 1280x720px (fixed)
- Complete <!DOCTYPE html> document
- Inline CSS only (no external stylesheets)
- Responsive font sizing: clamp(min, preferred, max)
- Prevent overflow: overflow: hidden, word-wrap: break-word
- Apply theme colors from provided specs

Do NOT:
- Execute Python code
- Interpret CSS as Python
- Use template variables like {{variable}}

Output:
Return ONLY the complete HTML code, nothing else.
"""
```

---

## 🧪 Testing Patterns

### **Testing an Agent**

```python
import asyncio
from google.adk.runners import Runner
from google.adk.agents import RunConfig

async def test_my_agent():
    """Test agent with sample input."""
    agent = create_my_agent()
    runner = Runner()
    
    # Run agent
    async for event in runner.run_agent(
        agent=agent,
        query="Test query",
        config=RunConfig()
    ):
        print(f"{event.author}: {event.text}")

if __name__ == "__main__":
    asyncio.run(test_my_agent())
```

### **Testing a Tool**

```python
def test_my_tool():
    """Test tool with sample inputs."""
    result = my_function("test_param", 123)
    assert "result" in result
    assert result["result"] is not None
    print("✅ Tool test passed")

if __name__ == "__main__":
    test_my_tool()
```

---

## 📝 Logging Patterns

### **Structured Logging**

```python
import logging

logger = logging.getLogger(__name__)

# Info
logger.info(f"✅ Operation completed: {result}")

# Warning
logger.warning(f"⚠️ Potential issue: {issue}")

# Error
logger.error(f"❌ Operation failed: {error}")

# Debug
logger.debug(f"🔍 Debug info: {data}")

# With exception
try:
    risky_operation()
except Exception as e:
    logger.exception(f"🚨 Exception occurred: {e}")
```

---

## 🔄 Error Handling Patterns

### **Robust Function**

```python
async def robust_operation(param: str) -> dict:
    """
    Operation with comprehensive error handling.
    """
    try:
        # Validate input
        if not param:
            raise ValueError("param cannot be empty")
        
        # Attempt operation
        result = await do_something(param)
        
        # Validate output
        if not result:
            logger.warning("Operation returned empty result")
            return {"status": "empty", "data": None}
        
        return {"status": "success", "data": result}
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return {"status": "error", "error": str(e)}
        
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return {"status": "error", "error": "Internal error"}
```

---

## 🎯 Best Practices Summary

### **Agents**
- ✅ Unique, descriptive names
- ✅ Clear, explicit instructions
- ✅ Specify what NOT to do
- ✅ Define output format
- ❌ No template variables
- ❌ No code examples in instructions

### **Tools**
- ✅ Type annotations
- ✅ Error handling
- ✅ Logging
- ✅ Idempotent operations

### **Authentication**
- ✅ Always use `Depends(get_current_user)`
- ✅ Filter by user_id
- ✅ Validate user permissions

### **Database**
- ✅ Always filter by user_id
- ✅ Use indexes
- ✅ Validate data before insert/update
- ✅ Handle errors gracefully

### **SSE**
- ✅ Send progress updates
- ✅ Send done/error signals
- ✅ Handle exceptions in generator
- ✅ Clean up resources

### **Error Handling**
- ✅ Try-except blocks
- ✅ Log errors
- ✅ Return meaningful error messages
- ✅ Don't expose internal details to users

---

**Related Documents**:
- `AGENT_CATALOG.md` - Existing agent examples
- `API_REFERENCE.md` - API patterns
- `TROUBLESHOOTING.md` - Common issues

