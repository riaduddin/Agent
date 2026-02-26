from google.adk.agents import LlmAgent
from google.adk.tools.agent_tool import AgentTool
from .keyword_research.agent import keyword_agent
from .search_query.agent import search_query_agent
from dotenv import load_dotenv
import os
from root_agent.sub_agents import get_model_with_fallback
load_dotenv()
GEMINI_MODEL=os.getenv("GEMINI_MODEL_FLASH","gemini-2.0-flash")
from datetime import datetime
from google.adk.tools import google_search
# Get current date dynamically
current_year = datetime.now().year
current_month = datetime.now().strftime("%B")
current_date = datetime.now().strftime("%Y-%m-%d")


def create_keyword_research_agent():
    """
    Factory function to create the KeywordResearchAgent with dynamic date context and file context awareness.
    This allows the agent to prioritize file content when available for keyword research.
    """
    return LlmAgent(
    name="KeywordResearchAgent",
    model=get_model_with_fallback(),
    description="Orchestrates time-aware keyword research and search query generation with file context priority for presentation planning.",
    instruction=f"""
**Current Date Context: {current_date} (Year: {current_year})**

You have access to presentation_spec state with enhanced fields:
- topic, presentation_type, tone, color_theme, slide_count, audience_type
- duration_minutes, complexity_level, key_message, visual_style, content_focus

**PRIORITY CONTENT SOURCES:**
1. {{file_context}} (if available in state): PRIMARY source for keyword generation
2. **extracted_content** (if available in state): SECONDARY source for context
3. **presentation_spec**: FALLBACK source when no file content available

**Content Priority Logic:**
- **If {{file_context}} is available**: Generate keywords primarily from file content themes, topics, and domain-specific terms
- **If no file_context but extracted_content available**: Use extracted_content to guide keyword focus
- **If neither available**: Use presentation_spec for keyword generation

**Tasks:**
1. Check session state for {{file_context}} and extracted_content availability
2. Use keyword_agent to extract time-aware 'keywords', 'topics', 'goals' (returns keyword_extraction)
3. Use search_query_agent to generate relevant search queries (returns search_queries)
4. Extract the final search queries from search_query_agent output and return them


**File Context Integration:**
- When {{file_context}} is present, extract domain-specific terminology
- Focus on topics and themes mentioned in the uploaded content
- Generate searches that complement and expand on file content
- Use file content to determine technical depth and expertise level

**Temporal Intelligence:**
- Include recent developments and latest research
- Consider contemporary business/academic context
- Avoid outdated references or old timeframes

**Process Flow:**
1. Check state for {{file_context}} and extracted_content
2. Call keyword_agent with content priority → get keyword_extraction with keywords, topics, goals
3. Pass keyword_extraction to search_query_agent → get search_queries array (must contain 8-10 queries)
4. Return the search_queries as your final keywords output

**Output:** Return ONLY this JSON structure (no markdown, no extra text):

```json
{{"keywords": ["query1", "query2", "query3", "query4", "query5", "query6", "query7", "query8", "query9", "query10"]}}
```

The array must contain between 8-10 queries that are immediately useful for finding current, relevant information that complements the file_context (when available) for the specified presentation type and audience.
""",
    tools=[AgentTool(keyword_agent), AgentTool(search_query_agent)],
    output_key="keywords"
)