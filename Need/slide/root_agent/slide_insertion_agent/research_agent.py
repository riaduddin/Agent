"""
Research Agent Sub-Agent
Handles research tasks using Google Search and Qdrant retrieval
"""

from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool, google_search
import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add root directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
# Import Qdrant retrieval tool
try:
    from tools.qdrant_retrieval import retrieve_research_tool
except ImportError:
    retrieve_research_tool = None

load_dotenv()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
from google.adk.models.google_llm import Gemini
from google.genai import types

def research_topic(topic: str, search_query: str, user_id: str, p_id: str) -> str:
    """
    Research a topic using Google Search and Qdrant retrieval
    
    Args:
        topic: The topic to research
        search_query: Specific search query
        user_id: User ID for Qdrant search
        p_id: Presentation ID for Qdrant search
    
    Returns:
        JSON string with research results
    """
    print(f"🔍 RESEARCH: Searching for information about '{topic}'")
    try:
        research_results = {
            "topic": topic,
            "search_query": search_query,
            "google_results": None,
            "qdrant_results": None,
            "combined_insights": []
        }
        
        # Try Qdrant first (existing presentation research)
        if retrieve_research_tool:
            try:
                print(f"📚 Searching Qdrant database for existing research...")
                qdrant_result = retrieve_research_tool(
                    search_query=search_query,
                    user_id=user_id,
                    p_id=p_id,
                    limit=6
                )
                research_results["qdrant_results"] = qdrant_result
                print(f"✅ Qdrant search completed")
            except Exception as e:
                print(f"⚠️ Qdrant search failed: {e}")
        
        # Use Google Search for additional information
        try:
            print(f"🌐 Searching Google for current information...")
            google_result = google_search(search_query)
            research_results["google_results"] = google_result
            print(f"✅ Google search completed")
        except Exception as e:
            print(f"⚠️ Google search failed: {e}")
        
        # Combine insights
        insights = []
        if research_results["qdrant_results"]:
            insights.append("Found relevant research from presentation database")
        if research_results["google_results"]:
            insights.append("Found current information from web search")
        
        research_results["combined_insights"] = insights
        
        result = json.dumps({
            "success": True,
            "research_results": research_results,
            "insights": insights
        })
        print(f"✅ Research completed successfully")
        return result
        
    except Exception as e:
        print(f"❌ Error during research: {e}")
        return json.dumps({
            "success": False,
            "error": f"Research error: {str(e)}"
        })


# Create function tool
research_topic_tool = FunctionTool(research_topic)


def create_research_agent() -> LlmAgent:
    """
    Create the research agent
    """
    tools_list = [research_topic_tool, google_search]
    
    # Add Qdrant retrieval tool if available
    if retrieve_research_tool:
        tools_list.append(retrieve_research_tool)
    
    return LlmAgent(
        name="research_agent",
        model=Gemini(
            model=GEMINI_MODEL,
            retry_options=types.HttpRetryOptions(initial_delay=30, attempts=3,exp_base=2.0,jitter=0.3,http_status_codes=[429, 500, 502, 503, 504])
        ),
        description="Research agent that searches for information using Google Search and Qdrant database",
        tools=tools_list,
        instruction="""
You are a research agent specialized in gathering information for slide content.

When called with a research request:

1. **Analyze the research need**: Understand what information is required
2. **Search Qdrant database**: Look for existing research from the presentation context
3. **Search Google**: Find current, up-to-date information on the topic
4. **Combine insights**: Synthesize information from both sources
5. **Return structured results**: Provide organized research findings

## Research Strategy:
- **Qdrant first**: Use existing presentation research (faster, contextual)
- **Google second**: Get current information and new insights
- **Synthesize**: Combine both sources for comprehensive coverage

## Output Format:
Return structured JSON with:
- Research results from both sources
- Key insights and findings
- Recommendations for slide content

Always provide comprehensive, well-organized research that can be used to enhance slide content.
        """
    )
