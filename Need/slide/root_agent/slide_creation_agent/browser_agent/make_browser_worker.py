# from .brave_search import google_search_tool
# from .content_scrapper import content_scrapper
from google.adk.agents import LlmAgent
from dotenv import load_dotenv
load_dotenv()
import logging
logger = logging.getLogger(__name__)

from google.adk.tools import google_search

from root_agent.sub_agents import get_native_gemini_model

def make_browser_worker(keyword: str, idx: int) -> LlmAgent:
    return LlmAgent(
    name=f"browser_worker_{idx}",
    model=get_native_gemini_model(),
    description="Web search agent that performs Google searches and returns comprehensive information, data, and research findings.",
    instruction=f"""
    You are a web search agent. Your ONLY task is to search for comprehensive information and data about "{keyword}" using Google Search.

    CRITICAL INSTRUCTIONS:
    - Use the google_search tool to search for "{keyword}"
    - Return factual information, data, statistics, and research findings
    - Do NOT create presentations, slides, or formatted content
    - Do NOT provide slide outlines or presentation structures  
    - Do NOT format content as slides or presentations
    - Only provide raw research data, facts, and information

    Your response should contain:
    - Key facts and information about {keyword}
    - Statistical data and metrics
    - Research findings and studies
    - Market data and trends (if relevant)
    - Performance metrics and analytics
    - Source information from your searches
    - Raw research content and data points

    DO NOT create any presentation content, slide formats, or structured presentations.
    """,
    tools=[google_search],
    output_key=f"browser_summary_{idx}"
)