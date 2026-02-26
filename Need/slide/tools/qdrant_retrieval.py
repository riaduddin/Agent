"""
Qdrant Retrieval Tool for accessing stored browser research data
"""
from google.adk.tools import FunctionTool
from tools.qdrant_utils import get_qdrant_manager
import logging

logger = logging.getLogger(__name__)


def retrieve_research_context(query: str, user_id: str, p_id: str, limit: int = 6) -> str:
    """
    Retrieve relevant research context from Qdrant vector database.
    
    This tool searches the stored browser research data using semantic similarity
    and filters by user_id and presentation_id (p_id) to retrieve only relevant
    research for the current presentation.
    
    Args:
        query: The search query or topic to find relevant research for
        user_id: User ID to filter results  
        p_id: Presentation ID to filter results
        limit: Maximum number of chunks to retrieve (default: 6)
        
    Returns:
        A formatted string containing relevant research data with sources
        
    Example:
        >>> retrieve_research_context("AI trends 2025", "user123", "pres456", limit=5)
        "Research Context:
        
        1. [Keyword: AI Revolution 2025]
        AI is transforming industries with 78% of companies using AI...
        Sources: microsoft.com, forbes.com
        
        2. [Keyword: Machine Learning]
        Machine learning adoption increased by 35% in 2024...
        Sources: techcrunch.com
        ..."
    """
    # Retry logic for connection errors (WinError 64 on Windows)
    max_retries = 3
    last_error = None
    
    for attempt in range(max_retries):
        try:
            qdrant = get_qdrant_manager()
            results = qdrant.retrieve_research_data(
                query=query,
                user_id=user_id,
                p_id=p_id,
                limit=limit
            )
            
            if not results:
                return f"No research context found for query: '{query}' (user: {user_id}, p_id: {p_id})"
            
            # Prioritize file_context chunks and deduplicate by text
            # Filter using file_context metadata field for precise filtering
            file_ctx = [r for r in results if r.get('file_context') is True or r.get('keyword') == 'file_context']
            others = [r for r in results if not (r.get('file_context') is True or r.get('keyword') == 'file_context')]

            ordered = file_ctx + others

            seen = set()
            deduped = []
            for r in ordered:
                key = (r.get('text') or '').strip().lower()
                if not key or key in seen:
                    continue
                seen.add(key)
                deduped.append(r)
                if len(deduped) >= limit:
                    break

            # Format results
            formatted_output = "**Retrieved Research Context:**\n\n"
            for i, result in enumerate(deduped, 1):
                formatted_output += f"**{i}. [{result.get('keyword','context')}]** (Relevance: {result.get('score',0):.3f})\n"
                formatted_output += f"{result.get('text','')}\n"
                if result.get('sources'):
                    try:
                        sources_text = ", ".join([s.get('title','') for s in result['sources'][:3]])
                        if sources_text.strip():
                            formatted_output += f"_Sources: {sources_text}_\n"
                    except Exception:
                        pass
                formatted_output += "\n"
            
            logger.info(f"✅ Retrieved {len(results)} chunks for query '{query}'")
            return formatted_output
            
        except (ConnectionError, ConnectionResetError, OSError) as e:
            last_error = e
            if attempt < max_retries - 1:
                import time
                wait_time = 1 * (2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
                logger.warning(f"⚠️ Connection error on attempt {attempt + 1}/{max_retries}: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.error(f"❌ Max retries reached after connection errors: {e}")
                return f"Failed to retrieve research after {max_retries} connection attempts. Please check your network and services."
        except Exception as e:
            error_msg = f"Failed to retrieve research context: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return error_msg
    
    # If we get here, all retries failed with connection errors
    return f"Failed to retrieve research after {max_retries} attempts due to connection issues: {last_error}"


# Create the tool
retrieve_research_tool = FunctionTool(
    func=retrieve_research_context)
    # description="""
    # Retrieves relevant browser research data from the vector database.
    # Use this tool to access previously collected research information for a presentation.
    # The tool performs semantic search and returns the most relevant research chunks
    # filtered by user_id and presentation_id (p_id).
    
    # This is useful when you need to:
    # - Access research data collected by the browser agent
    # - Find specific information about topics covered in the research
    # - Get contextual information with sources for content creation
    
    # The tool returns formatted research text with sources and relevance scores.
    # """


