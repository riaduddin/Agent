"""
Image Search Tool for finding relevant images for slides
"""
from google.adk.tools import FunctionTool
import requests
import logging
import os
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Image search service configuration
IMAGE_SEARCH_URL = os.getenv("IMAGE_SEARCH_URL", "http://163.172.181.252:8010/search_images")


def search_images(search_queries: List[str], count_per_query: int) -> str:
    """
    Search for relevant images using the external image search service.
    
    This tool queries an image search API to find relevant images for slide content.
    Returns image URLs that can be embedded in presentation slides.
    
    Args:
        search_queries: List of search terms (e.g., ["AI technology", "healthcare innovation"])
        count_per_query: Number of images to return per query
        
    Returns:
        A formatted string containing image URLs, titles, and snippets
        
    Example:
        >>> search_images(["AI technology", "healthcare"], count_per_query=2)
        "**Image Results:**
        
        Query: AI technology
        1. AI Technology Concept
           URL: https://example.com/image1.jpg
           Snippet: AI Technology Concept illustration
        
        2. Modern AI Systems
           URL: https://example.com/image2.jpg
           Snippet: Modern AI Systems visualization
        
        Query: healthcare
        1. Healthcare Innovation
           URL: https://example.com/image3.jpg
           ..."
    """
    try:
        # Prepare request payload
        payload = {
            "search_images": search_queries,
            "options": {"count": count_per_query}
        }
        
        logger.info(f"🔍 Searching for images: {search_queries}")
        
        # Make API request
        response = requests.post(
            IMAGE_SEARCH_URL,
            json=payload,
            timeout=30
        )
        
        if response.status_code != 200:
            error_msg = f"Image search API returned status {response.status_code}"
            logger.error(f"❌ {error_msg}")
            return f"Failed to search images: {error_msg}"
        
        data = response.json()
        results = data.get("results", [])
        
        if not results:
            return f"No images found for queries: {search_queries}"
        
        # Format results
        formatted_output = "**Image Search Results:**\n\n"
        
        for result in results:
            query = result.get("query", "Unknown query")
            images = result.get("images", [])
            
            formatted_output += f"**Query: {query}**\n"
            
            if not images:
                formatted_output += "  No images found\n\n"
                continue
            
            for i, image in enumerate(images, 1):
                title = image.get("title", "Untitled")
                link = image.get("link", "No URL")
                snippet = image.get("snippet", "No description")
                
                formatted_output += f"{i}. {title}\n"
                formatted_output += f"   URL: {link}\n"
                formatted_output += f"   Description: {snippet}\n"
            
            formatted_output += "\n"
        
        logger.info(f"✅ Found images for {len(results)} queries")
        return formatted_output
        
    except requests.exceptions.Timeout:
        error_msg = "Image search API request timed out"
        logger.error(f"❌ {error_msg}")
        return f"Failed to search images: {error_msg}"
    except requests.exceptions.ConnectionError:
        error_msg = f"Cannot connect to image search service at {IMAGE_SEARCH_URL}"
        logger.error(f"❌ {error_msg}")
        return f"Failed to search images: {error_msg}. Make sure the service is running."
    except Exception as e:
        error_msg = f"Image search error: {str(e)}"
        logger.error(f"❌ {error_msg}")
        return f"Failed to search images: {error_msg}"


# Create the tool
search_images_tool = FunctionTool(
    func=search_images
)

