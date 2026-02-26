from app.llm.gemini_api_key_client import gemini_client
import json
import logging
from typing import Dict, Any, Optional
from app import config
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

class SearchRouterService:
    """
    Analyzes user queries to extract search filters for Entity-Centric Retrieval.
    Determines if the query targets a specific entity, asset, or timeframe.
    """

    _model = None

    @classmethod
    def _get_model(cls):
        """Returns the gemini_client if available."""
        if not gemini_client.is_available():
            logger.warning("Gemini API Key not configured for SearchRouterService.")
            return None
        return gemini_client

    @classmethod
    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=5), reraise=True)
    def analyze_query(cls, query: str) -> Dict[str, Any]:
        """
        Extracts search filters from a user query.
        
        Args:
            query: The user's natural language query.

        Returns:
            Dict containing:
            - is_entity_search (bool): True if filters were found.
            - filters (dict): The extracted filters (entity_name, vehicle_tag, etc.) used for 'restricts'.
        """
        model = cls._get_model()
        if not model:
            return {"is_entity_search": False, "filters": {}}

        prompt = f"""
        Analyze this search query. Does it target a specific Entity, Vehicle, Asset, or Document via an Identifier?
        
        If YES, extract the specific identifiers.
        If NO (it's a general question like "summarize the policy"), return empty filters.

        Target Attributes to Extract:
        - entity_name: Specific Person or Company Name
        - entity_id: ID numbers (NID, License, Invoice No, Tax ID)
        - vehicle_tag: Car Tag, Plate Number, VIN
        - doc_type: If asking for a specific *type* (e.g., "Show me his License", "Get the Invoice")
        - date: Specific year or date (e.g., "from 2015") -> extract as 'year' or 'date'

        Input Query: "{query}"

        Return JSON format:
        {{
            "is_entity_search": true/false,
            "filters": {{
                "entity_name": "...",
                "vehicle_tag": "...",
                "year": "..." 
            }}
        }}
        Omit empty filter keys.
        """

        try:
            response = model.generate_content(
                prompt,
                temperature=0.0,
                response_mime_type="application/json"
            )
            
            cleaned_text = response.text.strip()
            # Handle potential markdown code blocks
            if cleaned_text.startswith("```json"):
                cleaned_text = cleaned_text[7:-3]
            elif cleaned_text.startswith("```"):
                cleaned_text = cleaned_text[3:-3]
            
            result = json.loads(cleaned_text)
            
            # Sanitization: Ensure filters dict exists and values are strings
            if not result.get("filters"):
                result["filters"] = {}
            
            # Clean up the filters
            clean_filters = {}
            for k, v in result["filters"].items():
                if v and isinstance(v, str) and v.lower() not in ["none", "n/a", "null"]:
                    clean_filters[k] = v
            
            result["filters"] = clean_filters
            
            # Force is_entity_search false if no filters (double check)
            if not clean_filters:
                result["is_entity_search"] = False

            logger.info(f"Router Analysis for '{query}': {result}")
            return result

        except Exception as e:
            logger.error(f"Error analyzing query '{query}': {e}")
            return {"is_entity_search": False, "filters": {}}
