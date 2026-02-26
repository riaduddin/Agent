# backend/app/services/query_entity_extractor.py
import logging
import json
import time
import re
from app import config
from app.llm.gemini_api_key_client import gemini_client

logger = logging.getLogger(__name__)

def extract_searchable_entities(query: str) -> dict:
    """
    Dynamically extracts ALL searchable entities from user query using Gemini.
    Uses a flexible, document-agnostic approach that discovers entities without hardcoding types.
    
    Args:
        query: User's search query
    
    Returns:
        Dictionary with dynamically discovered entities:
        {
            "identifiers": ["1234", "EMP-5678", "CPS-2024-001"],  # Any IDs/numbers
            "names": ["john smith", "acme corp"],  # Person/org names
            "dates": {
                "specific": ["2024-01-15"],
                "ranges": [{"start": "2024-01-01", "end": "2024-01-31"}]
            },
            "amounts": ["500.00", "1250.75"],  # Monetary values
            "keywords": ["payroll", "check", "invoice"],  # Important keywords
            "entity_details": {  # Structured entity data with types
                "1234": {"type": "check_number", "value": "1234", "category": "identifier"},
                "EMP-5678": {"type": "employee_id", "value": "EMP-5678", "category": "identifier"}
            }
        }
    """
    default_structure = {
        "identifiers": [],
        "names": [],
        "dates": {"specific": [], "ranges": []},
        "amounts": [],
        "keywords": [],
        "entity_details": {}
    }

    try:
        if not gemini_client.is_available():
            logger.warning("Gemini client not available for entity extraction.")
            return default_structure

        prompt = f"""
You are an entity extraction expert. Analyze this search query and extract ALL searchable information.

DO NOT limit yourself to predefined categories. Extract ANY entity that could help find documents.

For EACH entity you find, determine:
1. What it is (person name, ID, date, amount, keyword, etc.)
2. Its normalized value
3. How it should be searched

Extract:
- **Identifiers**: ANY number, code, or ID (check numbers, employee IDs, case numbers, invoice numbers, license plates, SSN, passport numbers, account numbers, etc.)
- **Names**: People, organizations, vendors, departments, locations
- **Dates**: Specific dates or date ranges, in any format mentioned
- **Amounts**: Money, quantities, numeric values
- **Keywords**: Important document-related words (type of document, action, status)
- **Classifications**: Document categories or types mentioned
- **Categorical Data**: Any other structured information (colors, sizes, statuses, etc.)

Return a JSON object in this EXACT format:
{{
    "identifiers": [list of all IDs/numbers as strings],
    "names": [list of all person/org names, lowercase],
    "dates": {{
        "specific": [list of specific dates in YYYY-MM-DD format],
        "ranges": [list of date range objects with start/end in YYYY-MM-DD format]
    }},
    "amounts": [list of monetary values as strings without currency symbols],
    "keywords": [list of important searchable words, lowercase],
    "entity_details": {{
        "entity_value": {{"type": "auto_detected_type", "value": "normalized_value", "category": "identifier|name|date|amount|keyword"}}
    }}
}}

Rules:
- Extract EVERYTHING that could be searchable - be aggressive
- Normalize: IDs to uppercase, names to lowercase, dates to YYYY-MM-DD
- If no entities found for a category, use empty list/object
- Discover entity types dynamically - don't limit to known types
- For date ranges, if only month/year mentioned, create a range for that period

Query: "{query}"
"""
        
        t_start = time.time()
        
        response = gemini_client.generate_content(
            contents=prompt,
            model_name=config.GEMINI_OCR_MODEL_NAME,
            temperature=0.2, # Slightly higher to allow creative entity discovery
            response_mime_type="application/json"
        )
        
        if response and response.text:
            text = response.text.strip()
            
            # DEBUG: Log raw LLM response
            logger.info(f"RAW Gemini Entity Extraction Response: {text}")
            
            # Clean markdown fences if present
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            
            try:
                entities = json.loads(text)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON from Gemini: {e}")
                entities = {}
            
            # Merge with defaults
            for key in default_structure:
                if key not in entities:
                    entities[key] = default_structure[key]
            
            # DEBUG: Log extracted entities
            logger.info(f"LLM Extracted Entities: {entities.get('identifiers')}")
            
            logger.info(f"Extracted entities in {time.time() - t_start:.2f}s:")
            logger.info(f"  - Identifiers: {len(entities.get('identifiers', []))}")
            logger.info(f"  - Names: {len(entities.get('names', []))}")
            logger.info(f"  - Dates: {len(entities.get('dates', {}).get('specific', []))} specific, {len(entities.get('dates', {}).get('ranges', []))} ranges")
            logger.info(f"  - Amounts: {len(entities.get('amounts', []))}")
            logger.info(f"  - Keywords: {len(entities.get('keywords', []))}")
            logger.info(f"  - Details: {len(entities.get('entity_details', {}))}")
            
            return entities
        
        logger.warning(f"No entities extracted from query: {query}")
        return default_structure
        
    except Exception as e:
        logger.error(f"Failed to extract searchable entities: {e}", exc_info=True)
        return default_structure

