# backend/app/services/category_api_service.py
import logging
import requests
from typing import List, Dict, Optional, Tuple
from app import config

logger = logging.getLogger(__name__)

def get_api_auth_headers() -> Dict[str, str]:
    """
    Get authentication headers for the categories API.
    Uses X-API-Key header with JWT_SECRET_KEY from config.
    """
    headers = {
        "Content-Type": "application/json"
    }
    
    # Use JWT_SECRET_KEY as API key
    api_key = getattr(config, 'JWT_SECRET_KEY', None)
    if api_key:
        headers["X-API-Key"] = api_key
    else:
        logger.warning("JWT_SECRET_KEY not found in config, API request may fail")
    
    return headers

def fetch_categories_from_api() -> Optional[Dict]:
    """
    Fetches categories from the external API.
    Uses X-API-Key header with JWT_SECRET_KEY.
    Returns the full API response or None if failed.
    """
    try:
        api_url = getattr(config, 'CATEGORIES_API_URL', '')
        if not api_url:
            logger.warning("CATEGORIES_API_URL not configured, will use fallback categories")
            return None
        
        headers = get_api_auth_headers()
        
        if 'X-API-Key' not in headers:
            logger.warning("No X-API-Key available for categories API, request may fail")
        
        logger.info(f"Fetching categories from API: {api_url}")
        response = requests.get(api_url, headers=headers, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        categories_count = len(data.get('categories', []))
        logger.info(f"Successfully fetched {categories_count} categories from API")
        return data
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch categories from API: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching categories from API: {e}", exc_info=True)
        return None

def extract_category_codes(api_response: Dict) -> List[str]:
    """
    Extracts category codes from API response.
    Returns list of category codes (e.g., ["INVOICES", "CHECKS", "1099", ...])
    New API response structure: {"categories": [...]}
    """
    categories = api_response.get("categories", [])
    codes = []
    for cat in categories:
        if cat.get("code"):
            codes.append(cat["code"])
    return codes

def build_short_code_map(api_response: Dict) -> Dict[str, str]:
    """
    Builds a mapping from short_code to category code.
    Returns: {"CW": "CHILD_WELFARE_REPORTS", "CK": "CHECKS", ...}
    New API response structure: {"categories": [...]}
    """
    short_code_map = {}
    categories = api_response.get("categories", [])
    
    for cat in categories:
        category_code = cat.get("code")
        short_codes = cat.get("short_code", [])
        
        if category_code:
            # Map each short code to the category code
            for short_code in short_codes:
                if short_code:
                    # Convert to uppercase for case-insensitive matching
                    short_code_map[short_code.upper()] = category_code
    
    logger.info(f"Built short code map with {len(short_code_map)} mappings")
    return short_code_map

def get_categories_with_fallback() -> Tuple[List[str], Dict[str, str], List[Dict]]:
    """
    Gets categories from API with fallback to VALID_CATEGORIES.
    Uses X-API-Key header with JWT_SECRET_KEY for authentication.
    Returns: (list of category codes, short_code_map, full_categories_list)
    """
    # Try API
    api_response = fetch_categories_from_api()
    if api_response:
        codes = extract_category_codes(api_response)
        short_code_map = build_short_code_map(api_response)
        full_categories = api_response.get("categories", [])
        logger.info(f"Fetched categories from API: {len(codes)} codes, {len(short_code_map)} short code mappings")
        return codes, short_code_map, full_categories
    
    # Fallback to hardcoded categories
    logger.warning("API unavailable, using fallback VALID_CATEGORIES")
    fallback_codes = config.VALID_CATEGORIES
    # Return empty short_code_map and empty categories list for fallback
    return fallback_codes, {}, []

def get_categories_from_filename(file_name: str, categories: List[Dict]) -> List[str]:
    """
    Finds category names by checking short codes in the filename.
    Returns a list of category names if found, empty list otherwise.
    Case-insensitive matching.
    
    Args:
        file_name: The filename to check
        categories: List of category dictionaries with "code", "name", and "short_code" fields
    
    Returns:
        List of category names (strings) found in the filename
    """
    if not file_name or not categories:
        return []
    
    file_name_lower = file_name.lower()
    found_categories = set()
    
    for category in categories:
        short_codes = category.get("short_code", [])
        for code in short_codes:
            if code and code.lower() in file_name_lower:
                # category_name = category.get("name", category.get("code", "UNCATEGORIZED"))
                category_code = category.get("code", "UNCATEGORIZED")
                if category_code != "UNCATEGORIZED":
                    found_categories.add(category_code)
                    logger.info(f"Found short code '{code}' in filename '{file_name}', mapping to category '{category_code}'")
    
    return list(found_categories)

