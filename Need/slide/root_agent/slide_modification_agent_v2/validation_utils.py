"""
Validation utilities for ensuring HTML structure preservation
"""

from bs4 import BeautifulSoup
import logging

logger = logging.getLogger(__name__)


def validate_structure(original_html: str, modified_html: str) -> bool:
    """
    Verify that only content changed, not HTML structure
    
    Args:
        original_html: Original HTML string
        modified_html: Modified HTML string
    
    Returns:
        bool: True if structure unchanged, False if structure was modified
    """
    try:
        orig_soup = BeautifulSoup(original_html, 'html.parser')
        mod_soup = BeautifulSoup(modified_html, 'html.parser')
        
        # Check 1: Same number and types of tags
        orig_tags = [tag.name for tag in orig_soup.find_all(True)]
        mod_tags = [tag.name for tag in mod_soup.find_all(True)]
        
        if orig_tags != mod_tags:
            logger.warning(f"Tag structure changed!")
            logger.warning(f"Original tags: {orig_tags}")
            logger.warning(f"Modified tags: {mod_tags}")
            return False
        
        # Check 2: Same classes and IDs
        orig_elements = orig_soup.find_all(True)
        mod_elements = mod_soup.find_all(True)
        
        for orig_el, mod_el in zip(orig_elements, mod_elements):
            # Check classes
            orig_classes = orig_el.get('class', [])
            mod_classes = mod_el.get('class', [])
            if orig_classes != mod_classes:
                logger.warning(f"Classes changed on {orig_el.name}")
                logger.warning(f"Original: {orig_classes}")
                logger.warning(f"Modified: {mod_classes}")
                return False
            
            # Check IDs
            orig_id = orig_el.get('id')
            mod_id = mod_el.get('id')
            if orig_id != mod_id:
                logger.warning(f"ID changed on {orig_el.name}")
                logger.warning(f"Original: {orig_id}")
                logger.warning(f"Modified: {mod_id}")
                return False
        
        # Check 3: Same tag hierarchy
        orig_structure = get_tag_tree(orig_soup)
        mod_structure = get_tag_tree(mod_soup)
        
        if orig_structure != mod_structure:
            logger.warning(f"HTML hierarchy changed!")
            logger.warning(f"Original structure: {orig_structure}")
            logger.warning(f"Modified structure: {mod_structure}")
            return False
        
        logger.info("✅ Structure validation passed - only content changed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Validation error: {e}")
        return False


def get_tag_tree(soup) -> str:
    """
    Create a structural fingerprint of the HTML
    
    Args:
        soup: BeautifulSoup object
    
    Returns:
        str: Structural fingerprint
    """
    def traverse(element, depth=0):
        if element.name:
            children = [traverse(child, depth+1) for child in element.children if hasattr(child, 'name') and child.name]
            if children:
                return f"{element.name}({','.join(children)})"
            return element.name
        return ""
    
    return traverse(soup)


def extract_content_structure(html: str) -> dict:
    """
    Extract content structure from HTML for modification context
    
    Args:
        html: HTML string
    
    Returns:
        dict: Extracted content structure
    """
    try:
        soup = BeautifulSoup(html, 'html.parser')
        
        # Extract title
        h1_tag = soup.find('h1')
        title = h1_tag.get_text(strip=True) if h1_tag else ""
        
        # Extract headings
        headings = [h.get_text(strip=True) for h in soup.find_all(['h2', 'h3', 'h4'])]
        
        # Extract paragraphs
        paragraphs = [p.get_text(strip=True) for p in soup.find_all('p') if p.get_text(strip=True)]
        
        # Extract lists
        lists = []
        for ul in soup.find_all(['ul', 'ol']):
            list_items = [li.get_text(strip=True) for li in ul.find_all('li')]
            lists.append({
                "type": ul.name,
                "items": list_items
            })
        
        return {
            "title": title,
            "headings": headings,
            "paragraphs": paragraphs,
            "lists": lists
        }
        
    except Exception as e:
        logger.error(f"Error extracting content structure: {e}")
        return {
            "title": "",
            "headings": [],
            "paragraphs": [],
            "lists": []
        }


