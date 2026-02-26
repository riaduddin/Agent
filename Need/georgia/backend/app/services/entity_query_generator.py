# backend/app/services/entity_query_generator.py
import logging
from typing import List

logger = logging.getLogger(__name__)


def generate_entity_queries(entities: dict, max_queries: int = 8) -> List[str]:
    """
    Generates focused search queries based on dynamically extracted entities.
    Works with flexible entity structure - no hardcoded field names.
    
    Args:
        entities: Dictionary with dynamic entity structure:
            {
                "identifiers": ["1234", "EMP-5678"],
                "names": ["john smith"],
                "dates": {"specific": ["2024-01-15"], "ranges": [...]},
                "amounts": ["500.00"],
                "keywords": ["payroll", "check"],
                "entity_details": {...}
            }
        max_queries: Maximum number of entity queries to generate
    
    Returns:
        List of entity-focused search queries
    """
    if not entities:
        logger.warning("No entities provided for query generation")
        return []
    
    queries = []
    
    # Extract entity categories
    identifiers = entities.get("identifiers", [])
    names = entities.get("names", [])
    dates = entities.get("dates", {})
    specific_dates = dates.get("specific", [])
    date_ranges = dates.get("ranges", [])
    amounts = entities.get("amounts", [])
    keywords = entities.get("keywords", [])
    entity_details = entities.get("entity_details", {})
    
    # Strategy 1: Pure identifier queries (highest precision)
    # Example: "1234", "EMP-5678", "CHECK-001"
    for identifier in identifiers[:5]:  # Limit to top 5 identifiers
        queries.append(identifier)
        
        # Also try with context if we have keywords
        if keywords:
            main_keyword = keywords[0]
            queries.append(f"{main_keyword} {identifier}")
    
    # Strategy 2: Name-based queries
    for name in names[:3]:  # Limit to top 3 names
        queries.append(name)
        
        # Combine name with identifiers
        if identifiers:
            queries.append(f"{name} {identifiers[0]}")
        
        # Combine name with keywords
        if keywords:
            queries.append(f"{keywords[0]} {name}")
    
    # Strategy 3: Date + identifier combinations
    if specific_dates:
        main_date = specific_dates[0]
        year_month = main_date[:7] if len(main_date) >= 7 else main_date  # e.g., "2024-01"
        
        # Date alone
        queries.append(main_date)
        if len(main_date) >= 7:
            queries.append(year_month)
        
        # Date + identifier
        if identifiers:
            queries.append(f"{identifiers[0]} {main_date}")
            if len(main_date) >= 7:
                queries.append(f"{identifiers[0]} {year_month}")
        
        # Date + keyword
        if keywords and len(main_date) >= 7:
            queries.append(f"{keywords[0]} {year_month}")
    
    # Strategy 4: Amount-based queries
    if amounts:
        for amount in amounts[:2]:
            queries.append(f"amount {amount}")
            if identifiers:
                queries.append(f"{identifiers[0]} {amount}")
    
    # Strategy 5: Keyword combinations
    if keywords:
        # Primary keyword alone
        queries.append(keywords[0])
        
        # Keyword combinations
        if len(keywords) > 1:
            queries.append(f"{keywords[0]} {keywords[1]}")
    
    # Strategy 6: Use entity_details for type-aware queries
    # This allows the LLM's discovered entity types to influence query generation
    for entity_value, details in list(entity_details.items())[:5]:
        entity_type = details.get("type", "unknown")
        
        # Type-specific query patterns
        if "check" in entity_type.lower():
            queries.append(f"check {entity_value}")
            queries.append(f"check number {entity_value}")
        elif "employee" in entity_type.lower():
            queries.append(f"employee {entity_value}")
            queries.append(f"employee id {entity_value}")
        elif "case" in entity_type.lower():
            queries.append(f"case {entity_value}")
            queries.append(f"case number {entity_value}")
        elif "invoice" in entity_type.lower():
            queries.append(f"invoice {entity_value}")
            queries.append(f"invoice number {entity_value}")
        elif "purchase" in entity_type.lower() or "po" in entity_type.lower():
            queries.append(f"purchase order {entity_value}")
            queries.append(f"PO {entity_value}")
        elif "permit" in entity_type.lower():
            queries.append(f"permit {entity_value}")
            queries.append(f"permit number {entity_value}")
        # Generic fallback
        else:
            queries.append(f"{entity_type} {entity_value}")
    
    # Strategy 7: Multi-entity combinations for complex queries
    # Example: name + identifier + date
    if names and identifiers and specific_dates:
        date_part = specific_dates[0][:7] if len(specific_dates[0]) >= 7 else specific_dates[0]
        queries.append(f"{names[0]} {identifiers[0]} {date_part}")
    
    # Deduplicate and normalize
    unique_queries = []
    seen = set()
    for q in queries:
        q_normalized = q.lower().strip()
        if q_normalized not in seen and q_normalized and len(q_normalized) > 1:
            seen.add(q_normalized)
            # Keep original casing for the actual query
            unique_queries.append(q.strip())
    
    # Limit to max_queries
    limited_queries = unique_queries[:max_queries]
    
    logger.info(f"Generated {len(limited_queries)} entity queries:")
    logger.info(f"  - From {len(identifiers)} identifiers, {len(names)} names, {len(specific_dates)} dates")
    logger.info(f"  - Queries: {limited_queries}")
    
    return limited_queries
