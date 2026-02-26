# backend/app/services/entity_normalizer.py
"""
Entity Normalization Utilities

Provides consistent normalization of extracted entity values for:
1. Vector search restrictions (must match exactly)
2. Firestore queries
3. Display formatting

All normalization is designed to maximize search recall while
maintaining precision.
"""

import re
import logging
from typing import Optional, Any, List, Dict
from datetime import datetime

logger = logging.getLogger(__name__)


class EntityNormalizer:
    """
    Normalizes extracted entity values for consistent search and storage.
    """
    
    # Amount buckets for range-based filtering
    AMOUNT_BUCKETS = [
        (0, 100, "0-100"),
        (100, 500, "100-500"),
        (500, 1000, "500-1000"),
        (1000, 5000, "1000-5000"),
        (5000, 10000, "5000-10000"),
        (10000, 50000, "10000-50000"),
        (50000, 100000, "50000-100000"),
        (100000, float('inf'), "100000+")
    ]
    
    @classmethod
    def normalize_name(cls, name: Optional[str]) -> Optional[str]:
        """
        Normalize a person or organization name for search.
        
        - Converts to lowercase
        - Removes extra whitespace
        - Removes common suffixes (Jr., Sr., III, etc.)
        - Keeps only alphanumeric and spaces
        
        Args:
            name: Raw name string
            
        Returns:
            Normalized name or None if invalid
        """
        if not name or not isinstance(name, str):
            return None
            
        # Convert to lowercase
        normalized = name.lower().strip()
        
        # Remove common suffixes
        suffixes = [' jr.', ' jr', ' sr.', ' sr', ' iii', ' ii', ' iv', ' esq.', ' esq', ' phd', ' md']
        for suffix in suffixes:
            if normalized.endswith(suffix):
                normalized = normalized[:-len(suffix)]
        
        # Remove extra whitespace
        normalized = ' '.join(normalized.split())
        
        # Remove special characters but keep spaces and alphanumeric
        normalized = re.sub(r'[^a-z0-9\s]', '', normalized)
        
        # Final cleanup
        normalized = ' '.join(normalized.split()).strip()
        
        return normalized if len(normalized) >= 2 else None
    
    @classmethod
    def normalize_id(cls, id_value: Optional[str]) -> Optional[str]:
        """
        Normalize an identifier (check number, employee ID, case number, etc.).
        
        - Converts to uppercase
        - Removes spaces and special characters
        - Keeps alphanumeric only
        
        Args:
            id_value: Raw ID string
            
        Returns:
            Normalized ID or None if invalid
        """
        if not id_value or not isinstance(id_value, str):
            return None
            
        # Convert to uppercase and remove whitespace
        normalized = id_value.upper().strip()
        
        # Remove all non-alphanumeric characters
        normalized = re.sub(r'[^A-Z0-9]', '', normalized)
        
        return normalized if len(normalized) >= 1 else None
    
    @classmethod
    def normalize_check_number(cls, check_num: Optional[str]) -> Optional[str]:
        """
        Normalize a check number specifically.
        Handles various formats: #1234, Check No. 1234, 001234, etc.
        
        Args:
            check_num: Raw check number string
            
        Returns:
            Normalized check number (digits only, leading zeros removed)
        """
        if not check_num or not isinstance(check_num, str):
            return None
        
        # Extract digits only
        digits = re.sub(r'[^0-9]', '', check_num)
        
        if not digits:
            return None
        
        # Remove leading zeros but keep at least one digit
        normalized = digits.lstrip('0') or '0'
        
        return normalized
    
    @classmethod
    def normalize_date(cls, date_str: Optional[str]) -> Optional[str]:
        """
        Normalize a date to ISO format (YYYY-MM-DD).
        
        Handles various formats:
        - 2024-12-15
        - 12/15/2024
        - December 15, 2024
        - 15-Dec-2024
        - etc.
        
        Args:
            date_str: Raw date string
            
        Returns:
            ISO formatted date string (YYYY-MM-DD) or None if unparseable
        """
        if not date_str or not isinstance(date_str, str):
            return None
            
        date_str = date_str.strip()
        
        # Common date formats to try
        formats = [
            "%Y-%m-%d",      # 2024-12-15
            "%m/%d/%Y",      # 12/15/2024
            "%m-%d-%Y",      # 12-15-2024
            "%d/%m/%Y",      # 15/12/2024 (European)
            "%B %d, %Y",     # December 15, 2024
            "%b %d, %Y",     # Dec 15, 2024
            "%d-%b-%Y",      # 15-Dec-2024
            "%d %B %Y",      # 15 December 2024
            "%Y/%m/%d",      # 2024/12/15
            "%m/%d/%y",      # 12/15/24
            "%Y%m%d",        # 20241215
        ]
        
        for fmt in formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue
        
        # Try to extract year at minimum
        year_match = re.search(r'\b(19|20)\d{2}\b', date_str)
        if year_match:
            return year_match.group(0)  # Return just the year
            
        return None
    
    @classmethod
    def extract_year(cls, date_str: Optional[str]) -> Optional[str]:
        """
        Extract just the year from a date string.
        
        Args:
            date_str: Date string or ISO date
            
        Returns:
            4-digit year string or None
        """
        if not date_str:
            return None
            
        # If already ISO format, extract year
        if re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
            return date_str[:4]
        
        # Try to find 4-digit year
        year_match = re.search(r'\b(19|20)\d{2}\b', str(date_str))
        if year_match:
            return year_match.group(0)
            
        return None
    
    @classmethod
    def normalize_amount(cls, amount: Optional[Any]) -> Optional[float]:
        """
        Normalize a monetary amount to a float.
        
        Handles:
        - $1,234.56
        - 1234.56
        - $1,234
        - 1,234.00
        
        Args:
            amount: Raw amount string or number
            
        Returns:
            Float amount or None if unparseable
        """
        if amount is None:
            return None
            
        if isinstance(amount, (int, float)):
            return float(amount)
            
        if not isinstance(amount, str):
            return None
            
        # Remove currency symbols, commas, spaces
        cleaned = re.sub(r'[$,\s]', '', amount.strip())
        
        # Handle parentheses for negative (accounting format)
        if cleaned.startswith('(') and cleaned.endswith(')'):
            cleaned = '-' + cleaned[1:-1]
        
        try:
            return float(cleaned)
        except ValueError:
            return None
    
    @classmethod
    def get_amount_bucket(cls, amount: Optional[float]) -> Optional[str]:
        """
        Get the bucket label for an amount (for range filtering).
        
        Args:
            amount: Normalized float amount
            
        Returns:
            Bucket label string (e.g., "1000-5000")
        """
        if amount is None:
            return None
            
        amount = abs(amount)  # Use absolute value for bucketing
        
        for min_val, max_val, label in cls.AMOUNT_BUCKETS:
            if min_val <= amount < max_val:
                return label
                
        return "100000+"  # Fallback for very large amounts
    
    @classmethod
    def normalize_phone(cls, phone: Optional[str]) -> Optional[str]:
        """
        Normalize a phone number.
        
        Args:
            phone: Raw phone string
            
        Returns:
            Normalized phone (digits only, 10 or 11 digits)
        """
        if not phone or not isinstance(phone, str):
            return None
            
        # Extract digits only
        digits = re.sub(r'[^0-9]', '', phone)
        
        # US phone numbers are 10 digits (or 11 with country code)
        if len(digits) == 10:
            return digits
        elif len(digits) == 11 and digits.startswith('1'):
            return digits[1:]  # Remove leading 1
        elif len(digits) >= 7:
            return digits  # Return as-is for other formats
            
        return None
    
    @classmethod
    def normalize_ssn_last4(cls, ssn: Optional[str]) -> Optional[str]:
        """
        Extract and normalize last 4 digits of SSN.
        
        Args:
            ssn: SSN string (full or partial)
            
        Returns:
            Last 4 digits only
        """
        if not ssn or not isinstance(ssn, str):
            return None
            
        # Extract all digits
        digits = re.sub(r'[^0-9]', '', ssn)
        
        # Return last 4 if we have enough
        if len(digits) >= 4:
            return digits[-4:]
            
        return None
    
    @classmethod
    def normalize_department(cls, dept: Optional[str]) -> Optional[str]:
        """
        Normalize a department name.
        
        Args:
            dept: Raw department string
            
        Returns:
            Normalized department name (lowercase, trimmed)
        """
        if not dept or not isinstance(dept, str):
            return None
            
        # Convert to lowercase and clean whitespace
        normalized = ' '.join(dept.lower().split()).strip()
        
        return normalized if len(normalized) >= 2 else None
    
    @classmethod
    def normalize_doc_type(cls, doc_type: Optional[str]) -> Optional[str]:
        """
        Normalize a document type to uppercase.
        
        Args:
            doc_type: Raw document type string
            
        Returns:
            Uppercase document type
        """
        if not doc_type or not isinstance(doc_type, str):
            return None
            
        return doc_type.upper().strip()
    
    @classmethod
    def normalize_entity_value(cls, value: Any, field_name: str) -> Optional[str]:
        """
        Normalize an entity value based on the field name.
        
        This is the main dispatch method that selects the appropriate
        normalization based on the field type.
        
        Args:
            value: Raw value
            field_name: Name of the field (e.g., "check_number", "employee_name")
            
        Returns:
            Normalized string value or None
        """
        if value is None:
            return None
            
        field_lower = field_name.lower()
        
        # Name fields
        if any(x in field_lower for x in ['name', 'payee', 'payer', 'recipient', 'traveler', 
                                           'caseworker', 'supervisor', 'preparer', 'approver',
                                           'author', 'reviewer']):
            return cls.normalize_name(str(value))
        
        # Check number (special handling)
        if 'check' in field_lower and ('number' in field_lower or 'num' in field_lower or field_lower == 'check_number'):
            return cls.normalize_check_number(str(value))
        
        # ID fields
        if any(x in field_lower for x in ['_id', 'id_', 'number', 'code', 'tin', 'ssn', 'badge']):
            if 'ssn' in field_lower or 'last4' in field_lower:
                return cls.normalize_ssn_last4(str(value))
            return cls.normalize_id(str(value))
        
        # Date fields
        if any(x in field_lower for x in ['date', 'period', 'year']):
            if 'year' in field_lower:
                return cls.extract_year(str(value))
            return cls.normalize_date(str(value))
        
        # Amount fields
        if any(x in field_lower for x in ['amount', 'cost', 'pay', 'salary', 'expense', 'total', 
                                           'debit', 'credit', 'balance', 'price']):
            normalized_amount = cls.normalize_amount(value)
            if normalized_amount is not None:
                return str(round(normalized_amount, 2))
            return None
        
        # Phone fields
        if 'phone' in field_lower or 'fax' in field_lower:
            return cls.normalize_phone(str(value))
        
        # Department fields
        if 'department' in field_lower or 'dept' in field_lower:
            return cls.normalize_department(str(value))
        
        # Doc type fields
        if 'doc_type' in field_lower or 'document_type' in field_lower or 'type' in field_lower:
            return cls.normalize_doc_type(str(value))
        
        # Default: lowercase and trim
        if isinstance(value, str):
            return ' '.join(value.lower().split()).strip() or None
            
        return str(value).strip() or None


def build_vector_restrictions(
    entities: Dict[str, Any], 
    doc_type: str,
    doc_id: str,
    restriction_fields: List[str]
) -> List[Dict[str, Any]]:
    """
    Build vector search restrictions from extracted entities.
    
    IMPORTANT: Each namespace can only appear ONCE in restrictions.
    Multiple fields mapping to the same namespace are deduplicated.
    
    Args:
        entities: Dictionary of extracted entity values
        doc_type: Document type (for doc_type restriction)
        doc_id: Parent document ID (for RBAC restriction)
        restriction_fields: List of field names to use as restrictions
        
    Returns:
        List of restriction dictionaries for vector search
    """
    normalizer = EntityNormalizer()
    
    # Use dict to collect values per namespace (deduplication)
    namespace_values: Dict[str, set] = {}
    
    # Always add doc_id for RBAC
    if doc_id:
        namespace_values["doc_id"] = {doc_id}
    
    # Always add doc_type
    if doc_type:
        normalized_type = normalizer.normalize_doc_type(doc_type)
        if normalized_type:
            namespace_values["doc_type"] = {normalized_type}
    
    # Add entity-based restrictions (deduplicated by namespace)
    for field in restriction_fields:
        raw_value = entities.get(field)
        if raw_value:
            normalized_value = normalizer.normalize_entity_value(raw_value, field)
            if normalized_value:
                # Map field names to standardized namespaces
                namespace = map_field_to_namespace(field)
                
                # Add to existing namespace or create new
                if namespace not in namespace_values:
                    namespace_values[namespace] = set()
                namespace_values[namespace].add(normalized_value)
    
    # Convert to list format (each namespace appears exactly once)
    restrictions = []
    for namespace, values in namespace_values.items():
        restrictions.append({
            "namespace": namespace,
            "allow_list": list(values)
        })
    
    # Log for debugging
    logger.debug(f"Built {len(restrictions)} vector restrictions for doc_type={doc_type}")
    
    return restrictions


def map_field_to_namespace(field_name: str) -> str:
    """
    Map extraction field names to standardized vector namespace names.
    This keeps namespace count manageable while covering all document types.
    
    Args:
        field_name: Original field name from extraction
        
    Returns:
        Standardized namespace name
    """
    field_lower = field_name.lower()
    
    # Name fields → entity_name namespace
    if any(x in field_lower for x in ['employee_name', 'traveler_name', 'payee_name', 
                                       'payer_name', 'recipient_name', 'child_name',
                                       'caseworker_name', 'primary_entity_name',
                                       'author_name', 'vendor_name', 'company_name',
                                       'secondary_entity_name']):
        return "entity_name"
    
    # ID fields → entity_id namespace
    if any(x in field_lower for x in ['check_number', 'employee_id', 'case_number',
                                       'recipient_tin', 'primary_id', 'file_reference',
                                       'report_number', 'account_number', 'routing_number',
                                       'invoice_number', 'reference_number', 'secondary_id',
                                       'document_id', 'transaction_id', 'batch_number']):
        return "entity_id"
    
    # Amount fields → amount namespace
    if any(x in field_lower for x in ['amount', 'check_amount', 'invoice_amount', 
                                       'gross_pay', 'net_pay', 'total']):
        return "amount"
    
    # Date fields → date namespace
    if any(x in field_lower for x in ['date', 'period']):
        return "date"
    
    # Department → department namespace
    if 'department' in field_lower:
        return "department"
    
    # Fund/Account codes → fund_code namespace
    if any(x in field_lower for x in ['fund_code', 'fund_name', 'cost_center']):
        return "fund_code"
    
    # Location → location namespace
    if any(x in field_lower for x in ['city', 'destination', 'county', 'location', 'address']):
        return "location"
    
    # Bank/Financial → bank namespace
    if any(x in field_lower for x in ['bank_name', 'bank', 'financial_institution']):
        return "bank"
    
    # Document type → doc_type namespace  
    if any(x in field_lower for x in ['document_type', 'doc_type', 'detected_document_type', 'document_type_guess']):
        return "doc_type"
    
    # Status → status namespace
    if 'status' in field_lower:
        return "status"
    
    # Default: use the field name itself
    return field_name
