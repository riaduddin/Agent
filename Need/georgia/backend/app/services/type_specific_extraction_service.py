# backend/app/services/type_specific_extraction_service.py
"""
Type-Specific Entity Extraction Service (v2.1 - Multimodal)

This service extracts entities from documents using document-type-specific
schemas. Supports BOTH text-based and multimodal (vision) extraction.

Key Features:
1. Per-chunk extraction (not just first chunk)
2. Type-specific schemas for maximum accuracy
3. MULTIMODAL extraction - sends PDF/image directly to Gemini Vision
4. Fallback to OCR text when vision is unavailable
5. Structured JSON output from Gemini
6. Validation and normalization of extracted values
7. Vector restriction generation for search

Why Multimodal is Better:
- OCR text from Document AI can be messy/unorganized
- Direct vision sees the actual document layout
- Better handling of tables, forms, handwriting
- No loss of context from OCR errors
- ~30% better accuracy for complex documents

Usage:
    from app.services.type_specific_extraction_service import TypeSpecificExtractionService

    # PREFERRED: Multimodal extraction using PDF bytes
    entities = TypeSpecificExtractionService.extract_entities_multimodal(
        pdf_bytes=chunk_pdf_bytes,
        doc_type="CHECKS",
        filename="check_12345.pdf",
        fallback_text=ocr_text  # Optional fallback
    )

    # ALTERNATIVE: Text-only extraction (legacy)
    entities = TypeSpecificExtractionService.extract_entities(
        text=ocr_text,
        doc_type="CHECKS",
        filename="check_12345.pdf"
    )

    # Get vector restrictions for search
    restrictions = TypeSpecificExtractionService.get_vector_restrictions(
        entities=entities,
        doc_type="CHECKS",
        doc_id="parent_abc123"
    )
"""

import logging
import json
import re
import io
import base64
from typing import Dict, Any, List, Optional, Tuple, Union
from google import genai

from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from google.api_core import exceptions as google_exceptions

# Try to import PIL for image conversion (optional for PDF→image)
try:
    from PIL import Image
    import pdf2image
    HAS_PDF2IMAGE = True
except ImportError:
    HAS_PDF2IMAGE = False

from app import config
from app.services.extraction_schemas import (
    EXTRACTION_SCHEMAS,
    get_schema_for_doc_type,
    get_vector_restriction_fields,
    CATEGORY_TO_SCHEMA_MAP
)
from app.services.entity_normalizer import (
    EntityNormalizer,
    build_vector_restrictions,
    map_field_to_namespace
)
from app.utils.debug_logger import debug_log, set_debug_patterns
from app.llm.gemini_api_key_client import gemini_client

logger = logging.getLogger(__name__)


class TypeSpecificExtractionService:
    """
    Service for extracting entities from documents using type-specific schemas.
    """

    # _model = None # Removed as per instruction

    # Maximum text length to send to Gemini (characters)
    MAX_TEXT_LENGTH = 15000

    # Minimum text length required for extraction
    MIN_TEXT_LENGTH = 30

    # Minimum OCR confidence to attempt extraction
    MIN_OCR_CONFIDENCE = 0.2

    # Removed _get_model as per instruction

    @classmethod
    def _build_extraction_prompt(cls, text: str, doc_type: str, filename: str, schema: dict) -> str:
        """
        Build a Gemini prompt for entity extraction based on document type schema.
        Now uses UNIVERSAL EXTRACTION for all documents.
        """
        # Truncate text if too long
        if len(text) > cls.MAX_TEXT_LENGTH:
            text = text[:cls.MAX_TEXT_LENGTH] + "\n... [text truncated]"

        prompt = f"""You are an expert document analyzer. Analyze the document text below and extract structure information.

DOCUMENT TYPE HINT: {doc_type}
FILENAME: {filename}

TASK: Extract ALL searchable identifiers, names, dates, and amounts.

{cls._get_universal_field_list()}

**STRICT EXTRACTION RULE - NO BARRIERS:**
1. Extract EVERY identifiable number, name, date, and financial amount present in the text.
2. DO NOT limit yourself to any list. Grab ANY information that looks like a searchable entity.
3. Create descriptive new key-value pairs for important information not listed.
4. Your goal is 100% metadata capture—NOT classification.

DOCUMENT TEXT:
---
{text}
---

Return ONLY valid JSON with the extracted fields. Do not include any explanation or commentary."""

        return prompt

    @classmethod
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    def extract_entities(
        cls,
        text: str,
        doc_type: str,
        filename: str = "",
        ocr_confidence: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Extract entities from document text using type-specific schema.

        Args:
            text: OCR text content to analyze
            doc_type: Document type (e.g., "CHECKS", "PERSONNEL_FILES")
            filename: Original filename for context
            ocr_confidence: OCR confidence score (optional, for quality filtering)

        Returns:
            Dictionary of extracted entities with field names as keys
        """
        # Validate inputs
        if not text or len(text.strip()) < cls.MIN_TEXT_LENGTH:
            logger.warning(f"TypeSpecificExtractionService: Text too short for extraction ({len(text) if text else 0} chars)")
            return {}

        # Check OCR confidence if provided
        if ocr_confidence is not None and ocr_confidence < cls.MIN_OCR_CONFIDENCE:
            logger.warning(f"TypeSpecificExtractionService: OCR confidence too low ({ocr_confidence:.2f}), skipping extraction")
            return {}

        # Check client availability
        if not gemini_client.is_available():
            logger.error("TypeSpecificExtractionService: Gemini client not initialized")
            return {}

        # Get schema for document type
        schema = get_schema_for_doc_type(doc_type)
        if not schema:
            logger.warning(f"TypeSpecificExtractionService: No schema for doc_type '{doc_type}', using OTHER")
            schema = EXTRACTION_SCHEMAS["OTHER"]

        # Build prompt
        prompt = cls._build_extraction_prompt(text, doc_type, filename, schema)

        try:
            # Configure generation for JSON output
            config_gen = types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,  # Low temperature for extraction precision
                top_p=0.95,
                max_output_tokens=4096
            )

            # Generate response
            response = gemini_client.client.models.generate_content(
                model=config.GEMINI_OCR_MODEL_NAME,
                contents=[prompt],
                config=config_gen
            )

            if not response.text:
                logger.warning(f"TypeSpecificExtractionService: Empty response from Gemini for doc_type={doc_type}")
                return {}

            # Parse JSON response
            response_text = response.text.strip()

            # Clean up response if needed (remove code blocks)
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()

            # Parse JSON
            try:
                entities = json.loads(response_text)
            except json.JSONDecodeError as e:
                logger.error(f"TypeSpecificExtractionService: Failed to parse JSON response: {e}")
                logger.debug(f"Raw response: {response_text[:500]}")
                return {}

            # Validate and clean entities
            cleaned_entities = cls._clean_extracted_entities(entities, doc_type)

            # Add metadata
            cleaned_entities["_doc_type"] = doc_type
            cleaned_entities["_extraction_version"] = "2.0"

            logger.info(f"TypeSpecificExtractionService: Extracted {len(cleaned_entities)} fields for doc_type={doc_type}")

            return cleaned_entities

        except Exception as e:
            logger.error(f"TypeSpecificExtractionService: Extraction failed for doc_type={doc_type}: {e}", exc_info=True)
            return {}

    @classmethod
    def _get_universal_field_list(cls) -> str:
        """Centralized universal field list for consistent extraction across all prompt builders."""
        return """
UNIVERSAL ENTITY FIELDS (Extract ALL that apply):
1. IDENTIFIERS (CRITICAL): 
   - check_number (Check #, No., Reference # for payments)
   - invoice_number (Invoice #, Bill No.)
   - case_number (Case ID, Reference ID, File #)
   - employee_id (Badge #, Employee No.)
   - vendor_id (Vendor Number, Supplier ID, Vendor #)
   - purchase_order_number (PO#, Purchase Order #)
   - recipient_tin/payer_tin (SSN, EIN, Tax ID)
   - vendor_tax_id (Vendor EIN or Tax ID)
   - account_number (Partial if possible)
   - routing_number
   - control_number
   - control_id
   - reference_number

2. NAMES & VENDOR INFO:
   - employee_name
   - vendor_name (Company name, Supplier name)
   - vendor_address (Complete address of the vendor)
   - vendor_phone / vendor_email
   - child_name
   - parent_names
   - traveler_name
   - recipient_name
   - payer_name
   - payee_name
   - supervisor_name
   - caseworker_name

3. DATES:
   - primary_date (Main date on document)
   - start_date / end_date
   - pay_date / report_date
   - incident_date / court_date

4. FINANCIALS:
   - check_amount
   - tax_amount
   - total_amount (Reporting amount / Total due)
   - gross_pay / net_pay
   - amount_reported
   - specific_amount (Any other significant amount found)

5. ORGANIZATIONAL:
   - department
   - agency / firm_name
   - bank_name
   - county / district
   - fund_code / cost_center

6. OTHER SEARCHABLE INFORMATION:
   - document_title
   - location / address
   - phone_number / email
   - document_type_guess
   - summary (Brief description of content)
   - any_other_relevant_info (Any other searchable identifier or context not listed above)"""

    @classmethod
    def _build_vision_prompt(cls, doc_type: str, filename: str, schema: dict) -> str:
        """
        Build a Gemini prompt for VISION-based universal extraction.
        """
        prompt = f"""You are an expert document analyzer with vision capabilities.
Analyze the document image and extract structure information.

DOCUMENT TYPE HINT: {doc_type}
FILENAME: {filename}

TASK: Analyze the document image and:
1. Provide a COMPLETE transcription of all text.
2. Extract ALL relevant structured information without any category limitations.

{cls._get_universal_field_list()}

**STRICT EXTRACTION RULE - NO BARRIERS:**
1. Extract EVERY identifiable number, name, date, and financial amount present in the chunk.
2. DO NOT limit yourself to any list. Reach out and grab ANY information that looks like a searchable entity.
3. If you see a field that seems important for searching but is not in the list, extract it as a new key-value pair.
4. For the _full_transcription field, provide a COMPLETE transcription of all text using Markdown for structure.
5. Your goal is 100% data capture of all metadata—NOT classification.

Return ONLY valid JSON with the extracted fields. Do not include any explanation or commentary."""

        return prompt

    @classmethod
    def _build_category_detection_prompt(cls) -> str:
        """
        Build a prompt to detect the document category from our static list.

        Returns:
            Prompt string for category detection
        """
        # Static category list - these are the ONLY valid categories
        categories = [
            "CHECKS - Bank checks, cashier's checks, direct deposits, payment instruments",
            "1099 - Tax forms (1099-MISC, 1099-NEC, 1099-INT, etc.)",
            "PAYROLL_REPORTS_N_DOCUMENTS - Pay stubs, payroll registers, W-2s, earnings statements",
            "PERSONNEL_FILES - Employee records, HR documents, employment forms, performance reviews",
            "TRAVEL_REPORTS - Travel expense reports, reimbursement requests, travel authorizations",
            "LEAVE_DOCUMENTS - Leave requests, sick leave, vacation, FMLA forms",
            "CHILD_WELFARE_REPORTS - Child welfare cases, foster care, CPS documents",
            "MONTH_END_REPORTS - General ledger, journal entries, financial statements, GL reports",
            "PENDING_FILES - Documents awaiting processing or classification",
            "OTHER - Any document that doesn't fit the above categories"
        ]

        category_list = "\n".join([f"- {cat}" for cat in categories])

        prompt = f"""You are a document classification expert. Analyze this document image and determine which category it belongs to.

AVAILABLE CATEGORIES (choose EXACTLY ONE):
{category_list}

CLASSIFICATION RULES:
1. Examine the document carefully - look at headers, logos, form titles, and content
2. Choose the MOST SPECIFIC category that fits
3. If the document shows a check or payment instrument → CHECKS
4. If it's a tax form (1099, W-2) → 1099 or PAYROLL_REPORTS_N_DOCUMENTS
5. If it's an employee/HR document → PERSONNEL_FILES
6. If it's payroll-related (pay stub, earnings) → PAYROLL_REPORTS_N_DOCUMENTS
7. If unclear or doesn't match any specific category → OTHER

Return your response as JSON with exactly this format:
{{
    "detected_category": "CATEGORY_NAME",
    "confidence": "high/medium/low",
    "reason": "Brief explanation of why this category was chosen"
}}

Return ONLY valid JSON. No other text."""

        return prompt

    @classmethod
    def _build_combined_detection_and_extraction_prompt(cls, filename: str) -> str:
        """
        Build a SINGLE prompt that detects category AND extracts entities.
        This reduces API calls from 2 to 1 per chunk.

        Enhanced in v2.4 with:
        - Strict 11-category enforcement
        - Structured Markdown transcription with table formatting
        - Clear examples for expected output

        Args:
            filename: Original filename for context

        Returns:
            Combined prompt string
        """
        prompt = f"""You are an expert document analyzer. Analyze this document image/PDF and perform THREE tasks:

{cls._get_universal_field_list()}

**IMPORTANT: MULTI-DOCUMENT DETECTION**
This chunk may contain MULTIPLE separate documents (e.g., 2 checks on different pages, multiple invoices).
If you detect multiple distinct documents, you MUST extract EACH ONE separately and return a JSON array.

═══════════════════════════════════════════════════════════════════════════════
TASK 1: CATEGORY DETECTION (MANDATORY - STRICT)
═══════════════════════════════════════════════════════════════════════════════
You MUST classify this document into EXACTLY ONE of these 11 categories.
No other values are allowed. Choose the BEST match:

  1. 1099 - Tax forms (1099-MISC, 1099-NEC, 1099-INT, W-2, etc.)
  2. CHECKS - Bank checks, cashier's checks, direct deposits, payment vouchers
  3. CHILD_WELFARE_REPORTS - Child welfare cases, CPS documents, foster care records
  4. LEAVE_DOCUMENTS - Leave requests, sick leave, vacation, FMLA forms
  5. MONTH_END_REPORTS - General ledger, journal entries, financial statements, stock reports, inventory reports
  6. OTHER - Documents that don't fit specific categories but are classifiable
  7. PAYROLL_REPORTS_N_DOCUMENTS - Pay stubs, payroll registers, earnings statements
  8. PENDING_FILES - Documents awaiting processing or incomplete documents
  9. PERSONNEL_FILES - Employee records, HR documents, employment forms
  10. TRAVEL_REPORTS - Travel expense reports, reimbursements, travel authorizations
  11. UNCATEGORIZED - Cannot determine document type or unreadable content

═══════════════════════════════════════════════════════════════════════════════
TASK 2: FULL TEXT TRANSCRIPTION (CRITICAL - PRESERVE ALL CONTENT & STRUCTURE)
═══════════════════════════════════════════════════════════════════════════════
Provide a COMPLETE transcription that PRESERVES the document's visual layout.
NO TEXT OR DATA MAY BE OMITTED. Every word, number, header, footer must be captured.

**MANDATORY FORMATTING RULES:**

1. **TABLES/GRIDS** - ANY tabular data MUST use Markdown table format:
   | Column1 | Column2 | Column3 | Column4 |
   |---------|---------|---------|----------|
   | value1  | value2  | value3  | value4  |

2. **HEADERS/TITLES** - Use Markdown headings:
   # Main Document Title
   ## Section Header
   ### Sub-section

3. **LISTS** - Use bullet points or numbered lists:
   - Bullet item 1
   - Bullet item 2
   1. Numbered item 1
   2. Numbered item 2

4. **KEY-VALUE PAIRS** - Format as:
   **Label:** Value

5. **HANDWRITING** - Capture in brackets: [Handwritten: John Smith]

6. **STAMPS/SIGNATURES** - Note in brackets: [Stamp: PAID] [Signature: illegible]

7. **PARAGRAPHS** - Preserve paragraph breaks with blank lines

8. **ALL TEXT** - Include EVERY word, number, header, footer, fine print, watermarks

TASK 3: UNIVERSAL ENTITY EXTRACTION
═══════════════════════════════════════════════════════════════════════════════
{cls._get_universal_field_list()}

**STRICT EXTRACTION RULE - NO BARRIERS:**
1. Extract EVERY identifiable number, name, date, and financial amount present in the chunk.
2. DO NOT limit yourself to the categories or lists above. Reach out and grab ANY information that looks like a searchable entity or a primary identifier.
3. If you see a field that seems important for searching but is not in the list, extract it as a new key-value pair with a descriptive key names.
4. Your goal is 100% data capture of all metadata—NOT classification.

FILENAME: {filename}

═══════════════════════════════════════════════════════════════════════════════
OUTPUT FORMAT (JSON)
═══════════════════════════════════════════════════════════════════════════════

**CRITICAL: MULTIPLE DOCUMENTS IN ONE IMAGE/PDF**
If the image/PDF contains MULTIPLE separate documents (e.g., 2 checks, 2 invoices),
you MUST return a JSON ARRAY with one object per document:

[
  {{
    "_detected_category": "CHECKS",
    "_category_confidence": "high",
    "_full_transcription": "# Check 1\\n...",
    "check_number": "123456",
    "payee_name": "John Doe",
    ...
  }},
  {{
    "_detected_category": "CHECKS", 
    "_category_confidence": "high",
    "_full_transcription": "# Check 2\\n...",
    "check_number": "789012",
    "payee_name": "Jane Smith",
    ...
  }}
]

**SINGLE DOCUMENT** - Return a single JSON object:

{{
    "_detected_category": "EXACT_CATEGORY_FROM_11_OPTIONS_ABOVE",
    "_category_confidence": "high|medium|low",
    "_full_transcription": "# Document Title\n\n**Date:** 2024-01-15\n**Reference:** ABC123\n\n## Data Section\n\n| Col1 | Col2 | Col3 |\n|------|------|------|\n| val1 | val2 | val3 |\n\nAdditional paragraph text...\n\n[Signature: John Doe]",
    "field1": "extracted_value1",
    "field2": "extracted_value2"
}}

**EXAMPLE** - For a stock/inventory report:
{{
    "_detected_category": "MONTH_END_REPORTS",
    "_category_confidence": "high",
    "_full_transcription": "# Stock Report for 2016-08\n\n**Report Type:** Monthly Inventory\n**Generated:** 2016-08-31\n\n## Inventory Summary\n\n| Category | Product | Units Sold | Units in Stock | Unit Price |\n|----------|---------|------------|----------------|------------|\n| Beverages | Chai | 63 | 39 | 18.00 |\n| Beverages | Guaraná Fantástica | 40 | 20 | 4.50 |\n| Condiments | Aniseed Syrup | 25 | 13 | 10.00 |\n\n---\n\n**Total Products:** 77\n**Report Generated By:** System",
    "report_type": "Stock Report",
    "report_date": "2016-08",
    "department": "Inventory"
}}

**EXAMPLE** - Multiple checks on same page:
[
  {{
    "_detected_category": "CHECKS",
    "_category_confidence": "high", 
    "_full_transcription": "# OFFICIAL CHECK\\n\\n**Check Number:** 685241309\\n**Bank:** BANK ONE\\n**Date:** 11/20/2003\\n**Pay To:** GET SMART TECH, INC.\\n**Amount:** $11,000.00\\n**Remitter:** CONNEL COMMUNICATION",
    "check_number": "685241309",
    "payee_name": "GET SMART TECH, INC.",
    "payer_name": "CONNEL COMMUNICATION",
    "check_amount": "11000.00",
    "check_date": "2003-11-20",
    "bank_name": "BANK ONE"
  }},
  {{
    "_detected_category": "CHECKS",
    "_category_confidence": "high",
    "_full_transcription": "# CASHIER'S CHECK\\n\\n**Check Number:** 203064\\n**Bank:** Peoples Bank\\n**Date:** 05/20/2004\\n**Pay To:** STATE OF GEORGIA\\n**Amount:** $5,500.00",
    "check_number": "203064",
    "payee_name": "STATE OF GEORGIA",
    "check_amount": "5500.00",
    "check_date": "2004-05-20",
    "bank_name": "Peoples Bank OF NORTHERN KENTUCKY"
  }}
]

═══════════════════════════════════════════════════════════════════════════════
EXTRACTION RULES
═══════════════════════════════════════════════════════════════════════════════
1. _detected_category MUST be EXACTLY one of the 11 category names listed above
2. _full_transcription MUST be complete Markdown-formatted text with NO missing content:
   - Use Markdown tables for ANY tabular/grid data
   - Use headings for titles and sections
   - Use **bold** for labels in key-value pairs
   - Preserve ALL text - headers, footers, stamps, fine print, everything
3. Dates → YYYY-MM-DD format (e.g., 2024-01-15)
4. Amounts → numeric only, no $ or commas (e.g., 1234.56)
5. Names → full names exactly as written
6. Omit entity fields that are not visible in the document (but transcription must be complete)

Return ONLY valid JSON. No explanation or commentary."""

        return prompt

    @classmethod
    def _convert_pdf_to_images(cls, pdf_bytes: bytes, dpi: int = 150) -> List[bytes]:
        """
        Convert PDF bytes to a list of PNG image bytes.

        Args:
            pdf_bytes: Raw PDF file bytes
            dpi: Resolution for conversion (lower = faster, higher = better quality)

        Returns:
            List of PNG image bytes (one per page)
        """
        if not HAS_PDF2IMAGE:
            logger.warning("pdf2image not available, cannot convert PDF to images")
            return []

        try:
            # Convert PDF to PIL images
            images = pdf2image.convert_from_bytes(pdf_bytes, dpi=dpi)

            # Convert each image to bytes
            image_bytes_list = []
            for img in images:
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='PNG')
                img_byte_arr.seek(0)
                image_bytes_list.append(img_byte_arr.getvalue())

            return image_bytes_list
        except Exception as e:
            logger.error(f"Failed to convert PDF to images: {e}")
            return []

    @classmethod
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    def extract_entities_multimodal(
        cls,
        pdf_bytes: Optional[bytes] = None,
        image_bytes: Optional[bytes] = None,
        doc_type: str = "OTHER",
        filename: str = "",
        fallback_text: Optional[str] = None,
        ocr_confidence: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Extract entities using MULTIMODAL (vision) approach with automatic category detection.

        Uses a SINGLE API call to both detect category AND extract entities,
        reducing API calls from 2 to 1 per chunk for better throughput.

        Args:
            pdf_bytes: Raw PDF file bytes (will convert to image)
            image_bytes: Raw image bytes (PNG/JPEG) - use if already have image
            doc_type: Document type hint (may be overridden by detection)
            filename: Original filename for context
            fallback_text: OCR text to use if vision fails
            ocr_confidence: OCR confidence (only used for fallback)

        Returns:
            Dictionary of extracted entities including _detected_category
        """
        # === DEBUG: Track v2.4 extraction calls ===
        debug_log("[EXTRACTION] ========== extract_entities_multimodal CALLED ==========")
        debug_log("[EXTRACTION] Input params:",
                  pdf_bytes_size=len(pdf_bytes) if pdf_bytes else 0,
                  image_bytes_size=len(image_bytes) if image_bytes else 0,
                  doc_type=doc_type,
                  filename=filename,
                  has_fallback_text=bool(fallback_text))
        
        # Check client
        if not gemini_client.is_available():
            logger.error("TypeSpecificExtractionService: Gemini client not initialized")
            if fallback_text:
                return cls.extract_entities(fallback_text, doc_type, filename, ocr_confidence)
            return {}
        
        debug_log("[EXTRACTION] Gemini model initialized successfully")

        # Prepare multimodal parts
        content_parts = []

        if pdf_bytes:
            # Use direct PDF upload - Gemini 1.5+ natively supports PDFs
            debug_log("[EXTRACTION] Using direct PDF upload (Gemini native PDF support)")
            content_parts.append(
                types.Part.from_bytes(
                    data=pdf_bytes,
                    mime_type="application/pdf"
                )
            )
            logger.info(f"Sending PDF directly to Gemini ({len(pdf_bytes)} bytes)")
        elif image_bytes:
            # Handle direct image input
            mime_type = "image/png"
            if image_bytes[:2] == b'\xff\xd8':
                mime_type = "image/jpeg"

            content_parts.append(
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=mime_type
                )
            )
            debug_log("[EXTRACTION] Added image input to request", mime_type=mime_type)

        if not content_parts:
            logger.warning("No visual content available for multimodal extraction")
            debug_log("[EXTRACTION] No visual content available")
            if fallback_text:
                logger.info("Falling back to text-based extraction")
                debug_log("[EXTRACTION] Using text-based fallback extraction")
                return cls.extract_entities(fallback_text, doc_type, filename, ocr_confidence)
            return {}

        # Check if we need to auto-detect category
        is_generic_type = doc_type.upper() in {"OTHER", "GENERAL_DOCUMENT", "GENERAL", "UNKNOWN", "UNCLASSIFIED", "UNCATEGORIZED", "MISC", "DEFAULT", ""}
        debug_log("[EXTRACTION] Category detection check:", is_generic_type=is_generic_type, doc_type_upper=doc_type.upper())

        try:
            if is_generic_type:
                # ============================================================
                # COMBINED DETECTION + EXTRACTION (Single API Call)
                # ============================================================
                # For generic types, use combined prompt that detects category AND extracts in one call
                logger.info(f"Generic doc_type '{doc_type}' detected, using combined detection+extraction")
                debug_log("[EXTRACTION] Using COMBINED detection+extraction prompt (v2.4)")
                prompt_text = cls._build_combined_detection_and_extraction_prompt(filename)
            else:
                # ============================================================
                # SCHEMA-SPECIFIC EXTRACTION (Single API Call)
                # ============================================================
                # For known types, use the specific schema
                logger.info(f"Specific doc_type '{doc_type}' provided, using schema-specific extraction")
                debug_log("[EXTRACTION] Using SCHEMA-SPECIFIC extraction prompt for:", doc_type=doc_type)
                schema = get_schema_for_doc_type(doc_type)
                if not schema:
                    schema = EXTRACTION_SCHEMAS["OTHER"]
                prompt_text = cls._build_vision_prompt(doc_type, filename, schema)

            # Prepend prompt to content parts
            # Note: prompt should be a Part too if mixing
            request_parts = [types.Part.from_text(text=prompt_text)] + content_parts

            # Configure generation for JSON output
            config_gen = types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,  # Low temperature for extraction precision
                top_p=0.95,
                max_output_tokens=8192, # Increased for transcription
                safety_settings=[
                    # Map to types.SafetySetting
                    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
                    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
                    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
                    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
                ]
            )
            
            debug_log("[EXTRACTION] Calling Gemini API with multimodal input...", 
                      model_name=config.GEMINI_OCR_MODEL_NAME,
                      content_parts_count=len(content_parts))

            # Generate response with multimodal input - SINGLE API CALL
            response = gemini_client.client.models.generate_content(
                model=config.GEMINI_OCR_MODEL_NAME,
                contents=request_parts,
                config=config_gen
            )

            # Check if response was blocked or empty
            # google-genai response object logic might defer
            if not response.candidates:
                logger.warning(f"Gemini returned no candidates for doc_type={doc_type}")
                debug_log("[EXTRACTION] No candidates in Gemini response - likely blocked")
                if fallback_text:
                    return cls.extract_entities(fallback_text, doc_type, filename, ocr_confidence)
                return {}
            
            candidate = response.candidates[0]
            finish_reason = getattr(candidate, 'finish_reason', None)
            
            # Check finish_reason
            # In google-genai, FinishReason.STOP is likely "STOP" (string) or enum.
            # Printing finish_reason debug might help, but let's assume standard behavior.
            # If it is not 'STOP', it might be problematic.
            # Let's map robustly.
            # Note: google-genai types might have it as string or enum. 
            # We will convert to string to be safe.
            finish_reason_str = str(finish_reason)
            
            if finish_reason and "STOP" not in finish_reason_str and finish_reason != 1:
                logger.warning(f"Gemini response blocked/incomplete: finish_reason={finish_reason_str}")
                debug_log("[EXTRACTION] Gemini response blocked:", finish_reason=finish_reason_str)
                
                if fallback_text:
                    logger.info("Falling back to text-based extraction due to blocked response")
                    return cls.extract_entities(fallback_text, doc_type, filename, ocr_confidence)
                return {}
            
            # Now safely access text
            try:
                response_text = response.text
            except Exception as e:
                logger.warning(f"Cannot access response.text: {e}")
                debug_log("[EXTRACTION] Cannot access response.text:", error=str(e))
                if fallback_text:
                    return cls.extract_entities(fallback_text, doc_type, filename, ocr_confidence)
                return {}
            
            if not response_text:
                logger.warning(f"Empty response from Gemini Vision for doc_type={doc_type}")
                if fallback_text:
                    logger.info("Falling back to text-based extraction")
                    debug_log("[EXTRACTION] Empty Gemini response, using fallback")
                    return cls.extract_entities(fallback_text, doc_type, filename, ocr_confidence)
                return {}
            
            debug_log("[EXTRACTION] Gemini API response received", response_length=len(response_text))

            # Parse JSON response
            response_text = response_text.strip()

            # Clean up response if needed
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()

            # Parse JSON
            try:
                parsed_response = json.loads(response_text)
                debug_log("[EXTRACTION] JSON parsed successfully", response_type=type(parsed_response).__name__)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON from vision response: {e}")
                logger.debug(f"Raw response: {response_text[:500]}")
                debug_log("[EXTRACTION] JSON parse FAILED", error=str(e))
                if fallback_text:
                    logger.info("Falling back to text-based extraction")
                    return cls.extract_entities(fallback_text, doc_type, filename, ocr_confidence)
                return {}

            # Handle case where Gemini returns a list instead of a dict
            # This happens when there are MULTIPLE documents (e.g., multiple checks) in one chunk
            # We should preserve ALL documents, not merge them
            documents_list = []  # Store individual documents if multiple found
            
            if isinstance(parsed_response, list):
                debug_log("[EXTRACTION] Response is a LIST (multiple documents in chunk)", item_count=len(parsed_response))
                
                if len(parsed_response) == 0:
                    entities = {}
                elif len(parsed_response) == 1:
                    entities = parsed_response[0] if isinstance(parsed_response[0], dict) else {}
                else:
                    # MULTIPLE DOCUMENTS IN CHUNK - preserve all of them
                    # Extract common metadata from first item, store all documents in array
                    logger.info(f"Found {len(parsed_response)} documents in chunk (e.g., multiple checks)")
                    
                    # Get category and transcription from first document
                    first_item = parsed_response[0] if isinstance(parsed_response[0], dict) else {}
                    entities = {
                        "_detected_category": first_item.get("_detected_category"),
                        "_category_confidence": first_item.get("_category_confidence"),
                    }
                    
                    # Combine all transcriptions
                    all_transcriptions = []
                    for i, item in enumerate(parsed_response):
                        if isinstance(item, dict):
                            trans = item.get("_full_transcription", "")
                            if trans:
                                all_transcriptions.append(f"--- Document {i+1} ---\n{trans}")
                            
                            # Store each document's data
                            doc_data = {}
                            for key, value in item.items():
                                if not key.startswith("_") and value is not None:
                                    doc_data[key] = value
                            if doc_data:
                                documents_list.append(doc_data)
                    
                    if all_transcriptions:
                        entities["_full_transcription"] = "\n\n".join(all_transcriptions)
                    
                    debug_log("[EXTRACTION] Preserved multiple documents", 
                              document_count=len(documents_list),
                              transcription_sections=len(all_transcriptions))
            elif isinstance(parsed_response, dict):
                entities = parsed_response
            else:
                logger.error(f"Unexpected response type: {type(parsed_response)}")
                debug_log("[EXTRACTION] Unexpected response type", response_type=type(parsed_response).__name__)
                if fallback_text:
                    return cls.extract_entities(fallback_text, doc_type, filename, ocr_confidence)
                return {}

            debug_log("[EXTRACTION] Entities dict ready", entity_count=len(entities))

            # Extract detected category from response (for combined detection)
            detected_category = entities.pop("_detected_category", None) if isinstance(entities, dict) else None
            category_confidence = entities.pop("_category_confidence", None) if isinstance(entities, dict) else None
            
            debug_log("[EXTRACTION] Raw category from Gemini:", 
                      detected_category=detected_category,
                      category_confidence=category_confidence)

            if detected_category:
                # Validate detected category - strict 11 categories only
                valid_categories = {
                    "1099", "CHECKS", "CHILD_WELFARE_REPORTS", "LEAVE_DOCUMENTS",
                    "MONTH_END_REPORTS", "OTHER", "PAYROLL_REPORTS_N_DOCUMENTS",
                    "PENDING_FILES", "PERSONNEL_FILES", "TRAVEL_REPORTS", "UNCATEGORIZED"
                }
                detected_category = detected_category.upper().strip()
                if detected_category not in valid_categories:
                    logger.warning(f"Invalid detected category '{detected_category}', using UNCATEGORIZED")
                    debug_log("[EXTRACTION] INVALID category detected, defaulting to UNCATEGORIZED", invalid_category=detected_category)
                    detected_category = "UNCATEGORIZED"
                logger.info(f"Category detected from combined call: {detected_category} (confidence: {category_confidence})")
                debug_log("[EXTRACTION] Final category:", category=detected_category, confidence=category_confidence)
            else:
                detected_category = doc_type if not is_generic_type else "OTHER"
                debug_log("[EXTRACTION] No category in response, using default:", category=detected_category)

            # IMPORTANT: Extract transcription BEFORE cleaning (it starts with _ but we need to preserve it)
            full_transcription = entities.pop("_full_transcription", None) if isinstance(entities, dict) else None
            
            # Clean entities - for combined detection, be lenient with field validation
            if is_generic_type:
                # Don't filter fields for generic types - accept all extracted fields
                # This ensures that even if classified as UNCATEGORIZED, we keep visible entities
                cleaned_entities = {}
                for key, value in entities.items():
                    # Skip internal metadata keys except transcription (which is handled later)
                    if key.startswith("_") and key != "_full_transcription":
                        continue
                    if value is None:
                        continue
                    if isinstance(value, str) and not value.strip():
                        continue
                    if isinstance(value, str) and value.lower() in ["none", "n/a", "null", "unknown", "not found", "not available"]:
                        continue
                    if isinstance(value, str):
                        value = value.strip()
                        # Skip if just punctuation
                        if not re.search(r'[a-zA-Z0-9]', value):
                            continue
                    cleaned_entities[key] = value
            else:
                cleaned_entities = cls._clean_extracted_entities(entities, detected_category)
            
            # Add back the transcription if we captured it
            if full_transcription:
                cleaned_entities["_full_transcription"] = full_transcription
                debug_log("[EXTRACTION] Transcription preserved", length=len(full_transcription))
            
            # Add documents array if multiple documents were found in chunk
            if documents_list:
                cleaned_entities["_documents"] = documents_list
                cleaned_entities["_document_count"] = len(documents_list)
                debug_log("[EXTRACTION] Multiple documents stored", count=len(documents_list))
            
            debug_log("[EXTRACTION] Entities after cleaning:", entity_count=len(cleaned_entities))

            # Add metadata
            cleaned_entities["_doc_type"] = detected_category
            cleaned_entities["_original_doc_type"] = doc_type
            cleaned_entities["_detected_category"] = detected_category
            cleaned_entities["_extraction_version"] = "2.4_structured_transcription"
            cleaned_entities["_extraction_method"] = "multimodal"
            if category_confidence:
                cleaned_entities["_category_confidence"] = category_confidence

            entity_count = len([k for k in cleaned_entities.keys() if not k.startswith("_")])
            doc_count = cleaned_entities.get("_document_count", 1)
            logger.info(f"TypeSpecificExtractionService: Vision extraction - {entity_count} entities, {doc_count} document(s) for category={detected_category}")
            
            # === DEBUG: Final extraction result summary ===
            transcription_length = len(cleaned_entities.get("_full_transcription", ""))
            entity_keys = [k for k in cleaned_entities.keys() if not k.startswith("_")]
            debug_log("[EXTRACTION] ========== EXTRACTION COMPLETE ==========")
            debug_log("[EXTRACTION] Final result:",
                      detected_category=detected_category,
                      entity_count=entity_count,
                      document_count=doc_count,
                      transcription_length=transcription_length,
                      extraction_version="2.4")
            debug_log("[EXTRACTION] Extracted entity fields:", fields=entity_keys[:10])  # First 10 fields
            if doc_count > 1:
                debug_log("[EXTRACTION] Multiple documents in chunk:", documents=cleaned_entities.get("_documents", []))
            if transcription_length > 0:
                debug_log("[EXTRACTION] Transcription preview:", preview=cleaned_entities.get("_full_transcription", "")[:200])

            return cleaned_entities

        except google_exceptions.ResourceExhausted as e:
            logger.warning(f"Rate limited on vision extraction, will retry: {e}")
            debug_log("[EXTRACTION] Rate limited by Gemini API, will retry...")
            raise
        except google_exceptions.ServiceUnavailable as e:
            logger.warning(f"Service unavailable for vision extraction, will retry: {e}")
            debug_log("[EXTRACTION] Gemini API unavailable, will retry...")
            raise
        except Exception as e:
            logger.error(f"Vision extraction failed for doc_type={doc_type}: {e}", exc_info=True)
            debug_log("[EXTRACTION] EXCEPTION during extraction:", error=str(e))
            if fallback_text:
                logger.info("Falling back to text-based extraction after vision error")
                debug_log("[EXTRACTION] Using text-based fallback after error")
                return cls.extract_entities(fallback_text, doc_type, filename, ocr_confidence)
            return {}

    @classmethod
    def _build_transcription_prompt(cls, filename: str) -> str:
        """
        Build a Gemini prompt for full text transcription using vision.

        Args:
            filename: Original filename for context

        Returns:
            Prompt string for transcription
        """
        return f"""You are an expert document transcriber with vision capabilities.
Your task is to provide a COMPLETE and ACCURATE transcription of all text visible in the attached document image.

FILENAME: {filename}

TRANSCRIPTION RULES:
1. Transcribe EVERY word and number you see in the document.
2. Maintain the relative structure where possible (e.g., if it's a table, try to represent it logically).
3. Do not omit anything, even if it seems unimportant (stamps, small print, headers).
4. For checks, ensure you capture the payee, amount, date, bank name, and check number accurately.
5. For invoices, ensure you capture all line items, totals, dates, and vendor information.
6. If text is handwritten, do your best to transcribe it accurately.
7. Return ONLY the transcribed text. Do not include any meta-comments like "Here is the transcription" or "The document shows...".

Return the full text transcription now:"""

    @classmethod
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    def transcribe_content_multimodal(
        cls,
        pdf_bytes: Optional[bytes] = None,
        image_bytes: Optional[bytes] = None,
        filename: str = ""
    ) -> str:
        """
        Transcribe all content from a document using MULTIMODAL (vision) approach.
        This provides high-quality text for vector search.

        Args:
            pdf_bytes: Raw PDF file bytes
            image_bytes: Raw image bytes
            filename: Original filename for context

        Returns:
            Full transcribed text string
        """
        if not gemini_client.is_available():
            logger.error("TypeSpecificExtractionService: Gemini client not initialized for transcription")
            return ""

        # Get the image bytes - try to use direct PDF if available
        content_part = None
        if pdf_bytes:
            content_part = types.Part.from_bytes(
                data=pdf_bytes,
                mime_type="application/pdf"
            )
            logger.info(f"Sending PDF directly to Gemini for transcription ({len(pdf_bytes)} bytes)")
        elif image_bytes:
            mime_type = "image/png"
            if image_bytes[:2] == b'\xff\xd8':
                mime_type = "image/jpeg"
            
            content_part = types.Part.from_bytes(
                data=image_bytes,
                mime_type=mime_type
            )
            logger.info(f"Sending image to Gemini for transcription ({len(image_bytes)} bytes)")

        if not content_part:
            logger.warning("No visual content available for transcription")
            return ""

        prompt = cls._build_transcription_prompt(filename)

        try:
            # Low temperature for high fidelity transcription
            config_gen = types.GenerateContentConfig(
                temperature=0.0,
                top_p=0.95,
                max_output_tokens=8192
            )

            response = gemini_client.client.models.generate_content(
                model=config.GEMINI_MODEL_NAME,
                contents=[types.Part.from_text(text=prompt), content_part],
                config=config_gen
            )

            if not response.text:
                logger.warning("Empty transcription response from Gemini")
                return ""

            transcription = response.text.strip()
            logger.info(f"TypeSpecificExtractionService: Successfully transcribed {len(transcription)} characters via vision")
            return transcription

        except Exception as e:
            logger.error(f"Vision transcription failed: {e}", exc_info=True)
            return ""

    @classmethod
    def _clean_extracted_entities(cls, entities: Dict[str, Any], doc_type: str) -> Dict[str, Any]:
        """
        Clean and validate extracted entities.

        - Removes empty/null values
        - Validates field names against schema
        - Filters out obviously bad extractions

        Args:
            entities: Raw extracted entities dictionary
            doc_type: Document type for schema validation

        Returns:
            Cleaned entities dictionary
        """
        if not isinstance(entities, dict):
            return {}

        schema = get_schema_for_doc_type(doc_type)
        valid_fields = set(schema.get("fields", {}).keys())

        # BUILD UNIVERSAL ALLOW-LIST
        # We allow any field that is in the current schema OR in the universal set
        universal_fields = {
            "check_number", "invoice_number", "case_number", "employee_id", "vendor_id",
            "purchase_order_number", "recipient_tin", "payer_tin", "vendor_tax_id",
            "account_number", "routing_number", "control_number", "control_id", "reference_number",
            "employee_name", "vendor_name", "vendor_address", "vendor_phone", "vendor_email",
            "child_name", "parent_names", "traveler_name", "recipient_name", "payer_name",
            "payee_name", "supervisor_name", "casework_name", "primary_date", "start_date",
            "end_date", "pay_date", "report_date", "incident_date", "court_date",
            "check_amount", "tax_amount", "total_amount", "gross_pay", "net_pay",
            "amount_reported", "specific_amount", "department", "agency", "firm_name",
            "bank_name", "county", "district", "fund_code", "cost_center",
            "document_title", "location", "address", "phone_number", "email",
            "document_type_guess", "summary", "any_other_relevant_info"
        }

        cleaned = {}
        for key, value in entities.items():
            # Skip internal fields, except full_transcription
            if key.startswith("_") and key != "_full_transcription":
                continue

            # NEW: Allow field if it's in the specific schema OR the universal field list
            is_valid = (key in valid_fields) or (key in universal_fields) or is_generic_type
            
            if not is_valid:
                # If it's not on either list, we still keep it if it looks like a valid dynamic key (v2.4 capability)
                # but we log it for awareness
                logger.debug(f"Allowing dynamic field '{key}' not in schema/universal list for {doc_type}")
            
            if not is_valid and not config.ALLOW_DYNAMIC_ENTITIES: # Fallback if we want to be strict later
                 logger.debug(f"Skipping unexpected field '{key}' not in schema/universal list for {doc_type}")
                 continue

            # Skip empty values
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            if isinstance(value, str) and value.lower() in ["none", "n/a", "null", "unknown", "not found", "not available"]:
                continue

            # Clean string values
            if isinstance(value, str):
                value = value.strip()
                # Skip if just punctuation
                if not re.search(r'[a-zA-Z0-9]', value):
                    continue

            cleaned[key] = value

        return cleaned

    @classmethod
    def get_vector_restrictions(
        cls,
        entities: Dict[str, Any],
        doc_type: str,
        doc_id: str
    ) -> List[Dict[str, Any]]:
        """
        Generate vector search restrictions from extracted entities.

        Args:
            entities: Extracted entities dictionary
            doc_type: Document type
            doc_id: Parent document ID (for RBAC)

        Returns:
            List of restriction dictionaries for vector search
        """
        debug_log("[EXTRACTION] get_vector_restrictions called:",
                  doc_type=doc_type,
                  doc_id=doc_id,
                  entity_count=len(entities) if entities else 0)
        
        # Get restriction fields for this doc type
        restriction_fields = get_vector_restriction_fields(doc_type)
        debug_log("[EXTRACTION] Restriction fields for doc_type:", fields=restriction_fields)

        # Build restrictions using normalizer
        restrictions = build_vector_restrictions(
            entities=entities,
            doc_type=doc_type,
            doc_id=doc_id,
            restriction_fields=restriction_fields
        )
        
        debug_log("[EXTRACTION] Vector restrictions generated:", 
                  restriction_count=len(restrictions),
                  namespaces=[r.get('namespace') for r in restrictions] if restrictions else [])

        return restrictions

    @classmethod
    def extract_and_get_restrictions(
        cls,
        text: str,
        doc_type: str,
        doc_id: str,
        filename: str = "",
        ocr_confidence: Optional[float] = None
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Convenience method to extract entities AND generate restrictions in one call.

        Args:
            text: OCR text content
            doc_type: Document type
            doc_id: Parent document ID
            filename: Original filename
            ocr_confidence: OCR confidence score

        Returns:
            Tuple of (entities_dict, restrictions_list)
        """
        # Extract entities
        entities = cls.extract_entities(
            text=text,
            doc_type=doc_type,
            filename=filename,
            ocr_confidence=ocr_confidence
        )

        # Generate restrictions
        restrictions = cls.get_vector_restrictions(
            entities=entities,
            doc_type=doc_type,
            doc_id=doc_id
        )

        return entities, restrictions

    @classmethod
    def should_extract(cls, text: str, ocr_confidence: Optional[float] = None) -> bool:
        """
        Determine if extraction should be attempted for given text.

        Args:
            text: OCR text content
            ocr_confidence: OCR confidence score

        Returns:
            True if extraction should proceed, False otherwise
        """
        # Check text length
        if not text or len(text.strip()) < cls.MIN_TEXT_LENGTH:
            return False

        # Check OCR confidence
        if ocr_confidence is not None and ocr_confidence < cls.MIN_OCR_CONFIDENCE:
            return False

        return True

    @classmethod
    def batch_extract_entities(
        cls,
        chunks: List[Dict[str, Any]],
        default_doc_type: str = "OTHER"
    ) -> List[Dict[str, Any]]:
        """
        Extract entities from multiple chunks.

        Note: This processes sequentially. For true batching (multiple chunks
        in one API call), additional implementation would be needed.

        Args:
            chunks: List of chunk dictionaries with 'text', 'doc_type', 'filename'
            default_doc_type: Default document type if not specified

        Returns:
            List of entity dictionaries (same order as input)
        """
        results = []

        for chunk in chunks:
            text = chunk.get("text", "")
            doc_type = chunk.get("doc_type", default_doc_type)
            filename = chunk.get("filename", "")
            ocr_confidence = chunk.get("ocr_confidence")

            entities = cls.extract_entities(
                text=text,
                doc_type=doc_type,
                filename=filename,
                ocr_confidence=ocr_confidence
            )

            results.append(entities)

        return results


# Convenience functions for backward compatibility
def extract_chunk_entities(
    ocr_text: str,
    doc_type: str,
    filename: str = "",
    ocr_confidence: Optional[float] = None
) -> Dict[str, Any]:
    """
    Extract entities from a chunk's OCR text.

    This is the main function to call for per-chunk entity extraction.

    Args:
        ocr_text: The OCR text content of the chunk
        doc_type: The classified document type
        filename: Original parent filename for context
        ocr_confidence: OCR confidence score (optional)

    Returns:
        Dictionary of extracted entities
    """
    return TypeSpecificExtractionService.extract_entities(
        text=ocr_text,
        doc_type=doc_type,
        filename=filename,
        ocr_confidence=ocr_confidence
    )


def get_chunk_restrictions(
    entities: Dict[str, Any],
    doc_type: str,
    doc_id: str
) -> List[Dict[str, Any]]:
    """
    Generate vector restrictions from extracted entities.

    Args:
        entities: Extracted entities dictionary
        doc_type: Document type
        doc_id: Parent document ID

    Returns:
        List of vector restriction dictionaries
    """
    return TypeSpecificExtractionService.get_vector_restrictions(
        entities=entities,
        doc_type=doc_type,
        doc_id=doc_id
    )
