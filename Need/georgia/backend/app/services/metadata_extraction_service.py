from app.llm.gemini_api_key_client import gemini_client
import json
import logging
import re
from typing import Dict, Optional, Any
from app import config
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

class MetadataExtractionService:
    """
    Service to extract entity metadata (identifiers, dates, types) from document text
    using Gemini Flash. This metadata is used for Entity-Centric Vector Search.
    """

    _model = None

    @classmethod
    def _get_model(cls):
        """Returns the gemini_client if available."""
        if not gemini_client.is_available():
            logger.warning("Gemini API Key not configured for MetadataExtractionService.")
            return None
        return gemini_client

    @classmethod
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
    def extract_identifiers(cls, text_chunk: str, filename: str = "") -> Dict[str, Any]:
        """
        Analyzes text (usually the first chunk) to extract universal identifiers.
        
        Args:
            text_chunk: The text content to analyze.
            filename: The filename (useful context for the model).

        Returns:
            Dict containing extracted metadata keys (entity_name, entity_id, etc.)
        """
        model = cls._get_model()
        if not model:
            logger.warning("Metadata extraction skipped: Model not initialized.")
            return {}

        prompt = f"""
        Analyze this document text and filename. It could be ANY type (Personal, Organizational, Government, Medical, Vehicle, Historical).
        
        Extract the following identifiers if they are clearly present. If a field is not found, omit it.
        
        Required Output JSON Format:
        {{
            "entity_name": "Person Name OR Organization/Vendor/Business Name",
            "entity_id": "Unique ID string (e.g., NID, Driver License No, Tax ID, Passport No, Registration No, Invoice No)",
            "vehicle_tag": "Car Tag / Plate Number / VIN",
            "date": "Primary Document Date (YYYY-MM-DD format preferred, or YYYY)",
            "doc_type": "Short Type label (e.g., LICENSE, INVOICE, TAX_FORM, REGISTRATION, DEED, PRESCRIPTION)"
        }}

        Input Context:
        Filename: {filename}
        
        Document Text Start:
        ---
        {text_chunk[:10000]} 
        ---
        
        Return ONLY valid JSON.
        """

        try:
            response = model.generate_content(
                prompt, 
                temperature=0.1,
                response_mime_type="application/json"
            )
            
            if response.text:
                cleaned_text = response.text.strip()
                # Remove code blocks if present (though mime_type should prevent this)
                if cleaned_text.startswith("```json"):
                    cleaned_text = cleaned_text[7:-3]
                elif cleaned_text.startswith("```"):
                     cleaned_text = cleaned_text[3:-3]
                
                metadata = json.loads(cleaned_text)
                
                # Basic validation/cleanup of values
                cleaned_metadata = {}
                for k, v in metadata.items():
                    if v and isinstance(v, str) and v.lower() not in ["none", "n/a", "unknown", "null"]:
                        if k in ["doc_type", "entity_id", "vehicle_tag"]:
                            cleaned_metadata[k] = v.upper()
                        else:
                            cleaned_metadata[k] = v
                
                logger.info(f"Extracted Metadata for {filename}: {cleaned_metadata}")
                return cleaned_metadata
            
        except Exception as e:
            logger.error(f"Error extracting metadata for {filename}: {e}")
            # We fail gracefully (return empty dict) so ingestion doesn't break
            return {}
            
        return {}
