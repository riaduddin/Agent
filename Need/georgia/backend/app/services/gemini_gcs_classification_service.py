import logging
import json
from typing import Optional
from google.cloud import storage
from google import genai
from google.genai import types

from app import config
from app.utils.debug_logger import debug_log, debug_warn, debug_error
from app.llm.gemini_api_key_client import gemini_client

logger = logging.getLogger(__name__)

def download_blob_bytes(gcs_uri: str) -> Optional[bytes]:
    """Downloads a blob from GCS as bytes."""
    try:
        storage_client = storage.Client(project=config.PROJECT_ID)
        if gcs_uri.startswith("gs://"):
            gcs_uri = gcs_uri[5:]
        
        bucket_name, blob_name = gcs_uri.split("/", 1)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        return blob.download_as_bytes()
    except Exception as e:
        debug_error(f"Failed to download blob from GCS {gcs_uri}: {e}")
        return None

def classify_document_type_with_gemini_from_gcs_uri(
    chunk_gcs_uri: str,
    available_parser_labels: list[str]
) -> Optional[str]:
    
    if not gemini_client.is_available():
        debug_error("Gemini client not available.")
        return None

    # Download PDF content from GCS
    pdf_bytes = download_blob_bytes(chunk_gcs_uri)
    if not pdf_bytes:
        debug_error(f"Could not download PDF bytes from {chunk_gcs_uri}")
        return None

    # Create Part from bytes
    document_part = types.Part.from_bytes(
        data=pdf_bytes,
        mime_type="application/pdf"
    )

    # Create prompt instruction
    label_list_str = str(available_parser_labels).replace("'", '"')  # Gemini expects double quotes
    instruction = f"""given parser is 
{label_list_str}

and find the appropriate parser of the given PDF. 

# response format
1. Give only the string based on the available parser given
2. do not give any additional explanation or narration"""

    # Create message content
    contents = [
        types.Content(
            role="user",
            parts=[
                document_part,
                types.Part.from_text(text=".")
            ]
        ),
    ]

    try:
        response_text = ""
        
        # Use Gemini Client with streaming
        # Map safety settings to the dict format expected by the client wrapper
        safety_settings = [
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "OFF"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "OFF"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "OFF"},
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "OFF"},
        ]
        
        stream_gen = gemini_client.generate_content(
            contents=contents,
            model_name=config.GEMINI_OCR_MODEL_NAME,
            temperature=1,
            max_output_tokens=1024,
            system_instruction=instruction, # Pass simple string
            safety_settings=safety_settings,
            stream=True
        )

        for chunk_text in stream_gen:
            response_text += chunk_text

        debug_log(f"Raw Gemini response: {response_text}")  # Debugging output

        response_text = response_text.strip()
        if response_text in available_parser_labels:
            return response_text
        return None
    except Exception as e:
        debug_error(f"Error during classification: {e}")
        return None