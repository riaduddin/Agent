import asyncio
import logging
import sys
import os
from unittest.mock import MagicMock

# --- AGGRESSIVE MOCKING START ---
# Mock google.oauth2
sys.modules["google.oauth2"] = MagicMock()
sys.modules["google.oauth2.service_account"] = MagicMock()

# Mock google.cloud.firestore
mock_firestore = MagicMock()
sys.modules["google.cloud.firestore"] = mock_firestore
sys.modules["google.cloud.firestore_v1"] = MagicMock()
# Ensure Client can be instantiated
mock_firestore.Client.return_value = MagicMock()

# Mock google.cloud.aiplatform
sys.modules["google.cloud.aiplatform"] = MagicMock()
sys.modules["google.cloud.aiplatform_v1"] = MagicMock()

# Mock firebase_admin
sys.modules["firebase_admin"] = MagicMock()
sys.modules["firebase_admin.firestore"] = MagicMock()
# --- AGGRESSIVE MOCKING END ---

# Adjust path to allow imports
sys.path.append("/Volumes/development/georgia/new-version/georgia-v2/georgia-digitization-platform/backend")

# Now import the services
from app.services.metadata_extraction_service import MetadataExtractionService
from app.services.search_router_service import SearchRouterService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_extraction_agent():
    print("\n--- Verifying Metadata Extraction Agent ---")
    
    test_cases = [
        {
            "name": "Invoice Test",
            "filename": "invoice_123.pdf",
            "text": "INVOICE #INV-2023-001\nDate: 2023-10-15\nVendor: Acme Corp\nTotal: $500.00"
        },
        {
            "name": "License Test",
            "filename": "license_scan.jpg",
            "text": "DRIVING LICENSE\nName: John Doe\nLicense No: DL-987654321\nDOB: 01-01-1980\nExpires: 2025-01-01"
        },
        {
            "name": "Empty/Irrelevant Test",
            "filename": "random_note.txt",
            "text": "Just a random note about groceries."
        }
    ]

    for case in test_cases:
        print(f"\nTesting: {case['name']}")
        result = MetadataExtractionService.extract_identifiers(case['text'], case['filename'])
        print(f"Input Text Preview: {case['text'][:50]}...")
        print(f"Extracted Metadata: {result}")
        if case['name'] == "Invoice Test" and result.get("entity_id") == "INV-2023-001":
             print("PASS: Invoice ID extracted.")
        elif case['name'] == "License Test" and result.get("entity_name") == "John Doe":
             print("PASS: Name extracted.")
        elif case['name'] == "Empty/Irrelevant Test" and not result:
             print("PASS: Correctly returned empty for irrelevant text.")
        else:
             print("Check results manually.")

def verify_router_agent():
    print("\n--- Verifying Router Agent (Query Analysis) ---")
    
    queries = [
        "Show me the invoice INV-2023-001",
        "Find documents for John Doe",
        "Get me the car info for tag DHK-123",
        "Summarize the HR policy",
        "What happened in 2015?"
    ]

    for query in queries:
        print(f"\nQuery: '{query}'")
        result = SearchRouterService.analyze_query(query)
        print(f"Router Result: {result}")
        
        filters = result.get("filters", {})
        if "INV-2023-001" in query and filters.get("entity_id") == "INV-2023-001":
            print("PASS: Entity ID extracted.")
        elif "John Doe" in query and filters.get("entity_name") == "John Doe":
            print("PASS: Entity Name extracted.")
        elif "policy" in query and not result.get("is_entity_search"):
             print("PASS: Correctly identified as generic search.")
        else:
             print("Check results manually.")

if __name__ == "__main__":
    verify_extraction_agent()
    verify_router_agent()
