# georgia-digitization-platform/backend/tests/test_delete_embeddings.py
import os
import logging
import pytest

from app.services import vertex_ai_service
from app import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration (from config module) ---
# PROJECT_ID = config.PROJECT_ID
# LOCATION = config.VERTEX_LOCATION
# VECTOR_INDEX_NAME_CONFIG = config.VECTOR_INDEX_NAME # Name from config (optional)
# EMBEDDING_MODEL_NAME = config.EMBEDDING_MODEL_NAME
# GEMINI_MODEL_NAME = config.GEMINI_MODEL_NAME
# INDEX_ENDPOINT_ID_CONFIG = config.VECTOR_INDEX_ENDPOINT_ID # ID from config (optional)
# DEPLOYED_INDEX_ID_CONFIG = config.VECTOR_DEPLOYED_INDEX_ID # ID from config (optional)
# PRIVATE_ENDPOINT_IP_CONFIG = config.VECTOR_PRIVATE_ENDPOINT_IP # Private IP from config (optional)

# # --- Default/Generated Names (used if config values are missing) ---
# DEFAULT_ENDPOINT_DISPLAY_NAME = "georgia-doc-index-endpoint"
# DEFAULT_INDEX_DISPLAY_NAME = "georgia-doc-index"
# DEFAULT_DEPLOYED_INDEX_ID = "deployed_georgia_index_001" # Must follow specific format rules
# EMBEDDING_DIMENSIONS = 768 # For text-embedding-004

def test_delete_all_embeddings():
    """Tests the delete_all_embeddings function."""
    try:
        # Call the function with parameters from the config
        vertex_ai_service.delete_datapoints_from_vector_search_index(
            datapoint_ids=["383ead20-746f-4998-b027-7e3962721350", "86a20c4f-23c8-47c3-a87c-5fda7f31000b"],
        )
        logger.info("Successfully initiated deletion of all embeddings (test).")
        # Add assertions here to check if the deletion was successful
        # For example, you could check if the index is empty after the deletion
        # However, this might require additional setup and API calls to verify
        # For now, we'll just check that the function ran without raising an exception
        print("worked")  # Replace with more specific assertions if possible

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        assert False, f"Test failed with exception: {e}"