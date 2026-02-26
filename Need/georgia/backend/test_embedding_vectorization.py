# backend/test_embedding_vectorization.py
import logging
import time
import sys
import os
from dotenv import load_dotenv # Import load_dotenv

# --- Setup logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Add app directory to sys.path to allow imports ---
# This assumes the script is run from the 'backend' directory's parent (e.g., 'c:/georgia')
# Or that the necessary paths are already set in PYTHONPATH
# For robustness when running directly, let's add the path explicitly
backend_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.join(backend_dir, 'app')
sys.path.insert(0, backend_dir) # Add backend dir to allow 'from app import ...'
sys.path.insert(0, os.path.dirname(backend_dir)) # Add parent dir if needed

# --- Load .env file explicitly ---
dotenv_path = os.path.join(backend_dir, '.env')
logger.info(f"Attempting to load .env file from: {dotenv_path}")
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)
    logger.info(".env file loaded.")
    # Optionally print loaded vars for debugging, be careful with secrets
    # print(f"GOOGLE_APPLICATION_CREDENTIALS: {os.getenv('GOOGLE_APPLICATION_CREDENTIALS')}")
else:
    logger.warning(f".env file not found at {dotenv_path}. Relying on system environment variables.")


logger.info(f"System Path: {sys.path}")

# --- Load .env file explicitly ---
# ... (dotenv loading code remains the same) ...

# --- Force re-import after loading .env ---
import importlib
try:
    logger.info("Attempting to import services...")
    # Import config first AFTER loading dotenv
    from app import config
    importlib.reload(config) # Force reload config
    logger.info(f"Config reloaded. VECTOR_INDEX_NAME_CONFIG: {config.VECTOR_INDEX_NAME}") # Debug print
    # Now import the service which uses the config
    from app.services import vertex_ai_service as vais
    importlib.reload(vais) # Force reload service
    logger.info("Imports successful.")
except ImportError as e:
    logger.error(f"ImportError: {e}. Make sure the script is run from the project root directory or the virtual environment is activated correctly.", exc_info=True)
    sys.exit(1)
except Exception as e:
     logger.error(f"An unexpected error occurred during import: {e}", exc_info=True)
     sys.exit(1)


def test_embedding_and_upsert():
    """Tests text embedding and vector index upsert."""
    logger.info("--- Starting Embedding and Vectorization Test ---")

    # --- Test Data ---
    sample_text = "This is a sample document text about bridge engineering reports."
    sample_id = f"test-datapoint-{int(time.time())}" # Unique ID for testing

    # --- 1. Test Embedding Generation ---
    logger.info(f"Attempting to generate embedding for text: '{sample_text}'")
    embedding = None
    try:
        embedding = vais.get_text_embedding(sample_text)
    except Exception as e:
        logger.error(f"Exception during get_text_embedding call: {e}", exc_info=True)

    if embedding:
        logger.info(f"Successfully generated embedding. Length: {len(embedding)}. First few values: {embedding[:5]}...")
    else:
        logger.error("Failed to generate embedding (returned None). Check vertex_ai_service logs for details.")
        logger.info("--- Test Finished (Embedding Failed) ---")
        return # Cannot proceed to upsert without embedding

    # --- 2. Test Vector Index Upsert ---
    logger.info(f"Attempting to upsert embedding to index with ID: '{sample_id}'")
    upsert_success = False
    try:
        upsert_success = vais.add_embedding_to_index(sample_id, embedding)
    except Exception as e:
         logger.error(f"Exception during add_embedding_to_index call: {e}", exc_info=True)

    if upsert_success:
        logger.info(f"Successfully called upsert function for ID '{sample_id}'. (Note: Actual upsert might be asynchronous in GCP).")
    else:
        logger.error(f"Failed to call upsert function for ID '{sample_id}'. Check vertex_ai_service logs for details.")

    logger.info("--- Test Finished ---")

if __name__ == "__main__":
    # Ensure necessary initialization happens (aiplatform.init is called in vertex_ai_service)
    logger.info("Running test script...")
    test_embedding_and_upsert()
    logger.info("Test script finished.")
