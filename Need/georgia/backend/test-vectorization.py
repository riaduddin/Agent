# backend/test-vectorization.py
import os
import sys
import traceback
import time # Import the time module
from dotenv import load_dotenv
import logging
import uuid

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Load Environment Variables ---
# Ensure the script can find the .env file relative to its location
backend_dir = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(backend_dir, '.env')
print(f"--- Loading .env file from: {dotenv_path} ---")
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)
    print(".env file loaded.")
else:
    print(f"WARN: .env file not found at {dotenv_path}. Relying on system environment variables.")
    # Exit if essential vars might be missing? Or proceed cautiously?
    # For a test script, maybe proceed but warn heavily.

# --- Add app directory to Python path ---
# This allows importing modules from the 'app' directory
sys.path.insert(0, os.path.abspath(os.path.join(backend_dir)))
print(f"Appended to sys.path: {os.path.abspath(os.path.join(backend_dir))}")

try:
    # --- Import necessary services AFTER adjusting path and loading .env ---
    from app.services import vertex_ai_service as vais
    from app import config # Import config to verify loaded values
    print("Successfully imported vertex_ai_service and config.")
except ImportError as e:
    print(f"\n--- FATAL ERROR: Failed to import application modules ---")
    print(f"Error: {e}")
    print("Please ensure:")
    print("1. You are running this script from the 'backend' directory OR the path setup is correct.")
    print("2. The necessary packages (google-cloud-aiplatform, etc.) are installed in your environment.")
    print(f"Current sys.path: {sys.path}")
    sys.exit(1)
except Exception as e:
    print(f"\n--- FATAL ERROR: An unexpected error occurred during import ---")
    print(f"Error: {e}")
    traceback.print_exc()
    sys.exit(1)


# --- Configuration Verification ---
print("\n--- Verifying Configuration ---")
print(f"PROJECT_ID: {config.PROJECT_ID}")
print(f"VERTEX_LOCATION: {config.VERTEX_LOCATION}")
print(f"EMBEDDING_MODEL_NAME: {config.EMBEDDING_MODEL_NAME}")
print(f"VECTOR_INDEX_NAME: {config.VECTOR_INDEX_NAME}")
print(f"VECTOR_INDEX_ENDPOINT_ID: {config.VECTOR_INDEX_ENDPOINT_ID}")
print(f"VECTOR_DEPLOYED_INDEX_ID: {config.VECTOR_DEPLOYED_INDEX_ID}")
print(f"VECTOR_PRIVATE_ENDPOINT_IP: {config.VECTOR_PRIVATE_ENDPOINT_IP}") # Will be None if not set
print("-----------------------------")

# --- Test Data ---
SAMPLE_TEXT = "This is a sample document chunk used for testing vectorization and search."
TEST_DATAPOINT_ID = f"test-vector-script-{uuid.uuid4()}" # Unique ID for testing upsert
TEST_FILENAME = "test_script_doc.txt"
TEST_START_PAGE = 1
TEST_END_PAGE = 1
TEST_GCS_URI = "gs://test-bucket/test_script_doc.txt" # Placeholder URI

def run_vectorization_tests():
    """Runs tests for embedding, upserting, and querying."""
    embedding = None

    # --- 1. Test Embedding Generation ---
    print(f"\n--- Testing Embedding Generation ---")
    print(f"Text: '{SAMPLE_TEXT}'")
    try:
        embedding = vais.get_text_embedding(SAMPLE_TEXT)
        if embedding:
            print(f"Successfully generated embedding (vector dimension: {len(embedding)}).")
            # print(f"Embedding preview: {embedding[:10]}...") # Optional: print first few values
        else:
            print("ERROR: Embedding generation returned None.")
            # No point continuing if embedding failed
            return
    except Exception as e:
        print(f"ERROR: Exception during embedding generation!")
        traceback.print_exc() # Print full traceback
        return # Stop if embedding fails

    # --- 2. Test Vector Upsert ---
    print(f"\n--- Testing Vector Upsert ---")
    print(f"Attempting to upsert datapoint ID: {TEST_DATAPOINT_ID}")
    try:
        # Note: add_embedding_to_index now returns None on success and raises Exception on failure
        vais.add_embedding_to_index(
            datapoint_id=TEST_DATAPOINT_ID,
            embedding=embedding,
            filename=TEST_FILENAME,
            start_page=TEST_START_PAGE,
            end_page=TEST_END_PAGE,
            gcs_uri=TEST_GCS_URI,
            text_preview=SAMPLE_TEXT
        )
        print(f"Successfully upserted datapoint {TEST_DATAPOINT_ID}.")
        # Wait a moment for the index to potentially update before querying
        print("Waiting 5 seconds for index update...")
        time.sleep(5)
    except Exception as e:
        print(f"ERROR: Exception during vector upsert!")
        print(f"Datapoint ID: {TEST_DATAPOINT_ID}")
        print(f"Index Name: {config.VECTOR_INDEX_NAME}")
        print(f"Endpoint ID: {config.VECTOR_INDEX_ENDPOINT_ID}")
        print(f"Deployed Index ID: {config.VECTOR_DEPLOYED_INDEX_ID}")
        traceback.print_exc() # Print full traceback
        # Continue to query test even if upsert failed, to test query separately
        # return # Optionally stop here

    # --- 3. Test Vector Query ---
    print(f"\n--- Testing Vector Query (Find Neighbors) ---")
    print(f"Querying with the generated embedding for '{SAMPLE_TEXT[:50]}...'")
    try:
        neighbors = vais.find_vector_neighbors(embedding, num_neighbors=5)
        if neighbors is not None:
            print(f"Successfully queried vector index. Found {len(neighbors)} neighbors:")
            for i, neighbor in enumerate(neighbors):
                print(f"  {i+1}. ID: {neighbor.get('id')}, Distance: {neighbor.get('distance'):.4f}")
                # Check if our test datapoint is among the neighbors
                if neighbor.get('id') == TEST_DATAPOINT_ID:
                    print(f"     -> Found our test datapoint!")
            if not neighbors:
                print("  (No neighbors found)")
        else:
            print("ERROR: Vector query returned None (indicating an error during the query process).")
    except Exception as e:
        print(f"ERROR: Exception during vector query!")
        print(f"Endpoint ID: {config.VECTOR_INDEX_ENDPOINT_ID}")
        print(f"Deployed Index ID: {config.VECTOR_DEPLOYED_INDEX_ID}")
        traceback.print_exc() # Print full traceback

    print("\n--- Vectorization Test Finished ---")

if __name__ == "__main__":
    # Basic check for essential config needed by the service
    if not all([config.PROJECT_ID, config.VERTEX_LOCATION, config.EMBEDDING_MODEL_NAME,
                config.VECTOR_INDEX_NAME, config.VECTOR_INDEX_ENDPOINT_ID, config.VECTOR_DEPLOYED_INDEX_ID]):
        print("\nERROR: Missing essential Vertex AI configuration in .env file.")
        print("Please ensure PROJECT_ID, VERTEX_LOCATION, EMBEDDING_MODEL_NAME, VECTOR_INDEX_NAME, VECTOR_INDEX_ENDPOINT_ID, and VECTOR_DEPLOYED_INDEX_ID are set.")
    else:
        run_vectorization_tests()
