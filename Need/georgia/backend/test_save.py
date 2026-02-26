import os
import datetime
import traceback

# Mimic worker environment loading
backend_dir = os.path.dirname(os.path.abspath(__file__))
from dotenv import load_dotenv
load_dotenv(os.path.join(backend_dir, '.env'))

try:
    from app import db, config
    print(f"DB Project: {db.project}")
    # print(f"DB Database: {db.database}") # database property might not exist on Client object directly depending on version, but let's try
    
    from app.models.metadata_model import save_chunk_details

    chunk_id = "TEST_CHUNK_MANUAL_SAVE"
    details_payload = {
         "chunk_id": chunk_id,
         "full_text": "This is a manual test text.",
         "entity_metadata": {"test_key": "test_value"},
         "vector_restricts": [{"namespace": "test", "allow_list": ["val"]}],
         "processing_timestamp": datetime.datetime.now(tz=datetime.timezone.utc),
         "worker_id": "manual_script",
         "original_filename": "manual_test.pdf"
     }

    print("Attempting to save chunk details...")
    save_chunk_details(chunk_id, details_payload)
    print("Save function returned.")
    
    # Verify immediately
    doc_ref = db.collection('document_chunk_details').document(chunk_id)
    doc = doc_ref.get()
    if doc.exists:
        print("SUCCESS: Test chunk saved and verified.")
        # Cleanup
        doc_ref.delete()
        print("Test chunk deleted.")
    else:
        print("FAILURE: Save function returned but document not found.")

except Exception as e:
    print("EXCEPTION CAUGHT:")
    print(e)
    traceback.print_exc()
