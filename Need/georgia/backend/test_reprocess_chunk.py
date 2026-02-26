
import os
import sys
import logging
from google.cloud import firestore

# Add backend directory to path so imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import app creation factory to initialize DB and Config
from app import create_app, db
from app.models.metadata_model import get_documents_with_legacy_chunks_batch
from app.services.doc_processing_helpers.bulk_processing_utils import reprocess_legacy_chunks_for_doc

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Force Project ID for test script
os.environ["GOOGLE_CLOUD_PROJECT"] = "shothik-project" 
os.environ["FIRESTORE_DATABASE_ID"] = "georgia"

def test_legacy_query():
    """
    Verifies that the new query strategy correctly finds documents with failed extraction chunks.
    """
    app = create_app()
    with app.app_context():
        print("\n" + "="*50)
        print("VERIFICATION: Testing 'get_documents_with_legacy_chunks_batch'")
        print("Strategy: Query 'extracted_entities._extraction_failed == True'")
        print("="*50 + "\n")

        try:
            # 1. Run the Batch Query
            logger.info("Running query...")
            doc_ids, last_snap, error = get_documents_with_legacy_chunks_batch(limit=50)

            if error:
                logger.error(f"❌ Query Failed: {error}")
                if "index" in str(error).lower():
                    print("\n" + "!"*80)
                    print("ACTION REQUIRED: You need to create a Firestore Index.")
                    print("Look for the URL in the error message above (if provided by Firestore client).")
                    print("!"*80 + "\n")
                return

            print(f"\n✅ Query Successful!")
            print(f"found {len(doc_ids)} unique documents with failed chunks.")
            
            if len(doc_ids) == 0:
                print("\n⚠️  No documents found. This might mean:")
                print("   a) There are strictly 0 chunks with '_extraction_failed: true'")
                print("   b) The field name matches exactly but no data has it set.")
                
                # Check random chunk to verify data shape manually?
                print("   checking a random chunk from DB to see structure...")
                chunks_ref = db.collection("document_chunks")
                sample = chunks_ref.limit(1).get()
                if sample:
                    print(f"   Sample ID: {sample[0].id}")
                    print(f"   Extracted Entities: {sample[0].to_dict().get('extracted_entities')}")
            else:
                print(f"Documents found: {list(doc_ids)}")
                
                # Optional: Simulate dry run for first doc
                first_doc_id = list(doc_ids)[0]
                print(f"\n[DRY RUN] Simulating reprocessing setup for Doc: {first_doc_id}")
                
                # Retrieve chunks for this doc to confirm they are indeed legacy/failed
                chunks = db.collection("document_chunks").where("parent_doc_id", "==", first_doc_id).stream()
                legacy_count = 0
                for c in chunks:
                    d = c.to_dict()
                    ee = d.get('extracted_entities', {})
                    if ee.get('_extraction_failed') == True:
                        legacy_count += 1
                        print(f"   - Chunk {c.id} IS a target (extraction_failed=True)")
                    else:
                        print(f"   - Chunk {c.id} is OK/Other (method={ee.get('_extraction_method')})")
                
                print(f"   Confirmed: {legacy_count} chunks in doc {first_doc_id} match the target criteria.")

        except Exception as e:
            logger.error(f"Test script failed with exception: {e}", exc_info=True)

if __name__ == "__main__":
    test_legacy_query()
