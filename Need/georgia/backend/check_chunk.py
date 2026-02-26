from google.cloud import firestore
from google.oauth2 import service_account
import os

# Initialize Firestore
cred_path = "key.json"
if not os.path.exists(cred_path):
    print(f"Error: {cred_path} not found.")
    exit(1)

credentials = service_account.Credentials.from_service_account_file(cred_path)

db = firestore.Client(
    project="shothik-project",
    credentials=credentials,
    database="georgia"
)

parent_doc_id = "Ys3L19f7WbZB59M805p9"
print(f"Checking chunks for parent doc: {parent_doc_id}")

chunks = db.collection("document_chunks").where("original_doc_firestore_id", "==", parent_doc_id).stream()
found_chunks = list(chunks)

print(f"Found {len(found_chunks)} chunks for parent.")

for chunk in found_chunks:
    chunk_id = chunk.id
    print(f"Checking details for chunk: {chunk_id}")
    
    details_ref = db.collection("document_chunk_details").document(chunk_id)
    details = details_ref.get()
    
    if details.exists:
        print(f"SUCCESS: Details found for {chunk_id}. Keys: {list(details.to_dict().keys())}")
    else:
        print(f"FAILURE: No details for {chunk_id}")
