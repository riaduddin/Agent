from google.cloud import firestore
import os
from dotenv import load_dotenv

# Load env to get credentials
load_dotenv()

project_id = os.getenv('PROJECT_ID')
database_id = os.getenv('FIRESTORE_DATABASE_ID', '(default)')
key_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')

print(f"Connecting to project: {project_id}, database: {database_id} using {key_path}")

db = firestore.Client(project=project_id, database=database_id)

# Try to get one document from document_chunks
chunks_ref = db.collection('document_chunks')
docs = chunks_ref.limit(1).stream()

found = False
for doc in docs:
    found = True
    print(f"Document ID: {doc.id}")
    data = doc.to_dict()
    # Filter out long text for readability
    if 'ocr_text_preview' in data: data['ocr_text_preview'] = data['ocr_text_preview'][:100] + "..."
    if 'extracted_text_preview' in data: data['extracted_text_preview'] = data['extracted_text_preview'][:100] + "..."
    print("Document Data Keys:", list(data.keys()))
    print("Example Data Snippet:", {k: data[k] for k in list(data.keys())[:10]})
    if 'extracted_entities' in data:
        print("extracted_entities:", data['extracted_entities'])
    if 'entities' in data:
        print("entities array found:", data['entities'])
    else:
        print("NO 'entities' array found.")

if not found:
    print("No documents found in 'document_chunks' collection.")
