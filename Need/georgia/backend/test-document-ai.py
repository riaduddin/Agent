import os
from dotenv import load_dotenv
from google.cloud import documentai

# Load environment variables from .env file
load_dotenv()

# Get configuration from environment variables
PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("LOCATION")  # e.g., 'us' or 'eu'
PROCESSOR_ID = "25e702c82ed8afd6" # The ID of your Document AI processor

# --- Print loaded values for verification ---
print(f"--- Loaded Environment Variables ---")
print(f"PROJECT_ID: {PROJECT_ID}")
print(f"LOCATION: {LOCATION}")
print(f"DOCUMENT_API_PROCESSOR_ID: {PROCESSOR_ID}")
print(f"------------------------------------")
# ------------------------------------------

# --- Configuration ---
# Option: Process a file in GCS
GCS_URI = "gs://georgia_bucket/chunks/3I3fKpJcBUh6qvsUWQkB/ca4eea3e-57d7-416d-9e1a-966bb1b013fd.pdf" # Ensure this GCS URI is valid and accessible
GCS_MIME_TYPE = "application/pdf"
# ---------------------

def process_document_gcs_sample(
    project_id: str, location: str, processor_id: str, gcs_uri: str, gcs_mime_type: str
):
    """
    Processes a document stored in Google Cloud Storage using the Document AI API.
    """
    if not gcs_uri or not gcs_mime_type:
        print("Error: gcs_uri and gcs_mime_type are required.")
        return

    print(f"\n--- Starting Document AI GCS Processing ---")
    print(f"Using Processor: {processor_id}")
    print(f"Project ID: {project_id}, Location: {location}")
    print(f"GCS URI: {gcs_uri}")
    print(f"GCS MIME Type: {gcs_mime_type}")

    # You must set the `api_endpoint` if you use a location other than "us".
    opts = {"api_endpoint": f"{location}-documentai.googleapis.com"} if location != "us" else {}
    print(f"Client Options: {opts}")

    try:
        client = documentai.DocumentProcessorServiceClient(client_options=opts)

        # The full resource name of the processor, e.g.:
        # `projects/{project_id}/locations/{location}/processors/{processor_id}`
        name = client.processor_path(project_id, location, processor_id)
        print(f"Constructed Processor Name: {name}")

        # Load GCS URI into Document AI GcsDocument Structure
        print("Creating GcsDocument...")
        gcs_document = documentai.GcsDocument(gcs_uri=gcs_uri, mime_type=gcs_mime_type)

        # Configure the process request
        # skip_human_review=True means the process skips the Human Review step.
        # Human Review is an optional step for processor types that support it.
        # https://cloud.google.com/document-ai/docs/human-review
        print("Creating ProcessRequest...")
        request = documentai.ProcessRequest(
            name=name,
            gcs_document=gcs_document,
            skip_human_review=True,
        )

        # Use the Process API to extract documents
        print("Sending request to Document AI API...")
        result = client.process_document(request=request)
        print("Received response from Document AI API.")

        # For a full list of Document object attributes, please reference this page:
        # https://cloud.google.com/document-ai/docs/reference/rest/v1/Document
        document = result.document

        print(f"\n--- Document AI Response ---")
        if document.text:
            print(f"Extracted Text ({len(document.text)} characters):")
            # Print first 500 characters for brevity
            print(document.text)
        else:
            print("No text extracted from the document.")

        # You can add more detailed processing here, e.g., iterating through pages, entities, etc.
        # print(f"The document contains {len(document.pages)} page(s).")

    except Exception as e:
        print(f"\n--- Error processing document ---")
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("--- Running Document AI Test Script ---")
    if not all([PROJECT_ID, LOCATION, PROCESSOR_ID, GCS_URI, GCS_MIME_TYPE]):
        print("\nError: Missing required configuration.")
        print("Please ensure PROJECT_ID, LOCATION, DOCUMENT_API_PROCESSOR_ID are set in your .env file,")
        print("and GCS_URI and GCS_MIME_TYPE are defined in the script.")
    else:
        print("Configuration loaded successfully.")
        process_document_gcs_sample(
            project_id=PROJECT_ID,
            location=LOCATION,
            processor_id=PROCESSOR_ID,
            gcs_uri=GCS_URI,
            gcs_mime_type=GCS_MIME_TYPE
        )
    print("\n--- Document AI Test Script Finished ---")
