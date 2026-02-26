import os
import argparse
from dotenv import load_dotenv
from google.cloud import documentai
from google.oauth2 import service_account # For explicit credential loading
from google.api_core.client_options import ClientOptions

def process_document_sample(
    project_id: str,
    location: str,
    processor_id: str, # Short processor ID
    gcs_uri: str,
    credentials_path: str,
    mime_type: str = "application/pdf"
):
    """
    Processes a document using Document AI with explicit credentials and full path construction.
    """
    print(f"--- Test Configuration ---")
    print(f"Project ID: {project_id}")
    print(f"Location: {location}")
    print(f"Processor ID (short): {processor_id}")
    print(f"GCS URI: {gcs_uri}")
    print(f"Credentials Path: {credentials_path}")
    print(f"MIME Type: {mime_type}")
    print(f"--------------------------\n")

    try:
        # Explicitly load credentials
        print(f"Loading credentials from: {credentials_path}")
        credentials = service_account.Credentials.from_service_account_file(credentials_path)
        print("Credentials loaded successfully.")

        # Initialize client with explicit credentials
        opts = ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com") if location != "us" else None
        print(f"Client options API endpoint: {opts.api_endpoint if opts else 'Default (us-documentai.googleapis.com)'}")
        
        client = documentai.DocumentProcessorServiceClient(credentials=credentials, client_options=opts)
        print("Document AI client initialized with explicit credentials.")

        # Construct the full processor path
        # This is the critical part to ensure it matches what the API expects
        name = client.processor_path(project_id, location, processor_id)
        print(f"Constructed Full Processor Path for API call: {name}\n")

        # Specify GCS document URI
        gcs_document = documentai.GcsDocument(gcs_uri=gcs_uri, mime_type=mime_type)
        process_request = documentai.ProcessRequest(
            name=name,
            gcs_document=gcs_document, # Use GcsDocument for GCS URIs
            skip_human_review=True
        )

        print(f"Sending request to Document AI API with processor path: {name}...")
        result = client.process_document(request=process_request)
        document = result.document

        print("\n--- Document AI Response ---")
        print(f"Document text length: {len(document.text)}")
        if document.entities:
            print(f"Found {len(document.entities)} entities.")
        # You can add more details from the 'document' object if needed
        print("--------------------------\n")
        print("SUCCESS: Document processed successfully!")

    except Exception as e:
        print(f"\n--- ERROR ---")
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        print(f"-------------")

if __name__ == "__main__":
    # Load .env file from the backend directory
    dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(dotenv_path):
        print(f"Loading .env file from: {dotenv_path}")
        load_dotenv(dotenv_path=dotenv_path)
    else:
        print(f"Warning: .env file not found at {dotenv_path}. Ensure environment variables are set.")

    # Get configuration from environment variables
    env_project_id = os.getenv("PROJECT_ID")
    env_location = os.getenv("DOCAI_LOCATION")
    env_processor_id = os.getenv("DOCUMENT_API_PROCESSOR_ID") # Short ID
    env_credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    # --- IMPORTANT: Set a default GCS URI for a test PDF file ---
    # You MUST replace this with a valid GCS URI of a PDF in your bucket
    # that the service account has access to.
    default_gcs_uri_for_testing = "gs://YOUR_BUCKET_NAME/path/to/your/test-document.pdf" 
    # Example: "gs://georgia_bucket/test_files/sample.pdf"

    parser = argparse.ArgumentParser(description="Test Document AI processing.")
    parser.add_argument("--project_id", default=env_project_id, help="Google Cloud Project ID.")
    parser.add_argument("--location", default=env_location, help="Document AI processor location (e.g., 'us', 'eu').")
    parser.add_argument("--processor_id", default=env_processor_id, help="Short ID of the Document AI processor.")
    parser.add_argument("--gcs_uri", default=default_gcs_uri_for_testing, help="GCS URI of the document to process.")
    parser.add_argument("--credentials_path", default=env_credentials_path, help="Path to Google Cloud service account key file.")
    
    args = parser.parse_args()

    if not all([args.project_id, args.location, args.processor_id, args.gcs_uri, args.credentials_path]):
        print("Error: Missing one or more required arguments or environment variables:")
        print("  --project_id or PROJECT_ID")
        print("  --location or DOCAI_LOCATION")
        print("  --processor_id or DOCUMENT_API_PROCESSOR_ID")
        print("  --gcs_uri (or update default_gcs_uri_for_testing in script)")
        print("  --credentials_path or GOOGLE_APPLICATION_CREDENTIALS")
        exit(1)
    
    if args.gcs_uri == "gs://YOUR_BUCKET_NAME/path/to/your/test-document.pdf":
        print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("ERROR: Please update 'default_gcs_uri_for_testing' in the script with a valid GCS URI,")
        print("       or provide one using the --gcs_uri argument.")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
        exit(1)

    process_document_sample(
        project_id=args.project_id,
        location=args.location,
        processor_id=args.processor_id,
        gcs_uri=args.gcs_uri,
        credentials_path=args.credentials_path
    )
