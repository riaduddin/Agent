# backend/app/config.py
import os
from dotenv import load_dotenv
from datetime import timedelta # Import timedelta

# Load environment variables from .env file located in the backend root (one level up from 'app')
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path=dotenv_path)

# --- Debug Mode Toggle ---
# Set to True to enable debug logging throughout the application
# Set to False to disable all debug output for production
DEBUG_MODE = os.getenv('DEBUG_MODE', 'false').lower() == 'true'

# --- General Flask Settings ---
SECRET_KEY = os.getenv('FLASK_SECRET_KEY', 'a_default_fallback_secret_key')
JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY')
# Set access token expiration to 1 day
JWT_ACCESS_TOKEN_EXPIRES = timedelta(days=1)
ROUTE_PREFIX = os.getenv('ROUTE_PREFIX', '').rstrip('/')

# --- GCP Project Settings ---
PROJECT_ID = os.getenv('PROJECT_ID')
# Determine location for Document AI (e.g., 'us' or 'eu')
DOCAI_LOCATION = os.getenv('LOCATION')
# Determine location for Vertex AI (e.g., 'us-central1')
VERTEX_LOCATION = os.getenv('VERTEX_LOCATION', 'us-central1') # Default if not set
GOOGLE_APPLICATION_CREDENTIALS = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')

# --- GCS Settings ---
BUCKET_NAME = os.getenv('BUCKET_NAME')
# Optional prefix for bulk processing source files
GCS_BULK_PROCESSING_PREFIX = os.getenv('GCS_BULK_PROCESSING_PREFIX', '') # Default to empty string (root)
# Ensure GCS_SOURCE_ROOT always ends with a trailing slash for consistency
_gcs_source_root = os.getenv('GCS_SOURCE_ROOT', 'Pending Files')
GCS_SOURCE_ROOT = _gcs_source_root if _gcs_source_root.endswith('/') else f"{_gcs_source_root}/"
FILE_MANAGEMENT_BUCKET_NAME = os.getenv('FILE_MANAGEMENT_BUCKET_NAME', BUCKET_NAME)

# --- Firestore Settings ---
FIRESTORE_DATABASE_ID = os.getenv('FIRESTORE_DATABASE_ID', '(default)') # Get specific DB ID

# --- Document AI Settings ---
DOCUMENT_API_PROCESSOR_ID = os.getenv('DOCUMENT_API_PROCESSOR_ID') # This is the current default parser
DEFAULT_PARSER_PROCESSOR_ID = os.getenv('DEFAULT_PARSER_PROCESSOR_ID', DOCUMENT_API_PROCESSOR_ID) # Fallback to existing default
DEFAULT_CLASSIFIED_DOCUMENT_TYPE = "GENERAL_DOCUMENT"


# --- Vertex AI Settings ---
VECTOR_INDEX_NAME = os.getenv('VECTOR_INDEX_NAME')
VECTOR_INDEX_ENDPOINT_ID = os.getenv('VECTOR_INDEX_ENDPOINT_ID')
VECTOR_DEPLOYED_INDEX_ID = os.getenv('VECTOR_DEPLOYED_INDEX_ID')
VECTOR_PRIVATE_ENDPOINT_IP = os.getenv('VECTOR_PRIVATE_ENDPOINT_IP') # Added for private endpoint connection
# Embedding model name (consistent with PRD/vertex_ai_service)
EMBEDDING_MODEL_NAME = "text-embedding-004"
# Gemini model names (centralized)
GEMINI_OCR_MODEL_NAME = os.getenv('GEMINI_OCR_MODEL_NAME', 'gemini-2.5-flash')
GEMINI_CHAT_MODEL_NAME = os.getenv('GEMINI_CHAT_MODEL_NAME', 'gemini-2.5-flash')
GEMINI_MODEL_NAME = GEMINI_OCR_MODEL_NAME # Backward compatibility fallback

# --- Pub/Sub Settings ---
PUBSUB_TOPIC_ID = os.getenv('PUBSUB_TOPIC_ID', 'test_topic')
PUBSUB_SUBSCRIPTION_ID = os.getenv('PUBSUB_SUBSCRIPTION_ID', 'test_topic-sub')

# Granular Pub/Sub settings for different scheduler tasks
# These default to the general settings if not specified
PUBSUB_BATCH_TOPIC_ID = os.getenv('PUBSUB_BATCH_TOPIC_ID', PUBSUB_TOPIC_ID)
PUBSUB_BATCH_SUBSCRIPTION_ID = os.getenv('PUBSUB_BATCH_SUBSCRIPTION_ID', PUBSUB_SUBSCRIPTION_ID)
PUBSUB_LEGACY_TOPIC_ID = os.getenv('PUBSUB_LEGACY_TOPIC_ID', 'legacy-reprocess-topic')
PUBSUB_LEGACY_SUBSCRIPTION_ID = os.getenv('PUBSUB_LEGACY_SUBSCRIPTION_ID', 'legacy-reprocess-sub')

# Cloud Scheduler Job Names (for startup validation)
SCHEDULER_JOB_BATCH_RUN = os.getenv('SCHEDULER_JOB_BATCH_RUN')
SCHEDULER_JOB_LEGACY_REPROCESS = os.getenv('SCHEDULER_JOB_LEGACY_REPROCESS')
SCHEDULER_LOCATION = os.getenv('SCHEDULER_LOCATION', 'us-central1')

# --- Redis Settings (for Task Queue and Caching) ---
REDIS_HOST =os.getenv('REDIS_HOST')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_DB = int(os.getenv('REDIS_DB', 0))
# Load username and password for Redis Cloud
REDIS_USERNAME = os.getenv('REDIS_USERNAME') # Added
REDIS_PASSWORD = os.getenv('REDIS_PASSWORD') # Added (uncommented and used)
# Cache TTL for vector search results (increased for better performance)
VECTOR_SEARCH_CACHE_TTL_SECONDS = int(os.getenv('VECTOR_SEARCH_CACHE_TTL_SECONDS', 7200)) # Default to 2 hours
# Cache TTL for Firestore chunk lookups (increased for better performance)
FIRESTORE_CHUNK_CACHE_TTL_SECONDS = int(os.getenv('FIRESTORE_CHUNK_CACHE_TTL_SECONDS', 7200)) # Default to 2 hours
# Cache TTL for query embeddings (longer since embeddings are deterministic)
EMBEDDING_CACHE_TTL_SECONDS = int(os.getenv('EMBEDDING_CACHE_TTL_SECONDS', 604800)) # Default to 7 days
# Cache TTL for query variations
QUERY_VARIATIONS_CACHE_TTL_SECONDS = int(os.getenv('QUERY_VARIATIONS_CACHE_TTL_SECONDS', 86400)) # Default to 24 hours

# --- SAML Settings ---
SAML_SP_ENTITY_ID = os.getenv('SAML_SP_ENTITY_ID')
SAML_SP_ACS_URL = os.getenv('SAML_SP_ACS_URL')
SAML_SP_SLO_URL = os.getenv('SAML_SP_SLO_URL')
SAML_SP_X509CERT = os.getenv('SAML_SP_X509CERT')
SAML_SP_PRIVATE_KEY = os.getenv('SAML_SP_PRIVATE_KEY')
SAML_IDP_ENTITY_ID = os.getenv('SAML_IDP_ENTITY_ID')
SAML_IDP_SSO_URL = os.getenv('SAML_IDP_SSO_URL')
SAML_IDP_SLO_URL = os.getenv('SAML_IDP_SLO_URL')
SAML_IDP_X509CERT = os.getenv('SAML_IDP_X509CERT')


# --- Retry Configuration (from .env with defaults) ---
GCS_DOWNLOAD_MAX_RETRY = int(os.getenv('GCS_DOWNLOAD_MAX_RETRY', '3'))
GCS_UPLOAD_MAX_RETRY = int(os.getenv('GCS_UPLOAD_MAX_RETRY', '3'))

# --- Worker Settings ---
MAX_CONCURRENT_CHUNK_TASKS = int(os.getenv('MAX_CONCURRENT_CHUNK_TASKS', '5')) # Max parallel chunks per parent doc

# --- Legacy Chunk Reprocessing Settings ---
# Maximum age in days for legacy chunks to be reprocessed during nightly batch jobs
# Chunks older than this threshold will be skipped to optimize processing
LEGACY_CHUNK_MAX_AGE_DAYS = int(os.getenv('LEGACY_CHUNK_MAX_AGE_DAYS', '7'))

# --- Document Preprocessing Validation Settings ---
MIN_PDF_PAGE_COUNT = int(os.getenv('MIN_PDF_PAGE_COUNT', '1')) # Minimum allowed pages for a PDF
MAX_PDF_PAGE_COUNT = int(os.getenv('MAX_PDF_PAGE_COUNT', '2000')) # Maximum allowed pages for a PDF
# Default password to try for encrypted PDFs, if any. Empty means no default password.
DEFAULT_PDF_PASSWORD = os.getenv('DEFAULT_PDF_PASSWORD', '')
DOCAI_OCR_MAX_RETRY = int(os.getenv('DOCAI_OCR_MAX_RETRY', '5'))
FIRESTORE_SAVE_MAX_RETRY = int(os.getenv('FIRESTORE_SAVE_MAX_RETRY', '3'))
FIRESTORE_UPDATE_MAX_RETRY = int(os.getenv('FIRESTORE_UPDATE_MAX_RETRY', '3'))
EMBEDDING_MAX_RETRY = int(os.getenv('EMBEDDING_MAX_RETRY', '5'))
VECTOR_UPSERT_MAX_RETRY = int(os.getenv('VECTOR_UPSERT_MAX_RETRY', '3'))

# --- Document Categories ---
VALID_CATEGORIES = ["1099", "CHECKS", "CHILD_WELFARE_REPORTS", "LEAVE_DOCUMENTS", "MONTH_END_REPORTS", "PAYROLL_REPORTS_N_DOCUMENTS", "PENDING_FILES", "PERSONNEL_FILES", "TRAVEL_REPORTS", "OTHER"]

# --- External Categories API Settings ---
ADMIN_SERVER_URL = os.getenv('ADMIN_SERVER_URL', 'http://localhost:5005/api')
CATEGORIES_API_URL = f"{ADMIN_SERVER_URL}/file-categories/all"
# CATEGORIES_API_URL = os.getenv('CATEGORIES_API_URL', 'https://doc-digitization.shothik.ai/admin-api/file-categories/all')
# API uses X-API-Key header with JWT_SECRET_KEY value

# --- External Accessible Documents API Settings ---
ACCESSIBLE_DOCS_API_URL = f"{ADMIN_SERVER_URL}/file-categories/accessible"
# API uses Authorization header with Bearer token

# --- Validation ---
# Check for essential missing variables and warn (but don't crash on startup)
# This allows Cloud Run to start the service even if some variables are missing
# Features requiring these variables will fail at runtime with clearer error messages
required_vars = {
    'JWT_SECRET_KEY': JWT_SECRET_KEY,
    'PROJECT_ID': PROJECT_ID,
    'DOCAI_LOCATION': DOCAI_LOCATION,
    'VERTEX_LOCATION': VERTEX_LOCATION,
    'GOOGLE_APPLICATION_CREDENTIALS': GOOGLE_APPLICATION_CREDENTIALS,
    'BUCKET_NAME': BUCKET_NAME,
    'DOCUMENT_API_PROCESSOR_ID': DOCUMENT_API_PROCESSOR_ID,
    'FIRESTORE_DATABASE_ID': FIRESTORE_DATABASE_ID,
    # Redis vars are technically optional if defaults are acceptable for local dev
    # 'REDIS_HOST': REDIS_HOST,
    # 'REDIS_PORT': REDIS_PORT,
    # Vector Search vars are now optional at startup, will be checked/created by setup endpoint
    # 'VECTOR_INDEX_NAME': VECTOR_INDEX_NAME,
    # 'VECTOR_INDEX_ENDPOINT_ID': VECTOR_INDEX_ENDPOINT_ID,
    # 'VECTOR_DEPLOYED_INDEX_ID': VECTOR_DEPLOYED_INDEX_ID,
}

# Separate check for truly required vars at startup
essential_vars = {k: v for k, v in required_vars.items() if 'VECTOR_' not in k}
missing_vars = [name for name, value in essential_vars.items() if not value]
if missing_vars:
    # Changed from raise to log WARNING - allows app to start for Cloud Run health checks
    import logging
    logging.warning(f"Missing required environment variables: {', '.join(missing_vars)}")
    logging.warning("Features requiring these variables will fail at runtime.")
    # Uncomment the line below to enforce strict validation (will prevent startup):
    # raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Validate credentials path
if GOOGLE_APPLICATION_CREDENTIALS and not os.path.exists(GOOGLE_APPLICATION_CREDENTIALS):
     # Note: This check assumes the path is absolute or relative to the execution directory.
     # If running in Docker, the path needs to be valid within the container.
     import logging
     logging.warning(f"GOOGLE_APPLICATION_CREDENTIALS path '{GOOGLE_APPLICATION_CREDENTIALS}' does not exist.")
     # Depending on ADC fallback logic, might not want to raise an error here, just warn.
     # raise FileNotFoundError(f"Service account key file not found at {GOOGLE_APPLICATION_CREDENTIALS}")

# Configuration loaded - debug message handled by debug_logger when DEBUG_MODE is True

# Vector Search Tuning (optimized for 50 chunks to LLM)
VECTOR_SEARCH_TOP_K_ORIGINAL = int(os.getenv('VECTOR_SEARCH_TOP_K_ORIGINAL', 40))
VECTOR_SEARCH_TOP_K_VARIATION = int(os.getenv('VECTOR_SEARCH_TOP_K_VARIATION', 10))
VECTOR_SEARCH_MERGED_CAP = int(os.getenv('VECTOR_SEARCH_MERGED_CAP', 100)) # Increased to support fusion results
VECTOR_SEARCH_VARIATION_COUNT = int(os.getenv('VECTOR_SEARCH_VARIATION_COUNT', 3))

# Vector Search Progressive Fallback Configuration
# Multipliers to progressively increase neighbor count if no results found
# Example: With TOP_K_ORIGINAL=25, will try: 25 (1x) → 75 (3x) → 125 (5x)
VECTOR_SEARCH_FALLBACK_MULTIPLIERS = [1, 3, 5]  # Try 1x, then 3x, then 5x neighbors

# Quality thresholds for fallback decisions
# Minimum number of neighbors required before considering the search successful
# If fewer results found, fallback will continue to try broader searches
VECTOR_SEARCH_MIN_QUALITY_NEIGHBORS = int(os.getenv('VECTOR_SEARCH_MIN_QUALITY_NEIGHBORS', 30))

# Maximum distance threshold (optional) - set to None to disable
# For DOT_PRODUCT_DISTANCE, values closer to 1 are better matches
# Neighbors with distance > this value will be filtered out
VECTOR_SEARCH_MAX_DISTANCE_THRESHOLD = float(os.getenv('VECTOR_SEARCH_MAX_DISTANCE_THRESHOLD', 0.35)) or None

# Multi-Query Entity-Based Search Configuration
# All variables default to optimal values if not present in environment
MULTI_QUERY_SEARCH_ENABLED = os.getenv('MULTI_QUERY_SEARCH_ENABLED', 'true').lower() == 'true'

# Entity Query Parameters
ENTITY_QUERY_TOP_K = int(os.getenv('ENTITY_QUERY_TOP_K', 20))  # Neighbors per entity query
ENTITY_QUERY_MAX_COUNT = int(os.getenv('ENTITY_QUERY_MAX_COUNT', 5))  # Max entity queries to generate
ENTITY_QUERY_WEIGHT = float(os.getenv('ENTITY_QUERY_WEIGHT', 1.2))  # Entity match weight (>1.0 = higher priority)

# Result Fusion Parameters
FUSION_FINAL_TOP_K = int(os.getenv('FUSION_FINAL_TOP_K', 100))  # After fusion, before final selection
FUSION_MATCH_COUNT_WEIGHT = int(os.getenv('FUSION_MATCH_COUNT_WEIGHT', 100))  # Multiplier for match count
FUSION_SCORE_WEIGHT = int(os.getenv('FUSION_SCORE_WEIGHT', 10))  # Multiplier for weighted score
FUSION_DISTANCE_PENALTY = int(os.getenv('FUSION_DISTANCE_PENALTY', 5))  # Penalty for distance
FUSION_GUARANTEED_PER_QUERY = int(os.getenv('FUSION_GUARANTEED_PER_QUERY', 10)) # Top N results from EACH query to guarantee in final list

# Context Relevance Validation (Path C Fallback)
# Validates that retrieved chunks actually contain the query's critical identifiers
# Threshold: 0.4 means at least 40% weighted match (identifiers=50%, names=30%, dates/amounts=20%)
# Set to 0.0 to disable Path C validation entirely
CONTEXT_RELEVANCE_THRESHOLD = float(os.getenv('CONTEXT_RELEVANCE_THRESHOLD', 0.4))

# LLM Context Window Configuration
# Number of chunks to send to Gemini (default 100 for high recall with Flash models)
LLM_CONTEXT_CHUNK_LIMIT = int(os.getenv('LLM_CONTEXT_CHUNK_LIMIT', 100))


# Response Validation Configuration
# Enable/disable post-generation validation of LLM responses
ENABLE_RESPONSE_VALIDATION = os.getenv('ENABLE_RESPONSE_VALIDATION', 'true').lower() == 'true'

# Minimum confidence threshold for validation scores (0.0-1.0)
# If relevance or grounding score is below this, response is flagged
VALIDATION_CONFIDENCE_THRESHOLD = float(os.getenv('VALIDATION_CONFIDENCE_THRESHOLD', 0.7))
