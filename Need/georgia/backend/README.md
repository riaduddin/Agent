# Georgia Digitization Platform - Backend

A comprehensive Flask-based backend API for document processing, management, and AI-powered search capabilities. The backend integrates with Google Cloud services including Firestore, Cloud Storage, Document AI, and Vertex AI.

## Table of Contents

- [Quick Start](#quick-start)
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Creating Admin Users](#creating-admin-users)
- [Running the Backend](#running-the-backend)
- [Testing the APIs](#testing-the-apis)
- [API Documentation](#api-documentation)
- [Development](#development)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)
- [Additional Resources](#additional-resources)

---

## Quick Start

1. **Install dependencies:**
   ```bash
   cd georgia-digitization-platform/backend
   python -m venv venv
   venv\Scripts\activate  # Windows
   # OR
   source venv/bin/activate  # Linux/Mac
   pip install -r requirements.txt
   ```

2. **Configure environment:**
   ```bash
   cp example.env .env
   # Edit .env with your GCP project details
   ```

3. **Create admin user:**
   ```bash
   python create_admin.py
   # OR use ADC (if you have gcloud CLI):
   python create_admin_adc.py
   ```

4. **Start the server:**
   ```bash
   python run.py
   ```

5. **Test the API:**
   ```bash
   curl http://localhost:5001/health
   ```

For detailed instructions, see the sections below.

---

## Overview

The Georgia Digitization Platform backend provides RESTful APIs for:

- **Document Management**: Upload, process, and manage PDF documents
- **AI-Powered Search**: Vector search and chat with documents using Gemini
- **User Management**: Authentication, authorization, and user profiles
- **File Management**: GCS file operations, transfers, and organization
- **Batch Processing**: Automated document processing workflows
- **System Administration**: Processor rules, categories, and system configuration

### Tech Stack

- **Framework**: Flask (Python 3.12)
- **Authentication**: JWT with Flask-JWT-Extended
- **Database**: Google Cloud Firestore
- **Storage**: Google Cloud Storage (GCS)
- **AI Services**: 
  - Google Vertex AI (Gemini models, Embeddings)
  - Document AI (OCR processing)
  - Vector Search
- **Queue**: Redis
- **SAML**: SSO authentication support

### API Versions

- **API v1**: `/backend/api/v1` - Standard endpoints
- **API v2**: `/backend/api/v2` - RBAC-enabled secure endpoints

---

## Prerequisites

Before setting up the backend, ensure you have:

1. **Python 3.12** or higher
2. **Google Cloud Platform (GCP) Account** with:
   - A GCP project with billing enabled
   - Firestore database created
   - GCS bucket created
   - Document AI processor configured
   - Vertex AI enabled
   - Service account with appropriate permissions
3. **Redis** (local or cloud instance)
4. **Google Cloud SDK** (for local development with ADC)
5. **Service Account Key** (JSON file) with permissions for:
   - Firestore
   - Cloud Storage
   - Document AI
   - Vertex AI
   - Pub/Sub (optional)

---

## Installation

### 1. Clone the Repository

```bash
cd georgia-digitization-platform/backend
```

### 2. Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

**Note**: Some dependencies (like `xmlsec` and `lxml`) may require system libraries. On Linux, you may need:

```bash
sudo apt-get install libxml2-dev libxslt1-dev libxmlsec1-dev libxmlsec1-openssl
```

### 4. Install Service Account Key

Place your GCP service account JSON key file in the backend directory:

```bash
# Example: Copy your service account key
cp /path/to/your-service-account-key.json ./shothik-project-2cc7a51b6844.json
```

---

## Configuration

### 1. Create Environment File

Create a `.env` file in the `backend` directory:

```bash
cp example.env .env
# Or create manually
touch .env
```

### 2. Configure Environment Variables

Edit `.env` with your configuration:

```bash
# ============================================
# Authentication
# ============================================
JWT_SECRET_KEY=your-super-secret-jwt-key-change-this-in-production
FLASK_SECRET_KEY=your-flask-secret-key-change-this-in-production

# ============================================
# GCP Project Settings
# ============================================
PROJECT_ID=your-gcp-project-id
LOCATION=us  # Document AI location (us, eu, etc.)
VERTEX_LOCATION=us-central1  # Vertex AI location
GOOGLE_APPLICATION_CREDENTIALS=./shothik-project-2cc7a51b6844.json  # Path to service account key

# ============================================
# Firestore Configuration
# ============================================
FIRESTORE_DATABASE_ID=(default)  # Or your specific database ID

# ============================================
# Cloud Storage Configuration
# ============================================
BUCKET_NAME=your-gcs-bucket-name
FILE_MANAGEMENT_BUCKET_NAME=your-gcs-bucket-name  # Can be same as BUCKET_NAME
GCS_SOURCE_ROOT=Pending Files/  # Root folder for source files (must end with /)
GCS_BULK_PROCESSING_PREFIX=  # Optional prefix for bulk processing

# ============================================
# Document AI Configuration
# ============================================
DOCUMENT_API_PROCESSOR_ID=your-document-ai-processor-id
DEFAULT_PARSER_PROCESSOR_ID=your-document-ai-processor-id  # Fallback processor

# ============================================
# Vertex AI Configuration
# ============================================
VECTOR_INDEX_NAME=your-vector-index-name
VECTOR_INDEX_ENDPOINT_ID=your-vector-endpoint-id
VECTOR_DEPLOYED_INDEX_ID=your-deployed-index-id
VECTOR_PRIVATE_ENDPOINT_IP=  # Optional: Private endpoint IP for VPC

# ============================================
# Redis Configuration
# ============================================
REDIS_HOST=localhost  # Or your Redis Cloud host
REDIS_PORT=6379
REDIS_DB=0
REDIS_USERNAME=  # Optional: Redis username
REDIS_PASSWORD=  # Optional: Redis password

# ============================================
# Pub/Sub Configuration (Optional)
# ============================================
PUBSUB_TOPIC_ID=test_topic
PUBSUB_SUBSCRIPTION_ID=test_topic-sub

# ============================================
# Document Processing Settings
# ============================================
MIN_PDF_PAGE_COUNT=1
MAX_PDF_PAGE_COUNT=2000
DEFAULT_PDF_PASSWORD=  # Optional: Default password for encrypted PDFs

# ============================================
# Retry Configuration
# ============================================
GCS_DOWNLOAD_MAX_RETRY=3
GCS_UPLOAD_MAX_RETRY=3
DOCAI_OCR_MAX_RETRY=5
FIRESTORE_SAVE_MAX_RETRY=3
FIRESTORE_UPDATE_MAX_RETRY=3
EMBEDDING_MAX_RETRY=5
VECTOR_UPSERT_MAX_RETRY=3

# ============================================
# Worker Settings
# ============================================
MAX_CONCURRENT_CHUNK_TASKS=5  # Max parallel chunks per document

# ============================================
# SAML Configuration (Optional)
# ============================================
SAML_SP_ENTITY_ID=https://your-domain.com/backend/api/v1/auth/metadata/
SAML_SP_ACS_URL=https://your-domain.com/backend/api/v1/auth/sso/acs
SAML_SP_SLO_URL=https://your-domain.com/backend/api/v1/auth/sso/slo
SAML_SP_X509CERT=  # Your SP certificate
SAML_SP_PRIVATE_KEY=  # Your SP private key
SAML_IDP_ENTITY_ID=  # Your IdP entity ID
SAML_IDP_SSO_URL=  # Your IdP SSO URL
SAML_IDP_SLO_URL=  # Your IdP SLO URL
SAML_IDP_X509CERT=  # Your IdP certificate
```

### 3. Verify Configuration

The application will validate required environment variables on startup. Missing essential variables will cause the application to fail with a clear error message.

---

## Creating Admin Users

Before you can use the API, you need to create at least one admin user account.

### Method 1: Using Service Account (Standard)

```bash
python create_admin.py
```

This will prompt you for:
- Email address
- Password (min 8 characters)
- Name (optional)

**Example:**
```bash
python create_admin.py --email admin@example.com --password securepass123 --name "Admin User"
```

**Note:** If you get permission errors, see [FIX_FIRESTORE_PERMISSIONS.md](./FIX_FIRESTORE_PERMISSIONS.md) or use Method 2 below.

### Method 2: Using Application Default Credentials (Local Development)

If your service account doesn't have proper permissions, use your personal gcloud credentials:

```bash
# 1. Authenticate with gcloud
gcloud auth application-default login

# 2. Set your project
gcloud config set project georgia-demo

# 3. Create admin user
python create_admin_adc.py
```

### Method 3: Direct Firestore Creation

If both methods fail, you can create the user directly in Firestore:

1. Go to [Firestore Console](https://console.cloud.google.com/firestore)
2. Navigate to the `users` collection
3. Create a document with ID = email (lowercase)
4. Add fields:
   - `email`: user email
   - `name`: display name
   - `role`: `"admin"`
   - `password`: hashed password (use Python to generate: `from werkzeug.security import generate_password_hash; print(generate_password_hash("your-password"))`)

### Getting Your Admin JWT Token

After creating an admin user, get your JWT token:

```bash
curl -X POST http://localhost:5001/backend/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "admin@example.com",
    "password": "your-password"
  }'
```

Save the `access_token` from the response for API requests.

---

## Running the Backend

### Local Development

#### Option 1: Using Python Directly

```bash
# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate  # Windows

# Set Google Application Credentials (if not in .env)
export GOOGLE_APPLICATION_CREDENTIALS=./shothik-project-2cc7a51b6844.json  # Linux/Mac
# OR
set GOOGLE_APPLICATION_CREDENTIALS=./shothik-project-2cc7a51b6844.json  # Windows

# Run the application
python run.py
```

The server will start on `http://localhost:5001` (port 5001 to avoid conflicts with AirPlay on macOS).

#### Option 2: Using Shell Script (Linux/Mac)

```bash
chmod +x run.sh
./run.sh
```

#### Option 3: Using Docker Compose

```bash
# Ensure .env file is configured
docker-compose up --build
```

This will also start a Redis container if not already running.

### Production Deployment

For production, use a WSGI server like Gunicorn:

```bash
gunicorn -w 4 -b 0.0.0.0:5001 --timeout 120 run:app
```

Or use the provided Dockerfile:

```bash
docker build -t georgia-backend .
docker run -p 5001:5001 --env-file .env georgia-backend
```

### Health Check

Once the server is running, verify it's working:

```bash
curl http://localhost:5001/health
```

Expected response:
```json
{"status": "ok"}
```

---

## Testing the APIs

### 1. Health Check

Test that the server is running:

```bash
curl http://localhost:5001/health
```

### 2. Authentication Flow

#### Login

```bash
curl -X POST http://localhost:5001/backend/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "admin@example.com",
    "password": "your-password"
  }'
```

**Response:**
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "user": {
    "id": "user-id",
    "name": "Admin User",
    "email": "admin@example.com",
    "role": "admin"
  }
}
```

Save the `access_token` for subsequent requests:

```bash
export TOKEN="your-access-token-here"
```

#### Get Current User

```bash
curl -X GET http://localhost:5001/backend/api/v1/auth/me \
  -H "Authorization: Bearer $TOKEN"
```

### 3. Document Management

#### Upload Document

```bash
curl -X POST http://localhost:5001/backend/api/v1/docs/upload \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@/path/to/your/document.pdf"
```

**Response:**
```json
{
  "results": [
    {
      "filename": "document.pdf",
      "status": "pending",
      "document_id": "doc-id-123",
      "message": "Document uploaded successfully"
    }
  ]
}
```

#### Get Document History

```bash
curl -X GET "http://localhost:5001/backend/api/v1/docs/history?limit=10" \
  -H "Authorization: Bearer $TOKEN"
```

#### Get Document Metadata

```bash
curl -X GET http://localhost:5001/backend/api/v1/docs/{document_id}/metadata \
  -H "Authorization: Bearer $TOKEN"
```

#### Chat with Documents

```bash
curl -X POST http://localhost:5001/backend/api/v1/docs/chat \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the main topics in these documents?",
    "session_id": "optional-session-id"
  }'
```

**Note**: Chat endpoint returns streaming responses. Use a tool that supports streaming (like Postman or a custom client).

### 4. User Management (Admin Only)

#### List All Users

```bash
curl -X GET http://localhost:5001/backend/api/v1/users/ \
  -H "Authorization: Bearer $ADMIN_TOKEN"
```

#### Create User

```bash
curl -X POST http://localhost:5001/backend/api/v1/users/ \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "New User",
    "email": "newuser@example.com",
    "password": "secure-password",
    "role": "user"
  }'
```

### 5. GCS File Management

#### List Source Directory

```bash
curl -X GET "http://localhost:5001/backend/api/v1/gcs/source-directory?path=&page=1&page_size=50" \
  -H "Authorization: Bearer $TOKEN"
```

#### List Root Directory

```bash
curl -X GET "http://localhost:5001/backend/api/v1/gcs/list-root?page=1&page_size=50" \
  -H "Authorization: Bearer $TOKEN"
```

#### List with Subfolders (Recursive)

```bash
curl -X GET "http://localhost:5001/backend/api/v1/gcs/list-with-subfolders?path=Georgia%2014/Pending%20Files/&page=1&page_size=50&search=document" \
  -H "Authorization: Bearer $TOKEN"
```

#### Search Files and Folders

```bash
curl -X GET "http://localhost:5001/backend/api/v1/gcs/search?search=document&type=both&path=Georgia%2014/Pending%20Files/&page=1&page_size=50" \
  -H "Authorization: Bearer $TOKEN"
```

#### Transfer File

```bash
curl -X POST http://localhost:5001/backend/api/v1/gcs/transfer \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "source": {"path": "Pending Files/source-file.pdf"},
    "destination": {"root": "Processed Files/", "path": ""},
    "operation": "move"
  }'
```

#### Transfer Folder

```bash
curl -X POST http://localhost:5001/backend/api/v1/gcs/transfer-folder \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "sourceFolderPath": "Georgia 14/Pending Files/Folder 1/",
    "destination": {"root": "Georgia 14/Processed Files/", "path": "2024/"},
    "operation": "move"
  }'
```

#### Get Transfer Status

```bash
curl -X GET http://localhost:5001/backend/api/v1/gcs/transfers/{transfer_id} \
  -H "Authorization: Bearer $TOKEN"
```

#### Get User Transfers (Paginated)

```bash
curl -X GET "http://localhost:5001/backend/api/v1/gcs/user-transfers?page=1&limit=50&status=succeeded&operation=move" \
  -H "Authorization: Bearer $TOKEN"
```

#### Rename File

```bash
curl -X PATCH http://localhost:5001/backend/api/v1/gcs/rename-file \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "filePath": "Georgia 14/Pending Files/document.pdf",
    "newFileName": "renamed-document.pdf"
  }'
```

#### Bulk Delete Files

```bash
curl -X DELETE http://localhost:5001/backend/api/v1/gcs/bulk-delete-files \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "filePaths": [
      "Georgia 14/Pending Files/document1.pdf",
      "Georgia 14/Pending Files/document2.pdf"
    ]
  }'
```

#### Download Folders as ZIP

```bash
curl -X POST http://localhost:5001/backend/api/v1/gcs/download-folders \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "folderPaths": [
      "Georgia 14/Pending Files/Folder 1/",
      "Georgia 14/Pending Files/Folder 2/"
    ]
  }' \
  --output folders.zip
```

### 6. Batch Processing (Admin Only)

#### Get Available Buckets

```bash
curl -X GET http://localhost:5001/backend/api/v1/batch/buckets \
  -H "Authorization: Bearer $ADMIN_TOKEN"
```

#### Start Batch Processing

```bash
curl -X POST http://localhost:5001/backend/api/v1/batch/start \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "bucket_name": "your-bucket-name"
  }'
```

### 7. System Routes

#### Get Document Categories (v2)

```bash
curl -X GET http://localhost:5001/backend/api/v2/system/categories \
  -H "Authorization: Bearer $TOKEN"
```

### 8. Logs

#### Get Processing Logs

```bash
curl -X GET "http://localhost:5001/backend/api/v1/logs?limit=50&level=ERROR" \
  -H "Authorization: Bearer $TOKEN"
```

#### Export Logs as CSV

```bash
curl -X GET "http://localhost:5001/backend/api/v1/logs/export?level=ERROR" \
  -H "Authorization: Bearer $TOKEN" \
  -o logs.csv
```

### Using Postman

A Postman collection is available at `Postman_Collection_Georgia_Digitization_Platform.json`. Import this into Postman for a complete set of pre-configured API requests.

1. Open Postman
2. Click **Import**
3. Select `Postman_Collection_Georgia_Digitization_Platform.json`
4. Configure the collection variable `base_url` to `http://localhost:5001`
5. Start with the **Login** request to get a token
6. Use the token in subsequent requests

### Testing with Python

Example test script:

```python
import requests
import json

BASE_URL = "http://localhost:5001"

# 1. Login
login_response = requests.post(
    f"{BASE_URL}/backend/api/v1/auth/login",
    json={"email": "admin@example.com", "password": "password"}
)
token = login_response.json()["access_token"]
headers = {"Authorization": f"Bearer {token}"}

# 2. Get current user
user_response = requests.get(
    f"{BASE_URL}/backend/api/v1/auth/me",
    headers=headers
)
print("Current user:", user_response.json())

# 3. Get document history
history_response = requests.get(
    f"{BASE_URL}/backend/api/v1/docs/history?limit=10",
    headers=headers
)
print("Document history:", history_response.json())
```

---

## API Documentation

For complete API documentation, see [API_DOCUMENTATION.md](./API_DOCUMENTATION.md).

### Quick Reference

| Endpoint Category | Base Path |
|------------------|-----------|
| Authentication | `/backend/api/v1/auth` |
| Documents | `/backend/api/v1/docs` |
| Users | `/backend/api/v1/users` |
| GCS Management | `/backend/api/v1/gcs` |
| Batch Processing | `/backend/api/v1/batch` |
| Processor Rules | `/backend/api/v1/processor-rules` |
| Logs | `/backend/api/v1/logs` |
| System (v2) | `/backend/api/v2/system` |
| Secure Documents (v2) | `/backend/api/v2/docs` |

### Authentication

Most endpoints require JWT authentication. Include the token in the `Authorization` header:

```
Authorization: Bearer <your-jwt-token>
```

### Error Responses

The API returns standard HTTP status codes:

- `200` - Success
- `201` - Created
- `400` - Bad Request
- `401` - Unauthorized
- `403` - Forbidden
- `404` - Not Found
- `500` - Internal Server Error

Error response format:
```json
{
  "msg": "Error description",
  "error": "Detailed error message",
  "error_type": "ErrorType"
}
```

---

## Development

### Project Structure

```
backend/
├── app/
│   ├── __init__.py          # Flask app factory
│   ├── config.py            # Configuration management
│   ├── models/              # Data models
│   ├── routes/              # API route blueprints
│   ├── services/            # Business logic
│   └── utils/               # Utility functions
├── tests/                   # Test files
├── run.py                   # Application entry point
├── requirements.txt         # Python dependencies
├── Dockerfile               # Docker configuration
├── docker-compose.yml       # Docker Compose setup
└── README.md                # This file
```

### Running Tests

Currently, test files are scaffolded. To run tests:

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_routes.py

# Run with verbose output
python -m pytest tests/ -v
```

### Code Style

Follow PEP 8 Python style guidelines. Consider using:

- **Black** for code formatting
- **Flake8** or **Pylint** for linting
- **mypy** for type checking

### Adding New Routes

1. Create a new blueprint in `app/routes/`
2. Define routes in the blueprint
3. Register the blueprint in `app/__init__.py`

Example:

```python
# app/routes/my_routes.py
from flask import Blueprint

my_bp = Blueprint('my', __name__)

@my_bp.route('/test')
def test():
    return {"message": "Test endpoint"}
```

```python
# app/__init__.py
from .routes.my_routes import my_bp
app.register_blueprint(my_bp, url_prefix=f'{API_PREFIX}/my')
```

---

## Deployment

### Docker Deployment

#### Build Image

```bash
docker build -t georgia-backend:latest .
```

#### Run Container

```bash
docker run -d \
  -p 5001:5001 \
  --env-file .env \
  --name georgia-backend \
  georgia-backend:latest
```

### Google Cloud Run

1. Build and push to Container Registry:

```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/georgia-backend
```

2. Deploy to Cloud Run:

```bash
gcloud run deploy georgia-backend \
  --image gcr.io/PROJECT_ID/georgia-backend \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars-from-file .env
```

### Kubernetes

Kubernetes deployment files are available in the `k8s/` directory:

```bash
kubectl apply -f k8s/backend-depl.yaml
```

### Environment Variables in Production

**Important**: Never commit `.env` files or service account keys to version control. Use:

- **Google Cloud Secret Manager** for sensitive values
- **Kubernetes Secrets** for K8s deployments
- **Environment variables** set in your deployment platform

---

## Troubleshooting

### Common Issues

#### 1. Import Errors

**Problem**: `ModuleNotFoundError when importing`

**Solution**: Ensure virtual environment is activated and dependencies are installed:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

#### 2. Firestore Connection Errors

**Problem**: `DefaultCredentialsError` or `403 Missing or insufficient permissions`

**Solutions**:
- **Permission Denied**: Your service account lacks Firestore permissions
  - See [FIX_FIRESTORE_PERMISSIONS.md](./FIX_FIRESTORE_PERMISSIONS.md) for detailed instructions
  - Quick fix: Grant `Cloud Datastore User` role to your service account
  - Alternative: Use `create_admin_adc.py` with Application Default Credentials
- **Credentials Not Found**: 
  - Verify `GOOGLE_APPLICATION_CREDENTIALS` points to valid service account key
  - Check the file path in your `.env` file
- **For Local Development**:
  - Run `gcloud auth application-default login`
  - Temporarily comment out `GOOGLE_APPLICATION_CREDENTIALS` in `.env`

#### 3. Redis Connection Errors

**Problem**: `ConnectionError` when connecting to Redis

**Solutions**:
- Verify Redis is running: `redis-cli ping`
- Check `REDIS_HOST` and `REDIS_PORT` in `.env`
- For Redis Cloud, verify `REDIS_PASSWORD` is set

#### 4. Port Already in Use

**Problem**: `Address already in use` on port 5001

**Solution**: Change port in `run.py` or kill the process using the port:
```bash
# Find process
lsof -i :5001  # Mac/Linux
netstat -ano | findstr :5001  # Windows

# Kill process
kill -9 <PID>  # Mac/Linux
taskkill /PID <PID> /F  # Windows
```

#### 5. Vector Search Connection Issues

**Problem**: Vector search endpoints fail

**Solutions**:
- Verify `VECTOR_INDEX_NAME`, `VECTOR_INDEX_ENDPOINT_ID`, and `VECTOR_DEPLOYED_INDEX_ID` are set
- Check Vector Search index is deployed in Vertex AI
- For private endpoints, verify `VECTOR_PRIVATE_ENDPOINT_IP` is configured

#### 6. SAML Authentication Not Working

**Problem**: SSO login fails

**Solutions**:
- Verify all SAML environment variables are set correctly
- Check `saml/settings.json` is generated (created automatically on startup)
- Verify certificates and keys are properly formatted (no newlines in env vars)

### Debug Mode

Enable debug logging by setting in `run.py`:

```python
app.run(host='0.0.0.0', port=5001, debug=True)
```

**Warning**: Never use `debug=True` in production!

### Viewing Logs

Application logs are printed to stdout. For Docker:

```bash
docker logs georgia-backend
docker logs -f georgia-backend  # Follow logs
```

### Testing Individual Components

Test specific services:

```bash
# Test Document AI
python test-document-ai.py

# Test Vector Search
python test-vectorization.py

# Test Gemini Classification
python test_gemini_gcs_classification.py
```

### Common Permission Issues

If you encounter permission errors:

1. **Firestore Permissions**: See [FIX_FIRESTORE_PERMISSIONS.md](./FIX_FIRESTORE_PERMISSIONS.md)
2. **Service Account Setup**: Ensure your service account has all required roles:
   - Cloud Datastore User (Firestore)
   - Storage Object Admin (GCS)
   - Document AI API User
   - Vertex AI User
3. **Project Mismatch**: Verify `PROJECT_ID` in `.env` matches your service account's project

---

## Additional Resources

### Documentation

- [API Documentation](./API_DOCUMENTATION.md) - Complete API reference with all endpoints
- [FILE_MANAGEMENT_API_DOCUMENTATION.md](../../FILE_MANAGEMENT_API_DOCUMENTATION.md) - Detailed file management API documentation
- [USER_TRANSFERS_API_DOCUMENTATION.md](../../USER_TRANSFERS_API_DOCUMENTATION.md) - User transfers API with pagination and filtering
- [FIX_FIRESTORE_PERMISSIONS.md](./FIX_FIRESTORE_PERMISSIONS.md) - Guide to fix Firestore permission issues
- [Postman Collection](./Postman_Collection_Georgia_Digitization_Platform.json) - Import into Postman for API testing (v2.2.0)

### Scripts

- `create_admin.py` - Create admin user using service account
- `create_admin_adc.py` - Create admin user using Application Default Credentials
- `run.py` - Main application entry point
- `worker.py` - Background worker for document processing

### External Links

- [Docker Documentation](https://docs.docker.com/) - Docker setup guide
- [Google Cloud Documentation](https://cloud.google.com/docs) - GCP services
- [Flask Documentation](https://flask.palletsprojects.com/) - Flask framework docs
- [Firestore Documentation](https://cloud.google.com/firestore/docs) - Firestore database docs

---

## Support

For issues, questions, or contributions:

1. Check the [Troubleshooting](#troubleshooting) section
2. Review [FIX_FIRESTORE_PERMISSIONS.md](./FIX_FIRESTORE_PERMISSIONS.md) for permission issues
3. Contact the development team
4. Create an issue in the project repository

---

## License

[Add your license information here]

---

**Last Updated**: January 2025

**Recent Updates**:
- Added new GCS file management endpoints: Rename File, Bulk Delete Files/Folders, Search, Download Folders as ZIP
- Enhanced transfer operations with folder transfer support and progress tracking
- Added paginated user transfers endpoint with filtering capabilities
- Updated API documentation and Postman collection (v2.2.0)

