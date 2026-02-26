# Georgia Digitization Platform - Backend API Documentation

## Table of Contents
1. [Overview](#overview)
2. [Authentication](#authentication)
3. [API Endpoints](#api-endpoints)
   - [Authentication Routes](#authentication-routes)
   - [Document Management Routes](#document-management-routes)
   - [User Management Routes](#user-management-routes)
   - [System Routes](#system-routes)
   - [Log Routes](#log-routes)
   - [Activity Log Routes](#activity-log-routes)
   - [GCS Management Routes](#gcs-management-routes)
   - [Processor Rules Routes](#processor-rules-routes)
   - [Batch Processing Routes](#batch-processing-routes)
   - [Secure Document Routes](#secure-document-routes)
4. [Error Handling](#error-handling)
5. [Rate Limiting](#rate-limiting)
6. [Examples](#examples)

---

## Overview

The Georgia Digitization Platform backend provides a comprehensive API for document processing, management, and AI-powered search capabilities. The API is built using Flask and follows RESTful principles.

### Base URLs
- **API v1**: `https://your-domain.com/backend/api/v1`
- **API v2**: `https://your-domain.com/backend/api/v2` (RBAC-enabled)

### Tech Stack
- **Framework**: Flask (Python)
- **Authentication**: JWT with Flask-JWT-Extended
- **Database**: Google Cloud Firestore
- **Storage**: Google Cloud Storage (GCS)
- **AI Services**: Google Vertex AI, Document AI, Gemini
- **Queue**: Redis
- **Frontend**: Next.js 15 with TypeScript

---

## Authentication

All endpoints except health check and authentication endpoints require a valid JWT token.

### Obtaining a Token

#### Login
```http
POST /backend/api/v1/auth/login
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "your-password"
}
```

**Response:**
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "user": {
    "id": "user-id",
    "name": "John Doe",
    "email": "user@example.com",
    "role": "user"
  }
}
```

#### SSO Login
```http
GET /backend/api/v1/auth/sso/login
```

Redirects to SSO provider for authentication.

### Using the Token
Include the JWT token in the Authorization header for all authenticated requests:

```http
Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...
```

---

## API Endpoints

### Authentication Routes

#### Health Check
```http
GET /health
```
**Response:** `{"status": "ok"}`

#### Login
```http
POST /backend/api/v1/auth/login
```
**Required:** Email and password
**Returns:** JWT token and user information

#### SSO Login
```http
GET /backend/api/v1/auth/sso/login
```
**Returns:** Redirect to SSO provider

#### SSO Assertion Consumer Service
```http
POST /backend/api/v1/auth/sso/acs
```
**Required:** SAML response from IdP
**Returns:** JWT token and redirect URL

#### Get Current User
```http
GET /backend/api/v1/auth/me
```
**Authentication:** Required
**Returns:** Current user information

#### Update Profile
```http
PATCH /backend/api/v1/auth/profile
```
**Authentication:** Required
**Required:** Name in request body

#### Change Password
```http
PATCH /backend/api/v1/auth/change-password
```
**Authentication:** Required
**Required:** Current password and new password

#### Logout
```http
POST /backend/api/v1/auth/logout
```
**Authentication:** Required
**Returns:** Success message

#### Test Ping
```http
GET /backend/api/v1/auth/test/ping
```
**Authentication:** Not required
**Returns:** Test response confirming auth routes are working

#### Test SAML Imports
```http
GET /backend/api/v1/auth/test/saml-imports
```
**Authentication:** Not required
**Returns:** Status of SAML library imports

#### Test SAML Settings
```http
GET /backend/api/v1/auth/test/saml-settings
```
**Authentication:** Not required
**Returns:** Status of SAML settings loading

---

### Document Management Routes

#### Upload Documents
```http
POST /backend/api/v1/docs/upload
```
**Authentication:** Required
**Required:** Multipart form with files
**Supported:** PDF files only
**Returns:** Upload results and status

#### Chat with Documents
```http
POST /backend/api/v1/docs/chat
```
**Authentication:** Required
**Required:** Query in request body
**Returns:** Streaming AI response with citations

#### Get Document History
```http
GET /backend/api/v1/docs/history?limit=10&start_after=&search=&status=
```
**Authentication:** Required
**Parameters:**
- `limit`: Number of records (1-100)
- `start_after`: Cursor for pagination
- `search`: Search term for filtering
- `status`: Filter by status (Pending, Processing, Failed, Completed)

#### Get Chat Sessions
```http
GET /backend/api/v1/docs/chat/sessions?limit=10
```
**Authentication:** Required
**Returns:** List of chat sessions for the user

#### Get Session Messages
```http
GET /backend/api/v1/docs/chat/sessions/{session_id}/messages
```
**Authentication:** Required
**Returns:** All messages in a specific session

#### Download Document
```http
GET /backend/api/v1/docs/download/{doc_id}
```
**Authentication:** Required
**Returns:** Signed URL for downloading original document

#### Get Document Metadata
```http
GET /backend/api/v1/docs/{doc_id}/metadata
```
**Authentication:** Required
**Returns:** Document metadata and download URL

#### Get Document Chunks
```http
GET /backend/api/v1/docs/{doc_id}/chunks?limit=20&start_after=
```
**Authentication:** Required
**Returns:** Paginated chunks for the document

#### Get Document Logs
```http
GET /backend/api/v1/docs/{doc_id}/logs?limit=50&start_after=&level=&step=
```
**Authentication:** Required
**Returns:** Processing logs for the document

#### Reprocess Document
```http
POST /backend/api/v1/docs/{doc_id}/reprocess
```
**Authentication:** Required
**Returns:** Success message if document is in error state

#### Force Reprocess Document
```http
POST /backend/api/v1/docs/{doc_id}/force-reprocess
```
**Authentication:** Required
**Returns:** Success message (deletes existing chunks and reprocesses)

#### Download Chunk
```http
GET /backend/api/v1/docs/chunks/{chunk_id}/download
```
**Authentication:** Required
**Returns:** Signed URL for downloading specific chunk

#### Rename Chat Session
```http
PATCH /backend/api/v1/docs/chat/sessions/{session_id}
```
**Authentication:** Required
**Required:** Title in request body

#### Delete Chat Session
```http
DELETE /backend/api/v1/docs/chat/sessions/{session_id}
```
**Authentication:** Required
**Returns:** Success message

#### Get Session Messages
```http
GET /backend/api/v1/docs/chat/sessions/{session_id}/messages
```
**Authentication:** Required
**Returns:** All messages in a specific chat session

#### Rename Chat Session
```http
PATCH /backend/api/v1/docs/chat/sessions/{session_id}
```
**Authentication:** Required
**Required:** Title in request body
**Returns:** Success message

#### Delete Chat Session
```http
DELETE /backend/api/v1/docs/chat/sessions/{session_id}
```
**Authentication:** Required
**Returns:** Success message

#### Vector Search Status
```http
GET /backend/api/v1/docs/status/vector-search
```
**Returns:** Status of vector search connection

#### Backfill Search Keywords
```http
POST /backend/api/v1/docs/backfill_search_keywords
```
**Authentication:** Required
**Returns:** Success message after backfilling search keywords for all documents

---

### User Management Routes

#### List Users
```http
GET /backend/api/v1/users/
```
**Authentication:** Required (Admin only)
**Returns:** List of all users

#### Create User
```http
POST /backend/api/v1/users/
```
**Authentication:** Required (Admin only)
**Required:** Name, email, password, role in request body

#### Get User by ID
```http
GET /backend/api/v1/users/{user_id}
```
**Authentication:** Required (Admin only)
**Returns:** User information

#### Update User
```http
PUT /backend/api/v1/users/{user_id}
```
**Authentication:** Required (Admin only)
**Required:** Fields to update in request body

#### Delete User
```http
DELETE /backend/api/v1/users/{user_id}
```
**Authentication:** Required (Admin only)
**Returns:** Success message

---

### System Routes

#### System Diagnosis
```http
GET /backend/api/v1/system/diagnosis
```
**Authentication:** Required
**Returns:** Comprehensive system health check including configuration, GCS, Firestore, Document AI, Vector Search, and Gemini status

#### Create Firestore Indexes
```http
POST /backend/api/v1/system/create-indexes
```
**Authentication:** Required
**Returns:** Status of Firestore composite index creation operations

#### Setup Vector Search
```http
POST /backend/api/v1/system/setup-vector-search
```
**Authentication:** Required
**Returns:** Status of Vector Search index and endpoint creation/setup

#### Setup Firestore Database
```http
POST /backend/api/v1/system/setup-firestore-database
```
**Authentication:** Required
**Required:** `locationId` in request body (e.g., "us-central1")
**Returns:** Status of Firestore database creation operation

#### Start Bulk Process
```http
POST /backend/api/v1/system/start-bulk-process
```
**Authentication:** Required
**Optional:** `gcs_prefix` in request body
**Returns:** Success message with prefix used for bulk processing

#### Get Bulk Process Stats
```http
GET /backend/api/v1/system/bulk-process-stats
```
**Authentication:** Optional
**Returns:** Statistics for the latest or currently running bulk processing run

#### Get Processing Dashboard Stats
```http
GET /backend/api/v1/system/processing-dashboard-stats
```
**Authentication:** Optional
**Returns:** Aggregated counts of documents by processing status and total chunk count

#### Process Pending Documents
```http
POST /backend/api/v1/system/process-pending
```
**Authentication:** Required
**Returns:** Count of documents enqueued for processing

#### Reset Stuck Documents
```http
POST /backend/api/v1/system/reset-stuck-documents
```
**Authentication:** Required
**Returns:** Count of documents reset to pending state

#### Clear Processing Data
```http
DELETE /backend/api/v1/system/clear-processing-data
```
**Authentication:** Required
**Warning:** Destructive operation - deletes all processing data from Firestore and Vector Search
**Returns:** Details of deleted collections and vector datapoints

#### Debug Routes
```http
GET /debug/routes
```
**Authentication:** Not required
**Returns:** HTML page listing all available API routes

#### Get Document Categories (V2)
```http
GET /backend/api/v2/system/categories
```
**Authentication:** Required
**Returns:** List of all document categories

---

### Log Routes

#### Get Processing Logs
```http
GET /backend/api/v1/logs?limit=50&start_after=&level=&step=&original_filename=&document_id=&worker_id=
```
**Authentication:** Optional
**Returns:** Paginated processing logs with filters

#### Export Logs as CSV
```http
GET /backend/api/v1/logs/export?level=&step=&document_id=&worker_id=
```
**Authentication:** Optional
**Returns:** CSV file download

#### Get System Logs for Message
```http
GET /backend/api/v1/logs/sessions/{session_id}/messages/{message_id}/system_logs
```
**Authentication:** Required
**Returns:** System logs for a specific chat message

---

### Activity Log Routes

#### Get Activity History
```http
GET /backend/api/v1/activity-logs/history?limit=50&start_after=&user_email=&activity_type=&start_date=&end_date=&success=
```
**Authentication:** Required
**Returns:** Paginated activity logs with optional filtering
**Authorization:** Regular users see only their own logs; admins can view all users' logs

#### Get Activity Types
```http
GET /backend/api/v1/activity-logs/activity-types
```
**Authentication:** Required
**Returns:** List of all available activity types organized by category

**Note:** For detailed documentation, see [ACTIVITY_LOG_API_DOCUMENTATION.md](../../ACTIVITY_LOG_API_DOCUMENTATION.md)

---

### GCS Management Routes

#### Get GCS Configuration
```http
GET /backend/api/v1/gcs/config
```
**Authentication:** Required
**Returns:** GCS bucket and source root configuration

#### List Source Directory
```http
GET /backend/api/v1/gcs/source-directory?path=&page=1&page_size=50
```
**Authentication:** Required
**Returns:** Contents of source directory with pagination

#### List Destinations
```http
GET /backend/api/v1/gcs/destinations?path=&page=1&page_size=50
```
**Authentication:** Required
**Returns:** Destination folders with pagination

#### Transfer Files
```http
POST /backend/api/v1/gcs/transfer
```
**Authentication:** Required
**Required:** Source and destination paths, operation type
**Returns:** Transfer status and ID

#### Get Transfer Status
```http
GET /backend/api/v1/gcs/transfers/{transfer_id}
```
**Authentication:** Required
**Returns:** Current transfer status

#### Get User Transfers
```http
GET /backend/api/v1/gcs/transfers
```
**Authentication:** Required
**Returns:** List of user's recent transfers

#### Transfer Folder
```http
POST /backend/api/v1/gcs/transfer-folder
```
**Authentication:** Required
**Required:** Source folder path, destination, operation
**Returns:** Transfer ID and total files

#### Get Transfer Progress
```http
GET /backend/api/v1/gcs/progress/{transfer_id}
```
**Authentication:** Required
**Returns:** Transfer progress information

#### Get All Progress
```http
GET /backend/api/v1/gcs/progress
```
**Authentication:** Required
**Returns:** Progress of all active transfers

#### Create Folder
```http
POST /backend/api/v1/gcs/create-folder
```
**Authentication:** Required
**Required:** Folder name and current path
**Returns:** Success message

#### Upload Files
```http
POST /backend/api/v1/gcs/upload-files?path=
```
**Authentication:** Required
**Required:** Files in multipart form, path parameter
**Returns:** Upload results and any errors

#### Get File Preview
```http
GET /backend/api/v1/gcs/preview/{file_path}
```
**Authentication:** Required
**Returns:** Preview URL for file viewing

#### List Destinations
```http
GET /backend/api/v1/gcs/destinations?path=&page=1&page_size=50
```
**Authentication:** Required
**Returns:** Destination folders with pagination (excludes source root)

#### List Destination Directory
```http
GET /backend/api/v1/gcs/destination-directory?root=&path=
```
**Authentication:** Required
**Required:** Root parameter
**Returns:** Contents of a specific destination directory

#### Get User Transfers
```http
GET /backend/api/v1/gcs/transfers
```
**Authentication:** Required
**Returns:** List of user's recent transfers

#### Transfer Folder
```http
POST /backend/api/v1/gcs/transfer-folder
```
**Authentication:** Required
**Required:** Source folder path, destination root, operation type
**Returns:** Transfer ID and total files count

#### Get Transfer Progress
```http
GET /backend/api/v1/gcs/progress/{transfer_id}
```
**Authentication:** Required
**Returns:** Progress information for a specific transfer

#### Get All Progress
```http
GET /backend/api/v1/gcs/progress
```
**Authentication:** Required
**Returns:** Progress of all active transfers

#### Get User Transfers (Paginated)
```http
GET /backend/api/v1/gcs/user-transfers?page=1&limit=50&status=&operation=&transfer_type=&date=&start_date=&end_date=
```
**Authentication:** Required
**Returns:** Paginated list of user transfers with filtering support
**Note:** See USER_TRANSFERS_API_DOCUMENTATION.md for detailed documentation

#### Upload Files
```http
POST /backend/api/v1/gcs/upload-files?path=
```
**Authentication:** Required
**Required:** Files in multipart form, path parameter (must be within source root)
**Returns:** Upload results with any errors

#### Upload Files to Path
```http
POST /backend/api/v1/gcs/upload-files-to-path?path=
```
**Authentication:** Required
**Required:** Files in multipart form
**Returns:** Upload results with Firestore metadata

#### Get File Preview
```http
GET /backend/api/v1/gcs/preview/{file_path}
```
**Authentication:** Required
**Returns:** Preview URL for file viewing in iframe

#### Proxy File Content
```http
GET /backend/api/v1/gcs/proxy/{file_path}?token=
```
**Required:** JWT token parameter
**Returns:** File content with proper headers for iframe embedding

#### Rename File
```http
PATCH /backend/api/v1/gcs/rename-file
```
**Authentication:** Required
**Required:** filePath and newFileName in request body
**Returns:** Success message with old and new paths

#### Delete File
```http
DELETE /backend/api/v1/gcs/delete-file
```
**Authentication:** Required
**Required:** filePath in request body
**Returns:** Success message

#### Bulk Delete Files
```http
DELETE /backend/api/v1/gcs/bulk-delete-files
```
**Authentication:** Required
**Required:** filePaths array in request body
**Returns:** Bulk delete results with success/failure details

#### Bulk Delete Folders
```http
DELETE /backend/api/v1/gcs/bulk-delete-folders
```
**Authentication:** Required
**Required:** folderPaths array in request body
**Returns:** Bulk delete results with success/failure details

#### List Root Directory
```http
GET /backend/api/v1/gcs/list-root?page=1&page_size=50
```
**Authentication:** Required
**Returns:** All files and folders in root directory with pagination

#### List Path Directory
```http
GET /backend/api/v1/gcs/list-path?path=&page=1&page_size=50
```
**Authentication:** Required
**Required:** path parameter
**Returns:** Files and folders in specific path with pagination

#### List Destinations with Subfolders
```http
GET /backend/api/v1/gcs/destinations-with-subfolders?path=&page=1&page_size=50&search=
```
**Authentication:** Required
**Returns:** Destination folders and files in tree structure (recursive) with optional search

#### List Destination Path
```http
GET /backend/api/v1/gcs/list-destination-path?path=&page=1&page_size=50
```
**Authentication:** Required
**Returns:** Files and folders in destination path with pagination

#### List with Subfolders
```http
GET /backend/api/v1/gcs/list-with-subfolders?path=&page=1&page_size=50&search=
```
**Authentication:** Required
**Returns:** Files and folders in path with all subfolders recursively, with optional search

#### Search Files and Folders
```http
GET /backend/api/v1/gcs/search?search=&type=both&path=&page=1&page_size=50
```
**Authentication:** Required
**Required:** search parameter
**Returns:** Search results for files and/or folders matching search term

#### Download Folders as ZIP
```http
POST /backend/api/v1/gcs/download-folders
```
**Authentication:** Required
**Required:** folderPaths array in request body
**Returns:** ZIP file stream with all files from specified folders

#### Get File Metadata
```http
GET /backend/api/v1/gcs/file/{file_path}/metadata
```
**Authentication:** Required
**Returns:** File metadata from Firestore

#### Get Folder Metadata
```http
GET /backend/api/v1/gcs/folder/{folder_path}/metadata
```
**Authentication:** Required
**Returns:** Folder metadata from Firestore

#### Get Files in Folder
```http
GET /backend/api/v1/gcs/folder/{folder_path}/files
```
**Authentication:** Required
**Returns:** All files in a specific folder from Firestore

---

### Processor Rules Routes

#### Create Processor Rule
```http
POST /backend/api/v1/processor-rules
```
**Authentication:** Required (Admin only)
**Required:** Rule configuration in request body
**Returns:** Rule ID and success message

#### Get All Processor Rules
```http
GET /backend/api/v1/processor-rules?enabled_only=true&order_by_priority=true
```
**Authentication:** Required (Admin only)
**Returns:** List of processor rules

#### Get Processor Rule by ID
```http
GET /backend/api/v1/processor-rules/{rule_id}
```
**Authentication:** Required (Admin only)
**Returns:** Specific rule information

#### Update Processor Rule
```http
PUT /backend/api/v1/processor-rules/{rule_id}
```
**Authentication:** Required (Admin only)
**Required:** Updated rule data in request body
**Returns:** Success message

#### Delete Processor Rule
```http
DELETE /backend/api/v1/processor-rules/{rule_id}
```
**Authentication:** Required (Admin only)
**Returns:** Success message

---

### Batch Processing Routes

#### Get GCS Buckets
```http
GET /backend/api/v1/batch/buckets
```
**Authentication:** Required (Admin only)
**Returns:** List of available GCS buckets

#### Start Batch Processing
```http
POST /backend/api/v1/batch/start
```
**Authentication:** Required (Admin only)
**Required:** Bucket name in request body
**Returns:** Run ID and success message

#### Get Batch Reports
```http
GET /backend/api/v1/batch/reports
```
**Authentication:** Required (Admin only)
**Returns:** List of batch processing runs

#### Get Run Files
```http
GET /backend/api/v1/batch/reports/{run_id}/files?limit=100&start_after=
```
**Authentication:** Required (Admin only)
**Returns:** Paginated list of files for a specific batch run

#### Trigger Batch Processing
```http
POST /backend/api/v1/batch/trigger
```
**Authentication:** Required (Admin only)
**Returns:** Success message

#### Set Default Configuration
```http
POST /backend/api/v1/batch/config
```
**Authentication:** Required (Admin only)
**Required:** Bucket name in request body
**Returns:** Success message

#### Get Default Configuration
```http
GET /backend/api/v1/batch/config
```
**Authentication:** Required (Admin only)
**Returns:** Current default bucket configuration

#### Start Categorization Batch
```http
POST /backend/api/v1/batch/system/start-categorization-batch
```
**Authentication:** Required (Admin only)
**Returns:** Success message and batch status

#### Add Ignored Folder
```http
POST /backend/api/v1/batch/ignored-folders
```
**Authentication:** Required (Admin only)
**Required:** Folder name in request body
**Returns:** Success message

#### Get Ignored Folders
```http
GET /backend/api/v1/batch/ignored-folders
```
**Authentication:** Required (Admin only)
**Returns:** List of ignored folders

#### Get System Logs for Message
```http
GET /backend/api/v1/logs/sessions/{session_id}/messages/{message_id}/system_logs
```
**Authentication:** Required
**Returns:** System logs for a specific chat message

---

### Secure Document Routes (API v2)

#### Get Secure History
```http
GET /backend/api/v2/docs/history?limit=10&start_after=&search=&status=
```
**Authentication:** Required
**Returns:** Document history filtered by user's accessible categories

#### Secure Chat with Documents
```http
POST /backend/api/v2/docs/chat
```
**Authentication:** Required
**Required:** Query in request body
**Returns:** Streaming AI response with RBAC filtering

#### Get Secure Document Metadata
```http
GET /backend/api/v2/docs/{doc_id}/metadata
```
**Authentication:** Required
**Returns:** Document metadata if user has access

---

## Error Handling

The API returns standard HTTP status codes and error responses:

### Common Status Codes
- `200` - Success
- `201` - Created
- `202` - Accepted (background processing)
- `400` - Bad Request
- `401` - Unauthorized
- `403` - Forbidden
- `404` - Not Found
- `409` - Conflict
- `500` - Internal Server Error
- `503` - Service Unavailable

### Error Response Format
```json
{
  "msg": "Error description",
  "error": "Detailed error message",
  "error_type": "ErrorType"
}
```

### Common Errors

#### Authentication Errors
- **401 Unauthorized**: Missing or invalid JWT token
- **403 Forbidden**: Insufficient permissions for the resource

#### Validation Errors
- **400 Bad Request**: Missing required fields or invalid data format
- **409 Conflict**: Resource already exists or operation conflicts

#### Processing Errors
- **503 Service Unavailable**: External service (GCS, Firestore, etc.) unavailable
- **500 Internal Server Error**: Unexpected server error

---

## Rate Limiting

The API implements rate limiting to ensure fair usage:

- **Authentication endpoints**: 10 requests per minute per IP
- **Document upload**: 5 files per minute per user
- **Chat endpoints**: 60 requests per minute per user
- **General endpoints**: 1000 requests per hour per user

Rate limit headers are included in responses:
- `X-RateLimit-Limit`: Request limit per window
- `X-RateLimit-Remaining`: Remaining requests in current window
- `X-RateLimit-Reset`: Time when the rate limit resets

---

## Examples

### 1. User Authentication Flow
```bash
# Login
curl -X POST http://localhost:5001/backend/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"admin@example.com","password":"password123"}'

# Use returned token for subsequent requests
curl -X GET http://localhost:5001/backend/api/v1/docs/history \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

### 2. Document Upload and Chat
```bash
# Upload a PDF document
curl -X POST http://localhost:5001/backend/api/v1/docs/upload \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -F "file=@document.pdf"

# Chat with the uploaded documents
curl -X POST http://localhost:5001/backend/api/v1/docs/chat \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query":"What are the main topics discussed in these documents?"}'
```

### 3. Admin Operations
```bash
# List all users (admin only)
curl -X GET http://localhost:5001/backend/api/v1/users/ \
  -H "Authorization: Bearer ADMIN_JWT_TOKEN"

# Create a processor rule (admin only)
curl -X POST http://localhost:5001/backend/api/v1/processor-rules \
  -H "Authorization: Bearer ADMIN_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "ruleName": "PDF Invoice Processor",
    "documentTypeLabel": "INVOICE",
    "targetParserProcessorId": "your-processor-id",
    "isEnabled": true,
    "priority": 1
  }'
```

### 4. File Management
```bash
# List source directory
curl -X GET http://localhost:5001/backend/api/v1/gcs/source-directory \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"

# Transfer a file
curl -X POST http://localhost:5001/backend/api/v1/gcs/transfer \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "source": {"path": "source/file.pdf"},
    "destination": {"root": "processed/", "path": ""},
    "operation": "move"
  }'
```

---

## Environment Variables

Key environment variables required for the API:

```bash
# Authentication
JWT_SECRET_KEY=your-jwt-secret
FLASK_SECRET_KEY=your-flask-secret

# GCP Configuration
PROJECT_ID=your-project-id
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
FIRESTORE_DATABASE_ID=(default)

# Storage
BUCKET_NAME=your-bucket-name
FILE_MANAGEMENT_BUCKET_NAME=your-bucket-name
GCS_SOURCE_ROOT=Pending Files/

# AI Services
DOCUMENT_API_PROCESSOR_ID=your-document-processor-id
VECTOR_INDEX_NAME=your-vector-index
VECTOR_INDEX_ENDPOINT_ID=your-endpoint-id
VECTOR_DEPLOYED_INDEX_ID=your-deployed-index-id

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=your-redis-password

# Document Processing
MIN_PDF_PAGE_COUNT=1
MAX_PDF_PAGE_COUNT=2000
DEFAULT_PDF_PASSWORD=

# SAML (Optional)
SAML_SP_ENTITY_ID=https://your-domain.com/backend/api/v1/auth/metadata/
SAML_SP_ACS_URL=https://your-domain.com/backend/api/v1/auth/sso/acs
```

---

## Deployment Notes

### Local Development
```bash
# Set up virtual environment
python -m venv venv
source venv/Scripts/activate  # Windows
source venv/bin/activate      # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp example.env .env

# Run the application
python run.py
```

### Docker Deployment
```bash
# Build the image
docker build -t georgia-digitization-platform .

# Run the container
docker run -p 5001:5001 georgia-digitization-platform
```

### Health Checks
The `/health` endpoint can be used for load balancer health checks and monitoring.

---

---

## API Summary

### Total Endpoints by Category

- **Authentication**: 9 endpoints (including test endpoints)
- **Document Management**: 18 endpoints
- **User Management**: 5 endpoints
- **System Administration**: 11 endpoints
- **GCS File Management**: 25 endpoints
- **Processor Rules**: 5 endpoints
- **Batch Processing**: 9 endpoints
- **Logs**: 3 endpoints
- **API v2 - Secure Documents**: 3 endpoints
- **API v2 - System**: 2 endpoints

**Total: 91 API endpoints**

### Quick Reference

#### Most Used Endpoints
1. `POST /backend/api/v1/auth/login` - User authentication
2. `POST /backend/api/v1/docs/upload` - Document upload
3. `POST /backend/api/v1/docs/chat` - AI-powered document chat
4. `GET /backend/api/v1/docs/history` - Document history
5. `GET /backend/api/v1/system/diagnosis` - System health check

#### Admin-Only Endpoints
- All `/backend/api/v1/users/*` endpoints
- All `/backend/api/v1/processor-rules/*` endpoints
- All `/backend/api/v1/batch/*` endpoints (except stats)
- `/backend/api/v1/system/clear-processing-data`
- `/backend/api/v1/system/reset-stuck-documents`

#### Streaming Endpoints
- `POST /backend/api/v1/docs/chat` - Server-Sent Events (SSE)
- `POST /backend/api/v2/docs/chat` - Server-Sent Events (SSE)

---

For more detailed information about specific endpoints, refer to the Postman collection (`Postman_Collection_Georgia_Digitization_Platform.json`) or contact the development team.