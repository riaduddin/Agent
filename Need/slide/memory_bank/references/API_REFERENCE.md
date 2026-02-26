# API Reference

Complete API documentation for all endpoints.

---

## 🔐 Authentication

All endpoints (except `/health`) require JWT authentication.

**Header**:
```
Authorization: Bearer <JWT_TOKEN>
```

**JWT Payload**:
```json
{
  "_id": "user_id",
  "email": "user@example.com",
  "package": "premium",
  "is_verified": true,
  "role": "user",
  "iat": 1234567890
}
```

---

## 📡 SSE Endpoints

### **POST /sse/presentations**

Create a new presentation with Server-Sent Events streaming.

**Request**:
```json
{
  "message": "Create a presentation about AI in healthcare",
  "p_id": "optional_presentation_id",
  "file_urls": ["https://...", "https://..."]
}
```

**Parameters**:
- `message` (string, required): User query describing the presentation
- `p_id` (string, optional): Presentation ID for updates/regeneration
- `file_urls` (array, optional): Array of file URLs (PDF, DOCX, TXT)

**Response**: SSE stream

**Event Types**:

1. **chunk** - Progress updates
```json
{
  "author": "agent_name",
  "text": "Progress message",
  "p_id": "presentation_id",
  "at": "2025-10-21T12:00:00"
}
```

2. **source** - Research sources
```json
{
  "author": "browser_agent",
  "title": "Source Title",
  "url": "https://...",
  "p_id": "presentation_id",
  "at": "2025-10-21T12:00:00"
}
```

3. **slide** - Generated HTML
```json
{
  "author": "enhanced_slide_generator_0",
  "text": "<!DOCTYPE html>...",
  "p_id": "presentation_id",
  "at": "2025-10-21T12:00:00"
}
```

4. **done** - Completion
```json
{
  "status": "completed",
  "p_id": "presentation_id"
}
```

**Example (JavaScript)**:
```javascript
const eventSource = new EventSource(
  'http://localhost:8000/sse/presentations',
  {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      message: 'AI in healthcare presentation'
    })
  }
);

eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(data);
};

eventSource.onerror = (error) => {
  console.error('SSE error:', error);
  eventSource.close();
};
```

**Status Codes**:
- `200` - Success (SSE stream)
- `401` - Unauthorized (invalid JWT)
- `403` - Forbidden (user not verified)
- `500` - Server error

---

### **POST /sse/presentations/clone**

Clone an existing presentation.

**Request**:
```json
{
  "original_p_id": "presentation_id_to_clone"
}
```

**Response**: Same SSE stream as `/sse/presentations`

---

## 📄 REST Endpoints

### **GET /health**

Health check endpoint (no authentication required).

**Response**:
```json
{
  "status": "healthy",
  "database": true
}
```

**Status Codes**:
- `200` - Service healthy
- `500` - Service unhealthy

---

### **GET /slides/**

Get all slides for a presentation.

**Parameters**:
- `p_id` (query, required): Presentation ID

**Response**:
```json
{
  "p_id": "presentation_id",
  "slides": [
    {
      "slide_number": 1,
      "html": "<!DOCTYPE html>...",
      "created_at": "2025-10-21T12:00:00"
    }
  ]
}
```

**Status Codes**:
- `200` - Success
- `401` - Unauthorized
- `404` - Presentation not found

---

### **GET /simulation-logs/{p_id}**

Stream agent execution logs for a presentation.

**Parameters**:
- `p_id` (path, required): Presentation ID

**Response**: NDJSON stream
```json
{"agent": "KeywordResearchAgent", "message": "Generated 10 queries", "timestamp": "..."}
{"agent": "BrowserAgent", "message": "Searching web...", "timestamp": "..."}
...
{"status": "completed"}
```

**Status Codes**:
- `200` - Success
- `401` - Unauthorized
- `404` - No logs found

---

### **POST /upload**

Upload files (PDF, DOCX, TXT) to GCS.

**Request**: `multipart/form-data`
```
files: [File, File, ...]
```

**Response**:
```json
{
  "uploads": [
    {
      "filename": "document.pdf",
      "public_url": "https://storage.googleapis.com/...",
      "signed_url": "https://storage.googleapis.com/...?signature=...",
      "object_name": "uploads/user_id/document.pdf"
    }
  ]
}
```

**Status Codes**:
- `200` - Success
- `401` - Unauthorized
- `413` - File too large
- `500` - Upload failed

---

### **GET /presentations**

Get all presentations for the authenticated user.

**Response**:
```json
{
  "presentations": [
    {
      "p_id": "presentation_id",
      "title": "AI in Healthcare",
      "slide_count": 6,
      "created_at": "2025-10-21T12:00:00",
      "updated_at": "2025-10-21T12:05:00"
    }
  ]
}
```

**Status Codes**:
- `200` - Success
- `401` - Unauthorized

---

### **DELETE /presentations/{p_id}**

Delete a presentation and its Qdrant data.

**Parameters**:
- `p_id` (path, required): Presentation ID

**Response**:
```json
{
  "message": "Presentation deleted successfully",
  "p_id": "presentation_id"
}
```

**Status Codes**:
- `200` - Success
- `401` - Unauthorized
- `404` - Presentation not found

---

## 🔧 Admin Endpoints

### **GET /admin/running-agents**

Get list of currently running agents (no auth currently).

**Response**:
```json
{
  "running_agents": [
    {
      "p_id": "presentation_id",
      "status": "running",
      "started_at": "2025-10-21T12:00:00"
    }
  ]
}
```

---

## 📊 Data Models

### **PresentationRequest**
```python
{
  "message": str,           # Required
  "p_id": Optional[str],    # Optional
  "file_urls": Optional[List[str]]  # Optional
}
```

### **ClonePresentationRequest**
```python
{
  "original_p_id": str      # Required
}
```

### **UploadResponseItem**
```python
{
  "filename": str,
  "public_url": str,
  "signed_url": str,
  "object_name": str
}
```

### **AuthenticatedUser**
```python
{
  "user_id": str,
  "email": str,
  "package": str,
  "is_verified": bool,
  "role": str
}
```

---

## 🌐 CORS Configuration

**Allowed Origins**: `*` (all origins)  
**Allowed Methods**: All  
**Allowed Headers**: All  
**Credentials**: Enabled

---

## 🔄 Rate Limiting

**Current Status**: ⚠️ Not implemented

**Recommendation**:
- 10 requests per minute per user
- 100 requests per hour per user
- 1 concurrent presentation generation per user

---

## 📝 Error Responses

### **Standard Error Format**
```json
{
  "detail": "Error message",
  "status_code": 401
}
```

### **Common Errors**

**401 Unauthorized**
```json
{
  "detail": "Invalid authentication token",
  "headers": {"WWW-Authenticate": "Bearer"}
}
```

**403 Forbidden**
```json
{
  "detail": "User is not verified. Please verify your account."
}
```

**404 Not Found**
```json
{
  "detail": "Presentation not found"
}
```

**500 Internal Server Error**
```json
{
  "detail": "Agent execution failed: ..."
}
```

---

## 🧪 Testing Examples

### **cURL**

```bash
# Health check
curl http://localhost:8000/health

# Create presentation
curl -X POST http://localhost:8000/sse/presentations \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message": "AI in healthcare"}'

# Get slides
curl "http://localhost:8000/slides/?p_id=abc123" \
  -H "Authorization: Bearer YOUR_TOKEN"

# Upload file
curl -X POST http://localhost:8000/upload \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "files=@document.pdf"
```

### **Python (requests)**

```python
import requests

token = "YOUR_JWT_TOKEN"
headers = {"Authorization": f"Bearer {token}"}

# Create presentation
response = requests.post(
    "http://localhost:8000/sse/presentations",
    headers=headers,
    json={"message": "AI in healthcare"},
    stream=True
)

# Process SSE
for line in response.iter_lines():
    if line:
        print(line.decode('utf-8'))
```

### **JavaScript (Fetch)**

```javascript
const token = "YOUR_JWT_TOKEN";

// Create presentation
fetch('http://localhost:8000/sse/presentations', {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    message: 'AI in healthcare'
  })
}).then(async response => {
  const reader = response.body.getReader();
  while (true) {
    const {done, value} = await reader.read();
    if (done) break;
    console.log(new TextDecoder().decode(value));
  }
});
```

---

## 📖 API Versioning

**Current Version**: v1 (implicit, no version prefix)

**Future**: Consider adding `/api/v1/` prefix for versioning.

---

## 🔒 Security Best Practices

1. **Always use HTTPS** in production
2. **Rotate JWT_SECRET** regularly
3. **Implement rate limiting**
4. **Validate file uploads** (size, type, content)
5. **Sanitize user inputs**
6. **Use signed URLs** for GCS (temporary access)
7. **Log security events**

---

**Related Documents**:
- `AUTHENTICATION.md` - JWT setup guide
- `TROUBLESHOOTING.md` - Common API errors
- `AGENT_CATALOG.md` - Agent details

