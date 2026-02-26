# File URLs Feature Documentation

## Overview

The Socket.IO implementation now supports optional `file_urls` parameter, allowing users to provide file URLs that will be processed into `file_context` for the presentation generation.

## Feature Details

### Request Model

```python
class PresentationRequest(BaseModel):
    message: str
    userId: Optional[str] = None  # Extracted from JWT token
    file_urls: Optional[List[str]] = None  # Optional file URLs
```

### Supported File Types

- **PDF**: `.pdf` files
- **DOCX**: `.docx`, `.doc` files  
- **TXT**: `.txt` files

### File Processing

Files are processed asynchronously using:
- `extract_text_from_url()`: Downloads and extracts content from individual files
- `build_file_context()`: Aggregates content from multiple files

## Usage Examples

### 1. With File URLs

```json
{
    "message": "Create a presentation about AI in Healthcare with 10 slides",
    "file_urls": [
        "https://example.com/healthcare_report.pdf",
        "https://example.com/ai_guidelines.docx"
    ]
}
```

**Result**: `file_context` will contain extracted text from both files.

### 2. Without File URLs

```json
{
    "message": "Create a presentation about Machine Learning with 8 slides"
}
```

**Result**: `file_context` will be empty.

### 3. Empty File URLs

```json
{
    "message": "Create a presentation about Data Science",
    "file_urls": []
}
```

**Result**: `file_context` will be empty.

## Implementation Details

### Session State

The session is created with the following state:

```python
initial_state = {
    "p_id": p_id,
    "user_id": user_id,
    "file_context": file_context  # Processed from file_urls
}
```

### File Processing Flow

1. **Request Received**: User provides `file_urls` (optional)
2. **File Download**: Each URL is downloaded using `httpx.AsyncClient`
3. **Content Extraction**: Files are processed based on type:
   - PDF: Uses Gemini to extract text
   - DOCX: Uses Gemini to extract text
   - TXT: Direct text extraction
4. **Context Building**: All extracted content is combined
5. **Session Creation**: `file_context` is stored in session state
6. **Agent Execution**: Agent can access `file_context` from session

### Error Handling

- **Download Failures**: Individual file failures don't stop processing
- **Unsupported Types**: Logged as warnings, skipped
- **Network Timeouts**: 60-second timeout per file
- **Empty Results**: Gracefully handled with empty `file_context`

## API Endpoints

### POST `/create-presentation`

**Request Body**:
```json
{
    "message": "string",
    "file_urls": ["string"]  // Optional
}
```

**Response**:
```json
{
    "p_id": "string",
    "status": "created"
}
```

### POST `/start-presentation/{p_id}`

**Headers**:
```
Authorization: Bearer <jwt_token>
```

**Response**:
```json
{
    "message": "Presentation generation started",
    "p_id": "string",
    "status": "processing"
}
```

## Testing

### Test Scripts

1. **`test_file_urls_support.py`**: Tests the implementation
2. **`test_file_urls_client.py`**: Tests the API endpoints
3. **Postman Collection**: Updated with examples

### Test Cases

1. **With Files**: Request with valid file URLs
2. **Without Files**: Request without `file_urls` field
3. **Empty Files**: Request with empty `file_urls` array
4. **Invalid Files**: Request with invalid/non-existent URLs
5. **Mixed Types**: Request with different file types

## Postman Collection

The Postman collection includes:

1. **"Create Presentation"**: With example file URLs
2. **"Create Presentation (No Files)"**: Without file URLs

### Example Requests

**With Files**:
```json
{
    "message": "Create a presentation about AI in Healthcare with 10 slides",
    "file_urls": [
        "https://example.com/document1.pdf",
        "https://example.com/document2.docx"
    ]
}
```

**Without Files**:
```json
{
    "message": "Create a presentation about Machine Learning with 8 slides"
}
```

## Error Scenarios

### Common Issues

1. **Invalid URLs**: Returns empty `file_context`
2. **Network Issues**: Individual files fail, others continue
3. **Unsupported Types**: Logged and skipped
4. **Large Files**: 60-second timeout per file
5. **Authentication**: JWT token required

### Error Responses

- **400 Bad Request**: Invalid request format
- **401 Unauthorized**: Missing or invalid JWT token
- **404 Not Found**: Invalid p_id
- **500 Internal Server Error**: Server-side processing error

## Best Practices

### File URLs

1. **Use HTTPS**: Ensure secure file access
2. **Public URLs**: Files must be publicly accessible
3. **Reasonable Size**: Avoid very large files
4. **Supported Types**: Use PDF, DOCX, or TXT files

### Request Format

1. **Valid JSON**: Ensure proper JSON formatting
2. **Required Fields**: Always include `message`
3. **Optional Fields**: `file_urls` and `userId` are optional
4. **Authentication**: Include JWT token in Authorization header

## Migration from SSE

The Socket.IO implementation maintains compatibility with the SSE version:

- Same file processing functions
- Same error handling
- Same session state structure
- Same agent execution flow

## Troubleshooting

### Common Issues

1. **"Context variable not found: file_context"**
   - **Cause**: Session created without `file_context`
   - **Fix**: Ensure `file_context` is included in initial state

2. **"Session not found"**
   - **Cause**: Session not created before agent execution
   - **Fix**: Ensure session creation before running agent

3. **File download failures**
   - **Cause**: Invalid URLs or network issues
   - **Fix**: Check URL validity and network connectivity

### Debug Steps

1. Check logs for file processing errors
2. Verify file URLs are accessible
3. Test with simple text files first
4. Check network connectivity
5. Verify JWT token validity

## Future Enhancements

1. **More File Types**: Support for images, presentations
2. **File Validation**: Pre-validate file accessibility
3. **Progress Tracking**: Real-time file processing status
4. **Caching**: Cache processed files for reuse
5. **Batch Processing**: Process multiple files in parallel
