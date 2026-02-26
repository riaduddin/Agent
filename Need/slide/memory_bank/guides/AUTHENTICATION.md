# Authentication Setup Guide

## Current Status ✅

Your Python FastAPI application already has **complete JWT authentication middleware** implemented in `auth_middleware.py`.

## How It Works

1. **Node.js Auth Service** (`@ridz-shothikai/shothik-auth-service`):
   - Generates JWT tokens when users log in
   - Signs tokens with a secret key

2. **Python Presentation Service** (this project):
   - Validates JWT tokens from Authorization headers
   - Extracts user information (`user_id`, `email`, `package`, `role`)
   - Protects endpoints using `Depends(get_current_user)`

## Required Configuration

### Step 1: Add JWT Configuration to `.env`

Your Python app needs the **same JWT secret and algorithm** used by your Node.js auth service.

Create or update `.env` file in the project root:

```env
# ==========================================
# EXISTING CONFIGURATIONS (keep these)
# ==========================================
DATABASE_URL=your_mongodb_connection_string
GEMINI_API_KEY=your_gemini_api_key
GOOGLE_APPLICATION_CREDENTIALS=service-account.json
GEMINI_MODEL_FLASH=gemini-2.5-flash
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_API_KEY=
GCS_BUCKET_NAME=your_bucket_name

# ==========================================
# JWT AUTHENTICATION (ADD THESE)
# ==========================================
# This MUST match the secret used by @ridz-shothikai/shothik-auth-service
JWT_SECRET=your_jwt_secret_key_from_nodejs_service
JWT_ALGORITHM=HS256
```

### Step 2: Get the JWT Secret from Your Node.js Auth Service

**Option A: Check your Node.js auth service configuration**
Look for the JWT_SECRET in your Node.js service's `.env` file or configuration.

**Option B: Ask your team/check documentation**
The JWT secret is typically shared between services that need to validate tokens.

### Step 3: Verify Token Format

Your Node.js auth service should generate JWT tokens with this payload structure:

```json
{
  "_id": "user_id_here",
  "sub": "user_id_here",
  "email": "user@example.com",
  "package": "premium",
  "is_verified": true,
  "role": "user",
  "iat": 1234567890
}
```

The Python middleware expects these fields (see `auth_middleware.py` lines 118-122).

## Current Protected Endpoints

Authentication is already applied to these endpoints:

### In `main.py`:
- `GET /slides/` - Get slides for a presentation
- `GET /simulation-logs/{p_id}` - Stream agent logs
- All other endpoints that use `Depends(get_current_user)`

### In `app_sse.py`:
- `POST /sse/presentations` - Create presentation with SSE
- `POST /sse/presentations/clone` - Clone presentation

## How to Use (Client-Side)

Your frontend should send requests with the Authorization header:

```javascript
fetch('http://your-api/slides/?p_id=123', {
  headers: {
    'Authorization': `Bearer ${jwtToken}`
  }
})
```

## Testing Authentication

### 1. Test with a valid token:
```bash
curl -H "Authorization: Bearer YOUR_JWT_TOKEN" \
     http://localhost:8000/health
```

### 2. Check logs:
```
INFO:auth_middleware:User authenticated: user_id (user@example.com)
```

### 3. Test with invalid token:
```
HTTP 401: Invalid authentication token
```

## Security Features Implemented ✅

1. **JWT Signature Verification** - Prevents token tampering
2. **Token Expiration Check** - Rejects expired tokens
3. **User Verification Check** - Only verified users can access endpoints
4. **Bearer Token Scheme** - Standard HTTP authentication
5. **Swagger UI Integration** - HTTPBearer security scheme

## Optional: Create Admin Endpoints

You can create role-based access control:

```python
# Add to auth_middleware.py
async def require_admin(
    current_user: AuthenticatedUser = Depends(get_current_user)
) -> AuthenticatedUser:
    """Require admin role"""
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user

# Use in main.py
@app.delete("/admin/presentations/{p_id}")
async def delete_presentation(
    p_id: str,
    admin: AuthenticatedUser = Depends(require_admin)
):
    # Only admins can access
    pass
```

## Troubleshooting

### Error: "Authentication configuration error"
- **Cause**: `JWT_SECRET` not set in `.env`
- **Fix**: Add `JWT_SECRET=your_secret` to `.env`

### Error: "Invalid authentication token"
- **Cause**: Token signed with different secret or algorithm
- **Fix**: Ensure Python app uses same `JWT_SECRET` and `JWT_ALGORITHM` as Node.js service

### Error: "User is not verified"
- **Cause**: `is_verified: false` in JWT payload
- **Fix**: User must verify their account in the auth service

### Error: "Token has expired"
- **Cause**: Token's `exp` claim has passed
- **Fix**: User must log in again to get a new token

## Next Steps

1. ✅ **Authentication middleware already implemented**
2. ⚠️ **Add `JWT_SECRET` to `.env`** (must match your Node.js service)
3. ✅ **Protected endpoints already configured**
4. 🔧 **Test with a real JWT token from your auth service**

## Summary

- **The `.npmrc` file is for your Node.js auth service**, not this Python project
- **This Python project only validates tokens**, it doesn't generate them
- **You only need to configure `JWT_SECRET` and `JWT_ALGORITHM` in `.env`**
- **No npm packages needed in this Python project**

Your authentication is already fully functional! Just add the JWT configuration and you're done. 🎉

