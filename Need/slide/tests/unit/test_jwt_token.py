#!/usr/bin/env python3
"""
Test JWT token validation.
"""

import jwt
import os

# Test JWT token
token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJfaWQiOiI2ODgwODdlMzA1MTk0OTc2YWVhOTk0MDMiLCJzdWIiOiI2ODgwODdlMzA1MTk0OTc2YWVhOTk0MDMiLCJlbWFpbCI6InJycmlhZHVkZGluQGdtYWlsLmNvbSIsInBhY2thZ2UiOiJ1bmxpbWl0ZWQiLCJpc192ZXJpZmllZCI6dHJ1ZSwicm9sZSI6InVzZXIiLCJpYXQiOjE3NjA5MzM4Njl9.E9cUSlM-uQRrQ48gpH47t7eUN9ioem-A63CCPVJhZnI"

# Get JWT secret from environment
JWT_SECRET = os.getenv("JWT_SECRET_KEY", "your-secret-key-here")
JWT_ALGORITHM = "HS256"

print(f"🔑 JWT Secret: {JWT_SECRET}")
print(f"🔑 JWT Algorithm: {JWT_ALGORITHM}")

try:
    # Decode the token
    payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    print(f"✅ Token decoded successfully!")
    print(f"📋 Payload: {payload}")
    
    user_id = payload.get("sub") or payload.get("_id")
    print(f"👤 User ID: {user_id}")
    
except Exception as e:
    print(f"❌ Token validation failed: {e}")
    print(f"🔍 Error type: {type(e).__name__}")
    
    # Try with different secrets
    print("\n🔍 Trying with different secrets...")
    
    secrets_to_try = [
        "your-secret-key-here",
        "secret",
        "my-secret-key",
        "jwt-secret",
        "presentation-secret"
    ]
    
    for secret in secrets_to_try:
        try:
            payload = jwt.decode(token, secret, algorithms=[JWT_ALGORITHM])
            print(f"✅ Found working secret: {secret}")
            print(f"📋 Payload: {payload}")
            break
        except:
            continue
    else:
        print("❌ No working secret found")
