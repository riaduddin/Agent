#!/usr/bin/env python3
"""
Test JWT Verification
Test the JWT verification logic used in Socket.IO connection handler
"""

import jwt
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_jwt_verification():
    """Test JWT verification logic"""
    print("🔍 Testing JWT Verification")
    print("=" * 40)
    
    # Test token
    token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJfaWQiOiI2ODgxYjhjMzE2ZjViODk0MzZkYzBiNzMiLCJzdWIiOiI2ODgxYjhjMzE2ZjViODk0MzZkYzBiNzMiLCJlbWFpbCI6Im1obWFoZWRpMDAwQGdtYWlsLmNvbSIsInBhY2thZ2UiOiJ2YWx1ZV9wbGFuIiwiaXNfdmVyaWZpZWQiOnRydWUsInJvbGUiOiJ1c2VyIiwiaWF0IjoxNzYxMzg3MzUxfQ.wxPfF0xdNlE4oMEF63JTqvdbPq1rYNdQCgB4vKJ9LcY"
    
    try:
        # Get JWT secret from environment
        JWT_SECRET = os.getenv("JWT_SECRET", "your-secret-key-here")
        JWT_ALGORITHM = "HS256"
        
        print(f"JWT_SECRET: {JWT_SECRET[:20]}...")
        print(f"JWT_ALGORITHM: {JWT_ALGORITHM}")
        
        # Decode token
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        print(f"✅ JWT decoded successfully")
        print(f"Payload: {payload}")
        
        # Extract user_id
        user_id = payload.get("sub") or payload.get("_id")
        print(f"User ID: {user_id}")
        
        if not user_id:
            print("❌ No user_id found in token")
        else:
            print("✅ User ID extracted successfully")
            
    except Exception as e:
        print(f"❌ JWT verification failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_jwt_verification()

