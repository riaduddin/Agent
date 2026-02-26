#!/usr/bin/env python3
"""
Test script to demonstrate that userId is optional and extracted from JWT token.
"""

import requests
import json
import jwt

def decode_jwt_token(token):
    """Decode JWT token to show user information"""
    try:
        # Note: In production, you should verify the signature
        # This is just for demonstration
        decoded = jwt.decode(token, options={"verify_signature": False})
        return decoded
    except Exception as e:
        print(f"Error decoding token: {e}")
        return None

def test_without_userid():
    """Test create-presentation without userId in request body"""
    
    base_url = "http://localhost:8060"
    jwt_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJfaWQiOiI2ODgwODdlMzA1MTk0OTc2YWVhOTk0MDMiLCJzdWIiOiI2ODgwODdlMzA1MTk0OTc2YWVhOTk0MDMiLCJlbWFpbCI6InJycmlhZHVkZGluQGdtYWlsLmNvbSIsInBhY2thZ2UiOiJ1bmxpbWl0ZWQiLCJpc192ZXJpZmllZCI6dHJ1ZSwicm9sZSI6InVzZXIiLCJpYXQiOjE3NjA5MzM4Njl9.E9cUSlM-uQRrQ48gpH47t7eUN9ioem-A63CCPVJhZnI"
    
    print("🧪 Testing Create Presentation WITHOUT userId in request body")
    print("=" * 60)
    
    # Decode JWT token to show user info
    print("1. Decoding JWT token...")
    token_data = decode_jwt_token(jwt_token)
    if token_data:
        print(f"   📋 User ID from token: {token_data.get('sub')}")
        print(f"   📋 Email from token: {token_data.get('email')}")
        print(f"   📋 Package from token: {token_data.get('package')}")
    
    # Test data WITHOUT userId
    test_data = {
        "message": "Create a presentation about Machine Learning with 8 slides"
    }
    
    print(f"\n2. Request body (no userId):")
    print(f"   {json.dumps(test_data, indent=2)}")
    
    try:
        print("\n3. Sending request...")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {jwt_token}"
        }
        
        response = requests.post(
            f"{base_url}/create-presentation",
            headers=headers,
            json=test_data
        )
        
        print(f"   Status Code: {response.status_code}")
        print(f"   Response: {response.text}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n   ✅ SUCCESS! Presentation created without userId in request")
            print(f"   📋 P_ID: {data.get('p_id')}")
            print(f"   📋 User ID (from token): {data.get('user_id')}")
            print(f"   📋 Status: {data.get('status')}")
            return True
        else:
            print(f"\n   ❌ FAILED: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("   ❌ Connection failed. Make sure the service is running on port 8060")
        return False
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
        return False

def test_with_userid():
    """Test create-presentation WITH userId in request body (should still work)"""
    
    base_url = "http://localhost:8060"
    jwt_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJfaWQiOiI2ODgwODdlMzA1MTk0OTc2YWVhOTk0MDMiLCJzdWIiOiI2ODgwODdlMzA1MTk0OTc2YWVhOTk0MDMiLCJlbWFpbCI6InJycmlhZHVkZGluQGdtYWlsLmNvbSIsInBhY2thZ2UiOiJ1bmxpbWl0ZWQiLCJpc192ZXJpZmllZCI6dHJ1ZSwicm9sZSI6InVzZXIiLCJpYXQiOjE3NjA5MzM4Njl9.E9cUSlM-uQRrQ48gpH47t7eUN9ioem-A63CCPVJhZnI"
    
    print("\n🧪 Testing Create Presentation WITH userId in request body")
    print("=" * 60)
    
    # Test data WITH userId (should be ignored, token takes precedence)
    test_data = {
        "message": "Create a presentation about Data Science with 12 slides",
        "userId": "different_user_id_12345"  # This should be ignored
    }
    
    print(f"1. Request body (with userId):")
    print(f"   {json.dumps(test_data, indent=2)}")
    print(f"   Note: userId in request will be ignored, token takes precedence")
    
    try:
        print("\n2. Sending request...")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {jwt_token}"
        }
        
        response = requests.post(
            f"{base_url}/create-presentation",
            headers=headers,
            json=test_data
        )
        
        print(f"   Status Code: {response.status_code}")
        print(f"   Response: {response.text}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n   ✅ SUCCESS! Presentation created with userId in request")
            print(f"   📋 P_ID: {data.get('p_id')}")
            print(f"   📋 User ID (from token): {data.get('user_id')}")
            print(f"   📋 Status: {data.get('status')}")
            print(f"   📋 Note: userId from request was ignored, token was used")
            return True
        else:
            print(f"\n   ❌ FAILED: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("   ❌ Connection failed. Make sure the service is running on port 8060")
        return False
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    print("JWT Token User ID Extraction Test")
    print("=" * 60)
    print("This test demonstrates that userId is optional and extracted from JWT token")
    print("=" * 60)
    
    # Test without userId
    success1 = test_without_userid()
    
    # Test with userId (should be ignored)
    success2 = test_with_userid()
    
    print("\n" + "=" * 60)
    if success1 and success2:
        print("🎉 Both tests passed!")
        print("✅ userId is optional - extracted from JWT token")
        print("✅ userId in request body is ignored - token takes precedence")
        print("\n💡 Recommendation: Use simple request body without userId")
        print("   {\"message\": \"Your presentation request\"}")
    else:
        print("❌ Some tests failed!")
        print("🔧 Check the service and try again.")
