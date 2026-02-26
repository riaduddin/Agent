#!/usr/bin/env python3
"""
Test script to verify Authorization header works instead of query parameter.
"""

import requests
import json

def test_authorization_header():
    """Test the create-presentation endpoint with Authorization header"""
    
    base_url = "http://localhost:8060"
    jwt_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJfaWQiOiI2ODgwODdlMzA1MTk0OTc2YWVhOTk0MDMiLCJzdWIiOiI2ODgwODdlMzA1MTk0OTc2YWVhOTk0MDMiLCJlbWFpbCI6InJycmlhZHVkZGluQGdtYWlsLmNvbSIsInBhY2thZ2UiOiJ1bmxpbWl0ZWQiLCJpc192ZXJpZmllZCI6dHJ1ZSwicm9sZSI6InVzZXIiLCJpYXQiOjE3NjA5MzM4Njl9.E9cUSlM-uQRrQ48gpH47t7eUN9ioem-A63CCPVJhZnI"
    
    print("🧪 Testing Authorization Header (Bearer Token)")
    print("=" * 60)
    
    # Test data
    test_data = {
        "message": "Create a presentation about Machine Learning with 8 slides"
    }
    
    print("1. Testing with Authorization header...")
    print(f"   Headers: Authorization: Bearer {jwt_token[:20]}...")
    print(f"   Body: {json.dumps(test_data, indent=2)}")
    
    try:
        # Test with Authorization header
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {jwt_token}"
        }
        
        response = requests.post(
            f"{base_url}/create-presentation",
            headers=headers,
            json=test_data
        )
        
        print(f"\n2. Response:")
        print(f"   Status Code: {response.status_code}")
        print(f"   Response: {response.text}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n   ✅ SUCCESS! Authorization header works!")
            print(f"   📋 P_ID: {data.get('p_id')}")
            print(f"   📋 User ID: {data.get('user_id')}")
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

def test_invalid_authorization():
    """Test with invalid authorization header"""
    
    base_url = "http://localhost:8060"
    
    print("\n🧪 Testing Invalid Authorization Header")
    print("=" * 60)
    
    test_data = {
        "message": "Create a presentation about Data Science"
    }
    
    print("1. Testing with invalid authorization...")
    
    try:
        # Test with invalid authorization
        headers = {
            "Content-Type": "application/json",
            "Authorization": "InvalidToken123"
        }
        
        response = requests.post(
            f"{base_url}/create-presentation",
            headers=headers,
            json=test_data
        )
        
        print(f"   Status Code: {response.status_code}")
        print(f"   Response: {response.text}")
        
        if response.status_code == 401:
            print("   ✅ SUCCESS! Invalid token properly rejected")
            return True
        else:
            print("   ❌ FAILED: Should have returned 401 for invalid token")
            return False
            
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
        return False

def test_missing_authorization():
    """Test without authorization header"""
    
    base_url = "http://localhost:8060"
    
    print("\n🧪 Testing Missing Authorization Header")
    print("=" * 60)
    
    test_data = {
        "message": "Create a presentation about AI"
    }
    
    print("1. Testing without authorization header...")
    
    try:
        # Test without authorization
        headers = {
            "Content-Type": "application/json"
        }
        
        response = requests.post(
            f"{base_url}/create-presentation",
            headers=headers,
            json=test_data
        )
        
        print(f"   Status Code: {response.status_code}")
        print(f"   Response: {response.text}")
        
        if response.status_code == 422:  # Missing required field
            print("   ✅ SUCCESS! Missing authorization properly handled")
            return True
        else:
            print("   ❌ FAILED: Should have returned 422 for missing authorization")
            return False
            
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    print("Authorization Header Test")
    print("=" * 60)
    print("This test verifies that Authorization header works instead of query parameter")
    print("=" * 60)
    
    # Test valid authorization
    success1 = test_authorization_header()
    
    # Test invalid authorization
    success2 = test_invalid_authorization()
    
    # Test missing authorization
    success3 = test_missing_authorization()
    
    print("\n" + "=" * 60)
    if success1 and success2 and success3:
        print("🎉 All tests passed!")
        print("✅ Authorization header works correctly")
        print("✅ Invalid tokens are properly rejected")
        print("✅ Missing authorization is handled properly")
        print("\n💡 Use Authorization header instead of query parameter:")
        print("   Authorization: Bearer your_jwt_token_here")
    else:
        print("❌ Some tests failed!")
        print("🔧 Check the service and try again.")
