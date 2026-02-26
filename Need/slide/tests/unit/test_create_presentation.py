#!/usr/bin/env python3
"""
Test script to verify the create-presentation endpoint works correctly.
"""

import requests
import json

def test_create_presentation():
    """Test the create-presentation endpoint"""
    
    # Configuration
    base_url = "http://localhost:8060"
    jwt_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJfaWQiOiI2ODgwODdlMzA1MTk0OTc2YWVhOTk0MDMiLCJzdWIiOiI2ODgwODdlMzA1MTk0OTc2YWVhOTk0MDMiLCJlbWFpbCI6InJycmlhZHVkZGluQGdtYWlsLmNvbSIsInBhY2thZ2UiOiJ1bmxpbWl0ZWQiLCJpc192ZXJpZmllZCI6dHJ1ZSwicm9sZSI6InVzZXIiLCJpYXQiOjE3NjA5MzM4Njl9.E9cUSlM-uQRrQ48gpH47t7eUN9ioem-A63CCPVJhZnI"
    
    # Test data - userId is optional (extracted from JWT token)
    test_data = {
        "message": "Create a presentation about AI in Healthcare with 10 slides"
    }
    
    print("🧪 Testing Create Presentation Endpoint")
    print("=" * 50)
    
    try:
        # Test health check first
        print("1. Testing health check...")
        health_response = requests.get(f"{base_url}/health")
        if health_response.status_code == 200:
            print("   ✅ Service is running")
        else:
            print(f"   ❌ Service health check failed: {health_response.status_code}")
            return False
        
        # Test create presentation
        print("2. Testing create presentation...")
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
            print(f"   ✅ Presentation created successfully!")
            print(f"   📋 P_ID: {data.get('p_id')}")
            print(f"   📋 Status: {data.get('status')}")
            print(f"   📋 User ID: {data.get('user_id')}")
            return True
        else:
            print(f"   ❌ Create presentation failed: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("   ❌ Connection failed. Make sure the service is running on port 8060")
        print("   💡 Start the service with: python run_windows_simple.bat")
        return False
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    print("Socket.IO Create Presentation Test")
    print("=" * 50)
    
    success = test_create_presentation()
    
    if success:
        print("\n🎉 Test passed! Create presentation endpoint is working.")
        print("📋 You can now use this endpoint in Postman.")
    else:
        print("\n❌ Test failed! Check the service and try again.")
        print("💡 Make sure to:")
        print("   1. Start the service: python run_windows_simple.bat")
        print("   2. Check if Redis is running")
        print("   3. Verify the JWT token is valid")
