#!/usr/bin/env python3
"""
Test script to verify the new Socket.IO endpoints work.
"""

import requests
import json
import os
import sys
from dotenv import load_dotenv

load_dotenv()

def test_socketio_endpoints():
    """Test the new Socket.IO endpoints"""
    print("🧪 Testing Socket.IO Endpoints")
    print("=" * 50)
    
    # Get configuration
    base_url = os.getenv("BASE_URL", "http://localhost:8060")
    jwt_token = os.getenv("JWT_TOKEN", "your_jwt_token_here")
    
    if jwt_token == "your_jwt_token_here":
        print("⚠️ Please set JWT_TOKEN in your .env file or environment")
        print("   Example: JWT_TOKEN=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...")
        return False
    
    headers = {
        "Authorization": f"Bearer {jwt_token}"
    }
    
    print(f"🔗 Base URL: {base_url}")
    print(f"🔑 JWT Token: {jwt_token[:20]}...")
    print()
    
    # Test 1: Create presentation
    print("📝 Test 1: Create Presentation")
    print("------------------------------")
    create_payload = {
        "message": "Create a presentation about AI in Healthcare with 3 slides",
        "file_urls": [
            "https://example.com/healthcare_report.pdf"
        ]
    }
    
    try:
        response = requests.post(
            f"{base_url}/create-presentation",
            headers={"Content-Type": "application/json", **headers},
            json=create_payload,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            p_id = data.get("p_id")
            print(f"✅ Presentation created: {p_id}")
            print(f"📋 Response: {json.dumps(data, indent=2)}")
        else:
            print(f"❌ Failed to create presentation: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error creating presentation: {e}")
        return False
    
    # Test 2: Get presentation status
    print(f"\n📊 Test 2: Get Presentation Status")
    print("----------------------------------")
    try:
        response = requests.get(
            f"{base_url}/presentation/{p_id}/status",
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Status retrieved: {data.get('status')}")
            print(f"📋 Response: {json.dumps(data, indent=2)}")
        else:
            print(f"❌ Failed to get status: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Error getting status: {e}")
    
    # Test 3: Get presentation data
    print(f"\n📄 Test 3: Get Presentation Data")
    print("--------------------------------")
    try:
        response = requests.get(
            f"{base_url}/presentation/{p_id}/data",
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Data retrieved: {len(data.get('slides', []))} slides")
            print(f"📋 Response: {json.dumps(data, indent=2)}")
        else:
            print(f"❌ Failed to get data: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Error getting data: {e}")
    
    # Test 4: Start presentation
    print(f"\n🚀 Test 4: Start Presentation")
    print("-----------------------------")
    try:
        response = requests.post(
            f"{base_url}/start-presentation/{p_id}",
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Presentation started: {data.get('status')}")
            print(f"📋 Response: {json.dumps(data, indent=2)}")
        else:
            print(f"❌ Failed to start presentation: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Error starting presentation: {e}")
    
    # Test 5: Get status again (after starting)
    print(f"\n📊 Test 5: Get Status Again")
    print("---------------------------")
    try:
        response = requests.get(
            f"{base_url}/presentation/{p_id}/status",
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Status retrieved: {data.get('status')}")
            print(f"📋 Response: {json.dumps(data, indent=2)}")
        else:
            print(f"❌ Failed to get status: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Error getting status: {e}")
    
    print("\n🎉 Socket.IO endpoint tests completed!")
    return True

if __name__ == "__main__":
    print("Socket.IO Endpoints Test")
    print("=" * 50)
    print("This test verifies that the new Socket.IO endpoints work")
    print("=" * 50)
    
    success = test_socketio_endpoints()
    
    if success:
        print("\n✅ SUCCESS!")
        print("🎉 Socket.IO endpoints are working!")
        print("\n📋 Available Endpoints:")
        print("   • POST /create-presentation - Create a new presentation")
        print("   • POST /start-presentation/{p_id} - Start presentation generation")
        print("   • GET /presentation/{p_id}/status - Get presentation status")
        print("   • GET /presentation/{p_id}/data - Get complete presentation data")
        print("\n🚀 The 404 error should now be resolved!")
    else:
        print("\n❌ FAILED!")
        print("🔧 There are still issues to resolve")
        sys.exit(1)
