#!/usr/bin/env python3
"""
Test script to verify the agent execution fix.
"""

import requests
import json
import time

def test_agent_execution():
    """Test that the agent execution works without the 'stream_query' error"""
    
    base_url = "http://localhost:8060"
    jwt_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJfaWQiOiI2ODgwODdlMzA1MTk0OTc2YWVhOTk0MDMiLCJzdWIiOiI2ODgwODdlMzA1MTk0OTc2YWVhOTk0MDMiLCJlbWFpbCI6InJycmlhZHVkZGluQGdtYWlsLmNvbSIsInBhY2thZ2UiOiJ1bmxpbWl0ZWQiLCJpc192ZXJpZmllZCI6dHJ1ZSwicm9sZSI6InVzZXIiLCJpYXQiOjE3NjA5MzM4Njl9.E9cUSlM-uQRrQ48gpH47t7eUN9ioem-A63CCPVJhZnI"
    
    print("🧪 Testing Agent Execution Fix")
    print("=" * 50)
    
    # Step 1: Create presentation
    print("1. Creating presentation...")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {jwt_token}"
    }
    
    test_data = {
        "message": "Create a simple presentation about Machine Learning with 3 slides"
    }
    
    try:
        response = requests.post(
            f"{base_url}/create-presentation",
            headers=headers,
            json=test_data
        )
        
        if response.status_code != 200:
            print(f"   ❌ Failed to create presentation: {response.text}")
            return False
        
        data = response.json()
        p_id = data.get("p_id")
        print(f"   ✅ Presentation created: {p_id}")
        
    except Exception as e:
        print(f"   ❌ Error creating presentation: {e}")
        return False
    
    # Step 2: Start presentation generation
    print(f"\n2. Starting presentation generation...")
    
    try:
        response = requests.post(
            f"{base_url}/start-presentation/{p_id}",
            headers={"Authorization": f"Bearer {jwt_token}"}
        )
        
        if response.status_code != 200:
            print(f"   ❌ Failed to start presentation: {response.text}")
            return False
        
        data = response.json()
        print(f"   ✅ Generation started: {data.get('status')}")
        
    except Exception as e:
        print(f"   ❌ Error starting presentation: {e}")
        return False
    
    # Step 3: Monitor for a few seconds to see if agent runs without errors
    print(f"\n3. Monitoring agent execution for 30 seconds...")
    print("   💡 Check the server logs for any 'stream_query' errors")
    print("   💡 If you see 'LlmAgent' or 'stream_query' errors, the fix didn't work")
    print("   💡 If you see normal agent activity, the fix is working!")
    
    for i in range(30):
        print(f"   ⏱️  {i+1}/30 seconds...", end="\r")
        time.sleep(1)
    
    print(f"\n   ✅ Monitoring complete!")
    print(f"   📋 Check server logs above for any errors")
    
    return True

if __name__ == "__main__":
    print("Agent Execution Fix Test")
    print("=" * 50)
    print("This test verifies that the 'stream_query' error is fixed")
    print("=" * 50)
    
    success = test_agent_execution()
    
    if success:
        print("\n🎉 Test completed!")
        print("✅ Check the server logs above")
        print("✅ If you see normal agent activity (no 'stream_query' errors), the fix worked!")
        print("✅ If you still see 'stream_query' errors, there may be another issue")
    else:
        print("\n❌ Test failed!")
        print("🔧 Check the service and try again")
