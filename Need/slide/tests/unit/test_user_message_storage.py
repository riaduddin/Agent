#!/usr/bin/env python3
"""
Test that user message is stored in agent output.
"""

import requests
import time
import json
import sys

BASE_URL = "http://127.0.0.1:8060"
JWT_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJfaWQiOiI2ODgwODdlMzA1MTk0OTc2YWVhOTk0MDMiLCJzdWIiOiI2ODgwODdlMzA1MTk0OTc2YWVhOTk0MDMiLCJlbWFpbCI6InJycmlhZHVkZGluQGdtYWlsLmNvbSIsInBhY2thZ2UiOiJ1bmxpbWl0ZWQiLCJpc192ZXJpZmllZCI6dHJ1ZSwicm9sZSI6InVzZXIiLCJpYXQiOjE3NjA5MzM4Njl9.E9cUSlM-uQRrQ48gpH47t7eUN9ioem-A63CCPVJhZnI"

def test_create_presentation_with_specific_message():
    """Test creating a presentation with a specific user message"""
    print("📝 Testing Create Presentation with Specific Message")
    print("=" * 50)
    
    # Use a very specific message that we can easily identify
    test_message = "Create a presentation about Quantum Computing and its applications in cryptography. Include sections on quantum algorithms, quantum key distribution, and post-quantum cryptography. Make it technical and detailed for computer science students."
    
    try:
        payload = {
            "message": test_message,
            "file_urls": []
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {JWT_TOKEN}"
        }
        
        response = requests.post(f"{BASE_URL}/create-presentation", 
                               json=payload, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Presentation created successfully!")
            print(f"📋 Presentation ID: {data.get('p_id')}")
            print(f"📋 Status: {data.get('status')}")
            print(f"📋 User Message: {test_message[:100]}...")
            return data.get('p_id'), test_message
        else:
            print(f"❌ Failed to create presentation: {response.status_code}")
            print(f"📝 Response: {response.text}")
            return None, None
            
    except Exception as e:
        print(f"❌ Error creating presentation: {e}")
        return None, None

def test_start_presentation(p_id):
    """Test starting a presentation"""
    print(f"\n🚀 Testing Start Presentation: {p_id}")
    print("=" * 50)
    
    try:
        headers = {
            "Authorization": f"Bearer {JWT_TOKEN}"
        }
        
        response = requests.post(f"{BASE_URL}/start-presentation/{p_id}", 
                               headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Presentation started!")
            print(f"📋 Message: {data.get('message')}")
            print(f"📋 Status: {data.get('status')}")
            return True
        else:
            print(f"❌ Failed to start presentation: {response.status_code}")
            print(f"📝 Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error starting presentation: {e}")
        return False

def test_check_status(p_id, max_checks=6):
    """Check presentation status multiple times"""
    print(f"\n📊 Monitoring Status: {p_id}")
    print("=" * 50)
    
    for i in range(max_checks):
        try:
            headers = {
                "Authorization": f"Bearer {JWT_TOKEN}"
            }
            
            response = requests.get(f"{BASE_URL}/presentation/{p_id}/status", 
                                  headers=headers, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                status = data.get('status', 'unknown')
                print(f"📋 Check {i+1}/{max_checks}: Status = {status}")
                
                if status in ['completed', 'failed']:
                    print(f"🏁 Final status: {status}")
                    return status
                    
            else:
                print(f"⚠️ Status check failed: {response.status_code}")
                
        except Exception as e:
            print(f"⚠️ Error checking status: {e}")
        
        # Wait before next check
        print("⏳ Waiting 15 seconds before next check...")
        time.sleep(15)
    
    print("⏰ Status monitoring completed (max checks reached)")
    return "timeout"

def main():
    """Main test function"""
    print("🧪 User Message Storage Test")
    print("=" * 50)
    print("This test verifies that user messages are stored in agent output")
    print("=" * 50)
    
    # Test 1: Create presentation with specific message
    p_id, test_message = test_create_presentation_with_specific_message()
    if not p_id:
        print("\n❌ Failed to create presentation!")
        print("💡 Make sure the server is running: python main.py")
        sys.exit(1)
    
    # Test 2: Start presentation
    if not test_start_presentation(p_id):
        print("\n❌ Failed to start presentation!")
        sys.exit(1)
    
    # Test 3: Monitor status
    final_status = test_check_status(p_id)
    
    print("\n🎯 Test Results Summary")
    print("=" * 50)
    print(f"📋 Presentation ID: {p_id}")
    print(f"📋 Test Message: {test_message[:100]}...")
    print(f"📋 Final Status: {final_status}")
    
    if final_status == "completed":
        print("✅ SUCCESS: Presentation completed successfully!")
        print("🎉 User message storage test completed!")
        print("\n💡 What to look for in server logs:")
        print("   1. '📝 Using user message: [your specific message]...'")
        print("   2. 'Stored agent output in agent_outputs_2' messages")
        print("   3. User message should be stored in database entries")
        print("   4. Event data should include user_message field")
        print("\n🔍 Check the database for:")
        print("   • agent_outputs_2 collection should have user_message field")
        print("   • Each agent output should contain the original user message")
        print("   • Event data should include user_message in broadcasts")
    elif final_status == "failed":
        print("❌ FAILED: Presentation generation failed!")
        print("💡 Check server logs for errors")
    elif final_status == "timeout":
        print("⏰ TIMEOUT: Presentation did not complete in time!")
        print("💡 Agent may be hanging - check the time/event limits")
    else:
        print("⚠️ UNKNOWN: Presentation status unclear!")
        print("💡 Check server logs for more information")

if __name__ == "__main__":
    main()
