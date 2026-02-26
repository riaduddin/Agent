#!/usr/bin/env python3
"""
Quick test for the complete workflow with agent output fixes.
"""

import requests
import time
import json
import sys

BASE_URL = "http://127.0.0.1:8060"
JWT_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJfaWQiOiI2ODgwODdlMzA1MTk0OTc2YWVhOTk0MDMiLCJzdWIiOiI2ODgwODdlMzA1MTk0OTc2YWVhOTk0MDMiLCJlbWFpbCI6InJycmlhZHVkZGluQGdtYWlsLmNvbSIsInBhY2thZ2UiOiJ1bmxpbWl0ZWQiLCJpc192ZXJpZmllZCI6dHJ1ZSwicm9sZSI6InVzZXIiLCJpYXQiOjE3NjA5MzM4Njl9.E9cUSlM-uQRrQ48gpH47t7eUN9ioem-A63CCPVJhZnI"

def test_server_status():
    """Test if server is running"""
    print("🔍 Testing Server Status")
    print("=" * 50)
    
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ Server is running and accessible")
            print(f"📋 Status: {data.get('status')}")
            print(f"📋 Message: {data.get('message')}")
            return True
        else:
            print(f"⚠️ Server responded with status: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Server is not running or not accessible")
        print("💡 Start the server with: python main.py")
        return False
    except Exception as e:
        print(f"❌ Error checking server: {e}")
        return False

def test_create_presentation():
    """Test creating a presentation"""
    print("\n📝 Testing Create Presentation")
    print("=" * 50)
    
    try:
        payload = {
            "topic": "Test Presentation for Agent Output",
            "description": "Testing agent output and workflow",
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
            return data.get('p_id')
        else:
            print(f"❌ Failed to create presentation: {response.status_code}")
            print(f"📝 Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Error creating presentation: {e}")
        return None

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
        print("⏳ Waiting 10 seconds before next check...")
        time.sleep(10)
    
    print("⏰ Status monitoring completed (max checks reached)")
    return "timeout"

def test_get_data(p_id):
    """Test getting presentation data"""
    print(f"\n📄 Testing Get Data: {p_id}")
    print("=" * 50)
    
    try:
        headers = {
            "Authorization": f"Bearer {JWT_TOKEN}"
        }
        
        response = requests.get(f"{BASE_URL}/presentation/{p_id}/data", 
                              headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Data retrieved successfully!")
            print(f"📋 Status: {data.get('status')}")
            print(f"📋 Slides: {len(data.get('slides', []))}")
            print(f"📋 Total slides: {data.get('total_slides', 0)}")
            
            # Show first few slides
            slides = data.get('slides', [])
            if slides:
                print(f"📋 First slide preview:")
                for i, slide in enumerate(slides[:3]):
                    print(f"   Slide {i+1}: {slide.get('title', 'No title')[:50]}...")
            
            return True
        else:
            print(f"❌ Failed to get data: {response.status_code}")
            print(f"📝 Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error getting data: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Quick Workflow Test")
    print("=" * 50)
    print("This test will verify the agent output fixes are working")
    print("=" * 50)
    
    # Test 1: Server status
    if not test_server_status():
        print("\n❌ Server is not running!")
        print("💡 Start the server with: python main.py")
        sys.exit(1)
    
    # Test 2: Create presentation
    p_id = test_create_presentation()
    if not p_id:
        print("\n❌ Failed to create presentation!")
        sys.exit(1)
    
    # Test 3: Start presentation
    if not test_start_presentation(p_id):
        print("\n❌ Failed to start presentation!")
        sys.exit(1)
    
    # Test 4: Monitor status
    final_status = test_check_status(p_id)
    
    # Test 5: Get data
    test_get_data(p_id)
    
    print("\n🎯 Test Results Summary")
    print("=" * 50)
    print(f"📋 Presentation ID: {p_id}")
    print(f"📋 Final Status: {final_status}")
    
    if final_status == "completed":
        print("✅ SUCCESS: Presentation completed successfully!")
        print("🎉 Agent output fixes are working!")
    elif final_status == "failed":
        print("❌ FAILED: Presentation generation failed!")
        print("💡 Check server logs for errors")
    elif final_status == "timeout":
        print("⏰ TIMEOUT: Presentation did not complete in time!")
        print("💡 Agent may be hanging - check the time/event limits")
    else:
        print("⚠️ UNKNOWN: Presentation status unclear!")
        print("💡 Check server logs for more information")
    
    print("\n💡 Next Steps:")
    print("   1. Check server logs for agent execution messages")
    print("   2. Look for 'Stored agent output in agent_outputs_2' messages")
    print("   3. Check for 'Processing event X' messages")
    print("   4. Verify agent completes within 2 minutes or 50 events")

if __name__ == "__main__":
    main()
