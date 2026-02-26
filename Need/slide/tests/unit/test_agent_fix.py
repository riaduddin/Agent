#!/usr/bin/env python3
"""
Test the agent fix for Invalid format specifier error.
"""

import requests
import time
from datetime import datetime

# Valid JWT token
TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJfaWQiOiI2ODgwODdlMzA1MTk0OTc2YWVhOTk0MDMiLCJzdWIiOiI2ODgwODdlMzA1MTk0OTc2YWVhOTk0MDMiLCJlbWFpbCI6InJycmlhZHVkZGluQGdtYWlsLmNvbSIsInBhY2thZ2UiOiJ1bmxpbWl0ZWQiLCJpc192ZXJpZmllZCI6dHJ1ZSwicm9sZSI6InVzZXIiLCJpYXQiOjE3NjEzNzczMzAsImV4cCI6MTc2MTQ2MzczMH0.dUeYJu4jNbaSfN8jRloiFiRTie1WkJ1prWtMe-_RQLg"

def test_agent_fix():
    """Test if the agent fix resolves the Invalid format specifier error"""
    print("🧪 Testing Agent Fix")
    print("=" * 50)
    
    try:
        # Create presentation
        print("📝 Creating presentation...")
        response = requests.post(
            "http://127.0.0.1:8060/create-presentation",
            json={
                "message": "Create a presentation about AI in Healthcare",
                "file_urls": []
            },
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {TOKEN}"
            }
        )
        
        if response.status_code == 200:
            p_id = response.json()["p_id"]
            print(f"✅ Presentation created: {p_id}")
        else:
            print(f"❌ Failed to create presentation: {response.status_code}")
            return
        
        # Start presentation
        print("🚀 Starting presentation...")
        start_response = requests.post(
            f"http://127.0.0.1:8060/start-presentation/{p_id}",
            headers={
                "Authorization": f"Bearer {TOKEN}"
            }
        )
        
        if start_response.status_code == 200:
            print("✅ Presentation started!")
        else:
            print(f"❌ Failed to start presentation: {start_response.status_code}")
            return
        
        # Monitor status
        print("📊 Monitoring status...")
        for i in range(12):  # Monitor for 1 minute
            try:
                status_response = requests.get(
                    f"http://127.0.0.1:8060/presentation/{p_id}/status",
                    headers={
                        "Authorization": f"Bearer {TOKEN}"
                    }
                )
                
                if status_response.status_code == 200:
                    data = status_response.json()
                    status = data.get('status', 'unknown')
                    title = data.get('title', 'N/A')
                    slides = data.get('total_slides', 0)
                    
                    print(f"📋 Check {i+1}: Status={status} | Title={title} | Slides={slides}")
                    
                    if status in ['completed', 'failed']:
                        print(f"🏁 Final status: {status}")
                        break
                else:
                    print(f"⚠️ Status check failed: {status_response.status_code}")
                    
            except Exception as e:
                print(f"⚠️ Error checking status: {e}")
            
            time.sleep(5)
        
        # Get final data
        print("\n📄 Getting final data...")
        try:
            data_response = requests.get(
                f"http://127.0.0.1:8060/presentation/{p_id}/data",
                headers={
                    "Authorization": f"Bearer {TOKEN}"
                }
            )
            
            if data_response.status_code == 200:
                data = data_response.json()
                print(f"✅ Final data retrieved!")
                print(f"   Status: {data.get('status', 'unknown')}")
                print(f"   Title: {data.get('title', 'N/A')}")
                print(f"   Total Slides: {data.get('total_slides', 0)}")
                
                slides = data.get('slides', [])
                if slides:
                    print(f"   Generated {len(slides)} slides!")
                    for i, slide in enumerate(slides[:3]):
                        print(f"     {i+1}. {slide.get('title', 'Untitled')}")
                else:
                    print(f"   No slides generated")
            else:
                print(f"❌ Failed to get final data: {data_response.status_code}")
                
        except Exception as e:
            print(f"❌ Error getting final data: {e}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_agent_fix()