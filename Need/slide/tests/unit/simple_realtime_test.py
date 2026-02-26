#!/usr/bin/env python3
"""
Simple real-time test using REST API only.
"""

import requests
import time
import json
from datetime import datetime

# Valid JWT token
VALID_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJfaWQiOiI2ODgwODdlMzA1MTk0OTc2YWVhOTk0MDMiLCJzdWIiOiI2ODgwODdlMzA1MTk0OTc2YWVhOTk0MDMiLCJlbWFpbCI6InJycmlhZHVkZGluQGdtYWlsLmNvbSIsInBhY2thZ2UiOiJ1bmxpbWl0ZWQiLCJpc192ZXJpZmllZCI6dHJ1ZSwicm9sZSI6InVzZXIiLCJpYXQiOjE3NjEzNzczMzAsImV4cCI6MTc2MTQ2MzczMH0.dUeYJu4jNbaSfN8jRloiFiRTie1WkJ1prWtMe-_RQLg"

def create_presentation():
    """Create a presentation"""
    print("📝 Creating presentation...")
    print(f"🕐 Started at: {datetime.now().strftime('%H:%M:%S')}")
    
    try:
        response = requests.post(
            "http://127.0.0.1:8060/create-presentation",
            json={
                "message": "Create a presentation about AI in Healthcare. Include sections on diagnostic imaging, drug discovery, and personalized medicine. Make it comprehensive for medical professionals.",
                "file_urls": []
            },
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {VALID_TOKEN}"
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            p_id = data["p_id"]
            print(f"✅ Presentation created successfully!")
            print(f"   📋 P_ID: {p_id}")
            print(f"   📋 Status: {data.get('status', 'unknown')}")
            print(f"   📋 Message: {data.get('message', 'N/A')}")
            return p_id
        else:
            print(f"❌ Failed to create presentation: {response.status_code}")
            print(f"📝 Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Error creating presentation: {e}")
        return None

def start_presentation(p_id):
    """Start presentation generation"""
    print(f"\n🚀 Starting presentation: {p_id}")
    print(f"🕐 Started at: {datetime.now().strftime('%H:%M:%S')}")
    
    try:
        response = requests.post(
            f"http://127.0.0.1:8060/start-presentation/{p_id}",
            headers={
                "Authorization": f"Bearer {VALID_TOKEN}"
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Presentation started successfully!")
            print(f"   📋 Message: {data.get('message', 'N/A')}")
            print(f"   📋 Status: {data.get('status', 'unknown')}")
            return True
        else:
            print(f"❌ Failed to start presentation: {response.status_code}")
            print(f"📝 Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error starting presentation: {e}")
        return False

def monitor_presentation_status(p_id):
    """Monitor presentation status in real-time"""
    print(f"\n📊 Monitoring presentation status: {p_id}")
    print(f"🕐 Started monitoring at: {datetime.now().strftime('%H:%M:%S')}")
    
    start_time = time.time()
    max_monitor_time = 120  # 2 minutes
    check_count = 0
    
    while time.time() - start_time < max_monitor_time:
        check_count += 1
        elapsed = time.time() - start_time
        
        try:
            response = requests.get(
                f"http://127.0.0.1:8060/presentation/{p_id}/status",
                headers={
                    "Authorization": f"Bearer {VALID_TOKEN}"
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                status = data.get('status', 'unknown')
                title = data.get('title', 'N/A')
                total_slides = data.get('total_slides', 0)
                created_at = data.get('created_at', 'N/A')
                updated_at = data.get('updated_at', 'N/A')
                
                print(f"📋 Check #{check_count} ({elapsed:.1f}s): Status={status} | Title={title} | Slides={total_slides}")
                print(f"   📅 Created: {created_at} | Updated: {updated_at}")
                
                if status in ['completed', 'failed']:
                    print(f"🏁 Final status: {status}")
                    return data
            else:
                print(f"⚠️ Status check #{check_count} failed: {response.status_code}")
                
        except Exception as e:
            print(f"⚠️ Error checking status #{check_count}: {e}")
        
        # Wait before next check
        time.sleep(5)
    
    print(f"⏰ Monitoring completed after {time.time() - start_time:.1f} seconds")
    return None

def get_presentation_data(p_id):
    """Get final presentation data"""
    print(f"\n📄 Getting presentation data: {p_id}")
    
    try:
        response = requests.get(
            f"http://127.0.0.1:8060/presentation/{p_id}/data",
            headers={
                "Authorization": f"Bearer {VALID_TOKEN}"
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Presentation data retrieved!")
            print(f"   📋 Status: {data.get('status', 'unknown')}")
            print(f"   📋 Title: {data.get('title', 'N/A')}")
            print(f"   📋 Total Slides: {data.get('total_slides', 0)}")
            
            slides = data.get('slides', [])
            if slides:
                print(f"   📋 Slides:")
                for i, slide in enumerate(slides[:3]):  # Show first 3 slides
                    print(f"      {i+1}. {slide.get('title', 'Untitled')}")
                    if slide.get('content'):
                        print(f"         Content: {slide['content'][:100]}...")
                if len(slides) > 3:
                    print(f"      ... and {len(slides) - 3} more slides")
            
            return data
        else:
            print(f"❌ Failed to get presentation data: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Error getting presentation data: {e}")
        return None

def main():
    """Main function - Complete real-time workflow using REST API only"""
    print("🧪 Simple Real-Time Data Monitor (REST API Only)")
    print("=" * 60)
    print("This will show you real-time data from create-presentation and start-presentation")
    print("using REST API polling (no Socket.IO required)")
    print("=" * 60)
    
    try:
        # Step 1: Create presentation
        p_id = create_presentation()
        if not p_id:
            print("❌ Failed to create presentation!")
            return
        
        # Step 2: Start presentation
        success = start_presentation(p_id)
        if not success:
            print("❌ Failed to start presentation!")
            return
        
        # Step 3: Monitor in real-time
        print(f"\n📊 Real-time monitoring started...")
        print(f"🕐 Started at: {datetime.now().strftime('%H:%M:%S')}")
        print(f"📋 Polling status every 5 seconds...")
        
        # Monitor for status changes
        final_status = monitor_presentation_status(p_id)
        
        # Step 4: Get final data
        final_data = get_presentation_data(p_id)
        
        # Summary
        print(f"\n🎯 Real-Time Data Summary")
        print("=" * 60)
        print(f"📋 Presentation ID: {p_id}")
        print(f"📋 Monitoring method: REST API polling")
        print(f"📋 Final status: {final_status.get('status', 'unknown') if final_status else 'unknown'}")
        
        if final_data:
            print(f"\n📄 Final Presentation Data:")
            print(f"   Status: {final_data.get('status', 'unknown')}")
            print(f"   Title: {final_data.get('title', 'N/A')}")
            print(f"   Total Slides: {final_data.get('total_slides', 0)}")
            
            slides = final_data.get('slides', [])
            if slides:
                print(f"\n📋 Generated Slides:")
                for i, slide in enumerate(slides):
                    print(f"   {i+1}. {slide.get('title', 'Untitled')}")
                    if slide.get('content'):
                        print(f"      Content: {slide['content'][:150]}...")
                    if slide.get('html_content'):
                        print(f"      HTML: {slide['html_content'][:100]}...")
        else:
            print(f"\n❌ No final data retrieved!")
            print(f"🔍 This might indicate:")
            print(f"   • Presentation generation failed")
            print(f"   • Agent execution issues")
            print(f"   • Database connection problems")
        
    except Exception as e:
        print(f"❌ Error in main workflow: {e}")
        print(f"🔍 Error type: {type(e).__name__}")

if __name__ == "__main__":
    main()
