#!/usr/bin/env python3
"""
Debug script to test agent execution and see what's happening.
"""

import asyncio
import os
import sys
import requests
import time
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

async def test_agent_execution_debug():
    """Test agent execution with detailed debugging"""
    print("🧪 Testing Agent Execution Debug")
    print("=" * 50)
    
    # Get configuration
    base_url = os.getenv("BASE_URL", "http://localhost:8060")
    jwt_token = os.getenv("JWT_TOKEN", "your_jwt_token_here")
    
    if jwt_token == "your_jwt_token_here":
        print("⚠️ Please set JWT_TOKEN in your .env file or environment")
        print("   Example: JWT_TOKEN=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...")
        return False
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {jwt_token}"
    }
    
    print(f"🔗 Base URL: {base_url}")
    print(f"🔑 JWT Token: {jwt_token[:20]}...")
    print()
    
    # Step 1: Create presentation
    print("📝 Step 1: Creating presentation...")
    create_payload = {
        "message": "Create a presentation about AI in Healthcare with 3 slides",
        "file_urls": []
    }
    
    try:
        response = requests.post(
            f"{base_url}/create-presentation",
            headers=headers,
            json=create_payload,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            p_id = data.get("p_id")
            print(f"✅ Presentation created: {p_id}")
        else:
            print(f"❌ Failed to create presentation: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error creating presentation: {e}")
        return False
    
    # Step 2: Check initial status
    print(f"\n📊 Step 2: Checking initial status...")
    try:
        response = requests.get(
            f"{base_url}/presentation/{p_id}/status",
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Initial status: {data.get('status')}")
        else:
            print(f"❌ Failed to get status: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Error getting status: {e}")
    
    # Step 3: Start presentation
    print(f"\n🚀 Step 3: Starting presentation...")
    try:
        response = requests.post(
            f"{base_url}/start-presentation/{p_id}",
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Presentation started: {data.get('status')}")
        else:
            print(f"❌ Failed to start presentation: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error starting presentation: {e}")
        return False
    
    # Step 4: Monitor status changes
    print(f"\n📊 Step 4: Monitoring status changes...")
    print("⏳ Waiting for status changes (checking every 2 seconds for 60 seconds)...")
    
    start_time = time.time()
    last_status = None
    
    while time.time() - start_time < 60:  # Monitor for 60 seconds
        try:
            response = requests.get(
                f"{base_url}/presentation/{p_id}/status",
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                current_status = data.get('status')
                
                if current_status != last_status:
                    print(f"📊 Status changed: {last_status} → {current_status} at {datetime.now().strftime('%H:%M:%S')}")
                    last_status = current_status
                    
                    if current_status in ['completed', 'failed']:
                        print(f"🏁 Final status: {current_status}")
                        break
                else:
                    print(f"⏳ Status: {current_status} (waiting...)")
            else:
                print(f"❌ Failed to get status: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error getting status: {e}")
        
        await asyncio.sleep(2)  # Wait 2 seconds between checks
    
    # Step 5: Get final data
    print(f"\n📄 Step 5: Getting final presentation data...")
    try:
        response = requests.get(
            f"{base_url}/presentation/{p_id}/data",
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            slides = data.get('slides', [])
            print(f"✅ Final data retrieved: {len(slides)} slides")
            print(f"📋 Final status: {data.get('status')}")
            print(f"📋 Title: {data.get('title', 'No title')}")
            
            if slides:
                print(f"📄 First slide preview: {slides[0].get('title', 'No title')[:50]}...")
            else:
                print("❌ No slides generated!")
        else:
            print(f"❌ Failed to get data: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Error getting data: {e}")
    
    print(f"\n🎉 Agent execution debug completed!")
    return True

async def main():
    """Main function"""
    print("Agent Execution Debug Test")
    print("=" * 50)
    print("This test monitors agent execution and shows what's happening")
    print("=" * 50)
    
    success = await test_agent_execution_debug()
    
    if success:
        print("\n✅ SUCCESS!")
        print("🎉 Agent execution debug completed!")
        print("\n📋 What this test shows:")
        print("   • Presentation creation process")
        print("   • Status changes over time")
        print("   • Final presentation data")
        print("   • Whether slides are actually generated")
        print("\n🔍 If the status goes to 'completed' immediately:")
        print("   • Check server logs for agent execution errors")
        print("   • Verify database session service is working")
        print("   • Check if the agent is actually running")
        print("   • Look for any silent failures in the agent loop")
    else:
        print("\n❌ FAILED!")
        print("🔧 There are issues with the agent execution")
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
