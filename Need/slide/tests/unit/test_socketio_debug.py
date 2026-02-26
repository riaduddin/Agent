#!/usr/bin/env python3
"""
Debug Socket.IO connection and events.
"""

import socketio
import asyncio
import requests
import time
import json

# Socket.IO client
sio = socketio.AsyncClient()

# Global variables
BASE_URL = "http://127.0.0.1:8060"
JWT_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJfaWQiOiI2ODgwODdlMzA1MTk0OTc2YWVhOTk0MDMiLCJzdWIiOiI2ODgwODdlMzA1MTk0OTc2YWVhOTk0MDMiLCJlbWFpbCI6InJycmlhZHVkZGluQGdtYWlsLmNvbSIsInBhY2thZ2UiOiJ1bmxpbWl0ZWQiLCJpc192ZXJpZmllZCI6dHJ1ZSwicm9sZSI6InVzZXIiLCJpYXQiOjE3NjA5MzM4Njl9.E9cUSlM-uQRrQ48gpH47t7eUN9ioem-A63CCPVJhZnI"

# Track received events
received_events = []
presentation_id = None

@sio.event
async def connect():
    """Called when connected to Socket.IO server"""
    print("✅ Connected to Socket.IO server!")
    print(f"📡 Session ID: {sio.sid}")

@sio.event
async def disconnect():
    """Called when disconnected from Socket.IO server"""
    print("❌ Disconnected from Socket.IO server")

@sio.event
async def agent_output(data):
    """Handle agent output events"""
    print(f"\n📨 Received Agent Output:")
    print(f"   📋 Type: {data.get('type', 'unknown')}")
    print(f"   📋 Author: {data.get('author', 'unknown')}")
    print(f"   📋 P_ID: {data.get('p_id', 'unknown')}")
    print(f"   📋 Timestamp: {data.get('timestamp', 'unknown')}")
    
    # Store the event
    received_events.append(data)
    
    # Show content based on type
    if data.get('type') == 'chunk':
        if 'html_content' in data:
            print(f"   🎨 HTML Content: {data['html_content'][:100]}...")
        elif 'thinking' in data:
            print(f"   💭 Thinking: {data['thinking'][:100]}...")
        elif 'text' in data:
            print(f"   📝 Text: {data['text'][:100]}...")
        elif 'topic' in data:
            print(f"   🎯 Topic: {data['topic']}")
        else:
            print(f"   📄 Content: {str(data)[:100]}...")
    
    print(f"   📊 Total events received: {len(received_events)}")

@sio.event
async def message(data):
    """Handle general message events"""
    print(f"\n📨 Received Message: {data}")

@sio.on('*')
async def catch_all(event, data):
    """Catch all events for debugging"""
    print(f"\n🔍 Received Event: {event}")
    print(f"   📋 Data: {data}")

async def create_presentation():
    """Create a presentation"""
    print("📝 Creating presentation...")
    
    payload = {
        "message": "Create a presentation about Machine Learning in Healthcare. Include sections on diagnostic imaging, drug discovery, and personalized medicine. Make it comprehensive for medical professionals.",
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
        print(f"✅ Presentation created: {data.get('p_id')}")
        return data.get('p_id')
    else:
        print(f"❌ Failed to create presentation: {response.status_code}")
        print(f"📝 Response: {response.text}")
        return None

async def start_presentation(p_id):
    """Start presentation generation"""
    print(f"🚀 Starting presentation: {p_id}")
    
    headers = {
        "Authorization": f"Bearer {JWT_TOKEN}"
    }
    
    response = requests.post(f"{BASE_URL}/start-presentation/{p_id}", 
                           headers=headers, timeout=10)
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Presentation started: {data.get('message')}")
        return True
    else:
        print(f"❌ Failed to start presentation: {response.status_code}")
        print(f"📝 Response: {response.text}")
        return False

async def connect_to_socketio(p_id):
    """Connect to Socket.IO with presentation ID"""
    print(f"🔌 Connecting to Socket.IO for p_id: {p_id}")
    
    try:
        # Connect to Socket.IO with authentication
        await sio.connect(
            f"{BASE_URL}",
            auth={
                "token": JWT_TOKEN
            }
        )
        
        # Wait a moment for connection to establish
        await asyncio.sleep(1)
        
        # Join the presentation room
        await sio.emit('join_presentation', {'p_id': p_id})
        print(f"✅ Joined presentation room: {p_id}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to connect to Socket.IO: {e}")
        return False

async def monitor_presentation(p_id, max_time=120):
    """Monitor presentation for a maximum time"""
    print(f"📊 Monitoring presentation {p_id} for up to {max_time} seconds...")
    print("🔍 Watching for Socket.IO events...")
    
    start_time = time.time()
    
    while time.time() - start_time < max_time:
        # Check presentation status
        headers = {
            "Authorization": f"Bearer {JWT_TOKEN}"
        }
        
        try:
            response = requests.get(f"{BASE_URL}/presentation/{p_id}/status", 
                                  headers=headers, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                status = data.get('status', 'unknown')
                print(f"📋 Status: {status}")
                
                if status in ['completed', 'failed']:
                    print(f"🏁 Final status: {status}")
                    break
                    
        except Exception as e:
            print(f"⚠️ Error checking status: {e}")
        
        # Wait before next check
        await asyncio.sleep(5)
    
    print(f"⏰ Monitoring completed. Total events received: {len(received_events)}")

async def main():
    """Main function"""
    print("🧪 Socket.IO Debug Test")
    print("=" * 50)
    print("This test will help debug Socket.IO connection and events")
    print("=" * 50)
    
    try:
        # Step 1: Create presentation
        p_id = await create_presentation()
        if not p_id:
            print("❌ Failed to create presentation!")
            return
        
        # Step 2: Connect to Socket.IO
        if not await connect_to_socketio(p_id):
            print("❌ Failed to connect to Socket.IO!")
            return
        
        # Step 3: Start presentation
        if not await start_presentation(p_id):
            print("❌ Failed to start presentation!")
            return
        
        # Step 4: Monitor presentation and receive events
        await monitor_presentation(p_id)
        
        # Step 5: Disconnect
        await sio.disconnect()
        
        print("\n🎯 Debug Results Summary")
        print("=" * 50)
        print(f"📋 Presentation ID: {p_id}")
        print(f"📋 Total events received: {len(received_events)}")
        
        if received_events:
            print("\n📊 Event Types Received:")
            event_types = {}
            for event in received_events:
                event_type = event.get('type', 'unknown')
                author = event.get('author', 'unknown')
                key = f"{event_type} ({author})"
                event_types[key] = event_types.get(key, 0) + 1
            
            for event_type, count in event_types.items():
                print(f"   • {event_type}: {count}")
        else:
            print("\n❌ NO EVENTS RECEIVED!")
            print("🔍 Possible issues:")
            print("   1. Socket.IO server not broadcasting events")
            print("   2. Authentication issues")
            print("   3. Room subscription problems")
            print("   4. Agent not generating events")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if sio.connected:
            await sio.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
