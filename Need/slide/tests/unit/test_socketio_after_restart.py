#!/usr/bin/env python3
"""
Test Socket.IO after server restart.
"""

import socketio
import asyncio
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Socket.IO client
sio = socketio.AsyncClient(
    logger=True,
    engineio_logger=True
)

# Valid JWT token
VALID_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJfaWQiOiI2ODgwODdlMzA1MTk0OTc2YWVhOTk0MDMiLCJzdWIiOiI2ODgwODdlMzA1MTk0OTc2YWVhOTk0MDMiLCJlbWFpbCI6InJycmlhZHVkZGluQGdtYWlsLmNvbSIsInBhY2thZ2UiOiJ1bmxpbWl0ZWQiLCJpc192ZXJpZmllZCI6dHJ1ZSwicm9sZSI6InVzZXIiLCJpYXQiOjE3NjEzNzczMzAsImV4cCI6MTc2MTQ2MzczMH0.dUeYJu4jNbaSfN8jRloiFiRTie1WkJ1prWtMe-_RQLg"

# Track received events
received_events = []

@sio.event
async def connect():
    print("✅ Connected to Socket.IO!")
    print(f"📡 Session ID: {sio.sid}")

@sio.event
async def disconnect():
    print("❌ Disconnected from Socket.IO")

@sio.event
async def agent_output(data):
    print(f"\n📨 Agent Output Received:")
    print(f"   📋 Type: {data.get('type', 'unknown')}")
    print(f"   📋 Author: {data.get('author', 'unknown')}")
    print(f"   📋 P_ID: {data.get('p_id', 'unknown')}")
    print(f"   📋 Timestamp: {data.get('timestamp', 'unknown')}")
    
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
    
    received_events.append(data)
    print(f"   📊 Total events received: {len(received_events)}")

@sio.event
async def message(data):
    print(f"\n📨 Message: {data}")

@sio.on('*')
async def catch_all(event, data):
    print(f"\n🔍 Event: {event} -> {data}")

async def test_socketio_after_restart():
    """Test Socket.IO after server restart"""
    print("🧪 Testing Socket.IO After Server Restart")
    print("=" * 50)
    print("Make sure you have restarted the server with: python main.py")
    print("=" * 50)
    
    try:
        # Create a presentation first
        print("📝 Creating presentation...")
        response = requests.post(
            "http://127.0.0.1:8060/create-presentation",
            json={
                "message": "Create a presentation about AI in Healthcare",
                "file_urls": []
            },
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {VALID_TOKEN}"
            }
        )
        
        if response.status_code == 200:
            p_id = response.json()["p_id"]
            print(f"✅ Presentation created: {p_id}")
        else:
            print(f"❌ Failed to create presentation: {response.status_code}")
            return
        
        # Test Socket.IO connection
        print(f"🔌 Connecting to Socket.IO with p_id={p_id}...")
        await sio.connect(f'http://127.0.0.1:8060?p_id={p_id}&token={VALID_TOKEN}')
        print("✅ Socket.IO connection successful!")
        
        # Wait a moment
        await asyncio.sleep(2)
        
        # Start the presentation
        print("🚀 Starting presentation...")
        start_response = requests.post(
            f"http://127.0.0.1:8060/start-presentation/{p_id}",
            headers={
                "Authorization": f"Bearer {VALID_TOKEN}"
            }
        )
        
        if start_response.status_code == 200:
            print("✅ Presentation started!")
            
            # Wait for events
            print("📊 Waiting for events...")
            await asyncio.sleep(20)
        else:
            print(f"❌ Failed to start presentation: {start_response.status_code}")
        
        await sio.disconnect()
        print("✅ Disconnected successfully!")
        
        # Summary
        print(f"\n🎯 Test Results:")
        print(f"   📋 Presentation ID: {p_id}")
        print(f"   📋 Total events received: {len(received_events)}")
        
        if received_events:
            print(f"\n📊 Event Types:")
            event_types = {}
            for event in received_events:
                event_type = event.get('type', 'unknown')
                author = event.get('author', 'unknown')
                key = f"{event_type} ({author})"
                event_types[key] = event_types.get(key, 0) + 1
            
            for event_type, count in event_types.items():
                print(f"   • {event_type}: {count}")
            
            print(f"\n🎉 Socket.IO is working correctly!")
        else:
            print(f"\n⚠️ No events received - this might indicate agent output issues")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print(f"🔍 Error type: {type(e).__name__}")
        
        if "Connection rejected by server" in str(e):
            print(f"\n💡 The server still needs to be restarted!")
            print(f"   Stop the server (Ctrl+C) and restart it with: python main.py")

async def main():
    """Main function"""
    await test_socketio_after_restart()

if __name__ == "__main__":
    asyncio.run(main())
