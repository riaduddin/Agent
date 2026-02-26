#!/usr/bin/env python3
"""
Fixed Socket.IO connection test.
"""

import socketio
import asyncio
import requests

# Socket.IO client
sio = socketio.AsyncClient()

@sio.event
async def connect():
    print("✅ Connected to Socket.IO!")
    print(f"📡 Session ID: {sio.sid}")

@sio.event
async def disconnect():
    print("❌ Disconnected from Socket.IO")

@sio.event
async def agent_output(data):
    print(f"📨 Agent Output: {data}")

@sio.event
async def message(data):
    print(f"📨 Message: {data}")

@sio.on('*')
async def catch_all(event, data):
    print(f"🔍 Event: {event} -> {data}")

async def test_connection_with_params():
    """Test Socket.IO connection with query parameters"""
    print("🧪 Testing Socket.IO Connection with Query Parameters")
    print("=" * 50)
    
    try:
        # Test connection with query parameters (as expected by server)
        print("1. Testing connection with query parameters...")
        
        # Create a test presentation first
        print("📝 Creating test presentation...")
        response = requests.post(
            "http://127.0.0.1:8060/create-presentation",
            json={
                "message": "Create a presentation about AI in Healthcare",
                "file_urls": []
            },
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJfaWQiOiI2ODgwODdlMzA1MTk0OTc2YWVhOTk0MDMiLCJzdWIiOiI2ODgwODdlMzA1MTk0OTc2YWVhOTk0MDMiLCJlbWFpbCI6InJycmlhZHVkZGluQGdtYWlsLmNvbSIsInBhY2thZ2UiOiJ1bmxpbWl0ZWQiLCJpc192ZXJpZmllZCI6dHJ1ZSwicm9sZSI6InVzZXIiLCJpYXQiOjE3NjA5MzM4Njl9.E9cUSlM-uQRrQ48gpH47t7eUN9ioem-A63CCPVJhZnI"
            }
        )
        
        if response.status_code == 200:
            p_id = response.json()["p_id"]
            print(f"✅ Presentation created: {p_id}")
        else:
            print(f"❌ Failed to create presentation: {response.status_code}")
            return
        
        # Connect with query parameters
        token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJfaWQiOiI2ODgwODdlMzA1MTk0OTc2YWVhOTk0MDMiLCJzdWIiOiI2ODgwODdlMzA1MTk0OTc2YWVhOTk0MDMiLCJlbWFpbCI6InJycmlhZHVkZGluQGdtYWlsLmNvbSIsInBhY2thZ2UiOiJ1bmxpbWl0ZWQiLCJpc192ZXJpZmllZCI6dHJ1ZSwicm9sZSI6InVzZXIiLCJpYXQiOjE3NjA5MzM4Njl9.E9cUSlM-uQRrQ48gpH47t7eUN9ioem-A63CCPVJhZnI"
        
        # Connect with query parameters
        await sio.connect(f'http://127.0.0.1:8060?p_id={p_id}&token={token}')
        print("✅ Connection with query parameters successful!")
        
        # Wait a moment
        await asyncio.sleep(2)
        
        # Start the presentation
        print("🚀 Starting presentation...")
        start_response = requests.post(
            f"http://127.0.0.1:8060/start-presentation/{p_id}",
            headers={
                "Authorization": f"Bearer {token}"
            }
        )
        
        if start_response.status_code == 200:
            print("✅ Presentation started!")
        else:
            print(f"❌ Failed to start presentation: {start_response.status_code}")
        
        # Wait for events
        print("📊 Waiting for events...")
        await asyncio.sleep(10)
        
        await sio.disconnect()
        print("✅ Disconnected successfully!")
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print(f"🔍 Error type: {type(e).__name__}")

async def main():
    """Main function"""
    await test_connection_with_params()

if __name__ == "__main__":
    asyncio.run(main())
