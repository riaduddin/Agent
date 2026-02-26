#!/usr/bin/env python3
"""
Socket.IO connection test with correct URL.
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

async def test_connection_correct_url():
    """Test Socket.IO connection with correct URL"""
    print("🧪 Testing Socket.IO Connection with Correct URL")
    print("=" * 50)
    
    try:
        # Test connection with correct URL (including /socket.io path)
        print("1. Testing connection with correct URL...")
        
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
        
        # Connect with correct URL
        token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJfaWQiOiI2ODgwODdlMzA1MTk0OTc2YWVhOTk0MDMiLCJzdWIiOiI2ODgwODdlMzA1MTk0OTc2YWVhOTk0MDMiLCJlbWFpbCI6InJycmlhZHVkZGluQGdtYWlsLmNvbSIsInBhY2thZ2UiOiJ1bmxpbWl0ZWQiLCJpc192ZXJpZmllZCI6dHJ1ZSwicm9sZSI6InVzZXIiLCJpYXQiOjE3NjA5MzM4Njl9.E9cUSlM-uQRrQ48gpH47t7eUN9ioem-A63CCPVJhZnI"
        
        # Try different connection methods
        print("\n2. Testing connection methods...")
        
        # Method 1: Direct URL with query params
        try:
            await sio.connect(f'http://127.0.0.1:8060?p_id={p_id}&token={token}')
            print("✅ Method 1 successful!")
            await sio.disconnect()
        except Exception as e:
            print(f"❌ Method 1 failed: {e}")
        
        # Method 2: URL with /socket.io path
        try:
            await sio.connect(f'http://127.0.0.1:8060/socket.io/?p_id={p_id}&token={token}')
            print("✅ Method 2 successful!")
            await sio.disconnect()
        except Exception as e:
            print(f"❌ Method 2 failed: {e}")
        
        # Method 3: URL with auth object
        try:
            await sio.connect('http://127.0.0.1:8060', auth={'p_id': p_id, 'token': token})
            print("✅ Method 3 successful!")
            await sio.disconnect()
        except Exception as e:
            print(f"❌ Method 3 failed: {e}")
        
        # Method 4: Simple connection
        try:
            await sio.connect('http://127.0.0.1:8060')
            print("✅ Method 4 successful!")
            await sio.disconnect()
        except Exception as e:
            print(f"❌ Method 4 failed: {e}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print(f"🔍 Error type: {type(e).__name__}")

async def main():
    """Main function"""
    await test_connection_correct_url()

if __name__ == "__main__":
    asyncio.run(main())
