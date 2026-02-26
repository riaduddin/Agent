#!/usr/bin/env python3
"""
Test Socket.IO connection like the frontend client code
"""

import socketio
import requests
import json
import time

# Configuration
SERVER_URL = "http://127.0.0.1:8060"
JWT_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJfaWQiOiI2ODgxYjhjMzE2ZjViODk0MzZkYzBiNzMiLCJzdWIiOiI2ODgxYjhjMzE2ZjViODk0MzZkYzBiNzMiLCJlbWFpbCI6Im1obWFoZWRpMDAwQGdtYWlsLmNvbSIsInBhY2thZ2UiOiJ2YWx1ZV9wbGFuIiwiaXNfdmVyaWZpZWQiOnRydWUsInJvbGUiOiJ1c2VyIiwiaWF0IjoxNzYxMzg3MzUxfQ.wxPfF0xdNlE4oMEF63JTqvdbPq1rYNdQCgB4vKJ9LcY"

def create_presentation():
    """Create a presentation"""
    print("🚀 Creating presentation...")
    
    response = requests.post(
        f"{SERVER_URL}/create-presentation",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {JWT_TOKEN}"
        },
        json={
            "message": "Create a presentation about Bangladesh Software Industry for 5 slides",
            "file_urls": []
        }
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Presentation created: {data['p_id']}")
        return data['p_id']
    else:
        print(f"❌ Failed to create presentation: {response.text}")
        return None

def start_presentation(p_id):
    """Start presentation"""
    print(f"🚀 Starting presentation: {p_id}")
    
    response = requests.post(
        f"{SERVER_URL}/start-presentation/{p_id}",
        headers={
            "Authorization": f"Bearer {JWT_TOKEN}"
        }
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Presentation started: {data['message']}")
        return True
    else:
        print(f"❌ Failed to start presentation: {response.text}")
        return False

def test_socketio_connection(p_id):
    """Test Socket.IO connection like frontend"""
    print(f"🔌 Connecting to Socket.IO for p_id: {p_id}")
    
    # Create socket.io client (like frontend)
    socket = socketio.Client()
    
    @socket.event
    def connect():
        print("✅ Socket connected:", socket.id)
    
    @socket.event
    def connected(payload):
        print("✅ Server welcome:", payload)
    
    @socket.event
    def agent_output(message):
        print("🎯 Agent Output:", json.dumps(message, indent=2))
    
    @socket.event
    def subscribed(data):
        print("✅ Subscribed to p_id:", data)
    
    @socket.event
    def error(err):
        print("❌ Socket error:", err)
    
    @socket.event
    def disconnect(reason):
        print("❌ Socket disconnected:", reason)
    
    try:
        # Connect like frontend (using auth parameter)
        socket.connect(
            SERVER_URL,
            auth={
                "p_id": p_id,
                "token": JWT_TOKEN
            },
            transports=["websocket"]
        )
        
        print("✅ Connected successfully!")
        
        # Wait for events
        print("🎯 Waiting for agent output events...")
        time.sleep(10)  # Wait 10 seconds for events
        
        socket.disconnect()
        return True
        
    except Exception as e:
        print(f"❌ Socket.IO connection failed: {e}")
        return False

def main():
    print("🧪 Frontend Socket.IO Test")
    print("=" * 40)
    
    # Step 1: Create presentation
    p_id = create_presentation()
    if not p_id:
        return
    
    # Step 2: Start presentation
    if not start_presentation(p_id):
        return
    
    # Step 3: Connect to Socket.IO
    test_socketio_connection(p_id)
    
    print("✅ Test completed!")

if __name__ == "__main__":
    main()
