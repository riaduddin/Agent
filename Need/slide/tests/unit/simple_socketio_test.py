#!/usr/bin/env python3
"""
Simple Socket.IO connection test
"""

import socketio
import requests
import json

# Configuration
SERVER_URL = "http://127.0.0.1:8060"
JWT_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJfaWQiOiI2ODgwODdlMzA1MTk0OTc2YWVhOTk0MDMiLCJzdWIiOiI2ODgwODdlMzA1MTk0OTc2YWVhOTk0MDMiLCJlbWFpbCI6InJycmlhZHVkZGluQGdtYWlsLmNvbSIsInBhY2thZ2UiOiJ1bmxpbWl0ZWQiLCJpc192ZXJpZmllZCI6dHJ1ZSwicm9sZSI6InVzZXIiLCJpYXQiOjE3NjE0MzQ0NzAsImV4cCI6MTc2MTUyMDg3MH0.atHXa2dwU-GqNnNNC75afrANpkVMDiTrp6EISIa4TWk"

def test_server_health():
    """Test if server is running"""
    try:
        response = requests.get(f"{SERVER_URL}/", timeout=5)
        print(f"✅ Server health check: {response.status_code}")
        return True
    except Exception as e:
        print(f"❌ Server health check failed: {e}")
        return False

def create_presentation():
    """Create a test presentation"""
    try:
        response = requests.post(
            f"{SERVER_URL}/create-presentation",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {JWT_TOKEN}"
            },
            json={
                "message": "Create a simple presentation about AI",
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
    except Exception as e:
        print(f"❌ Error creating presentation: {e}")
        return None

def test_socketio_connection(p_id):
    """Test Socket.IO connection"""
    sio = socketio.Client()
    
    @sio.event
    def connect():
        print("✅ Socket.IO connected!")
    
    @sio.event
    def disconnect():
        print("❌ Socket.IO disconnected!")
    
    @sio.event
    def agent_output(data):
        print(f"🎯 Agent Output: {json.dumps(data, indent=2)}")
    
    @sio.event
    def message(data):
        print(f"📨 Message: {json.dumps(data, indent=2)}")
    
    @sio.event
    def connect_error(data):
        print(f"❌ Connection error: {data}")
    
    try:
        # Try different connection methods
        print("🔌 Attempting Socket.IO connection...")
        
        # Method 1: With query parameters
        connection_url = f"{SERVER_URL}?p_id={p_id}&token={JWT_TOKEN}"
        print(f"   Trying: {connection_url}")
        
        sio.connect(connection_url)
        
        if sio.connected:
            print("✅ Connected successfully!")
            
            # Wait for a few seconds to see if we get any events
            import time
            time.sleep(5)
            
            sio.disconnect()
            return True
        else:
            print("❌ Connection failed")
            return False
            
    except Exception as e:
        print(f"❌ Socket.IO connection error: {e}")
        return False

def main():
    print("🧪 Simple Socket.IO Test")
    print("=" * 30)
    
    # Step 1: Check server health
    if not test_server_health():
        print("❌ Server is not running. Please start the server first.")
        return
    
    # Step 2: Create presentation
    p_id = create_presentation()
    if not p_id:
        print("❌ Failed to create presentation")
        return
    
    # Step 3: Test Socket.IO connection
    if test_socketio_connection(p_id):
        print("✅ Socket.IO test completed successfully!")
    else:
        print("❌ Socket.IO test failed")

if __name__ == "__main__":
    main()
