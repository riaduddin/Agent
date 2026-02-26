#!/usr/bin/env python3
"""
Test Socket.IO Connection Fix
Tests the complete Socket.IO connection flow with proper parameter handling
"""

import asyncio
import socketio
import json
import os
from datetime import datetime

# Test configuration
SERVER_URL = "http://127.0.0.1:8060"
JWT_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJfaWQiOiI2ODgxYjhjMzE2ZjViODk0MzZkYzBiNzMiLCJzdWIiOiI2ODgxYjhjMzE2ZjViODk0MzZkYzBiNzMiLCJlbWFpbCI6Im1obWFoZWRpMDAwQGdtYWlsLmNvbSIsInBhY2thZ2UiOiJ2YWx1ZV9wbGFuIiwiaXNfdmVyaWZpZWQiOnRydWUsInJvbGUiOiJ1c2VyIiwiaWF0IjoxNzYxMzg3MzUxfQ.wxPfF0xdNlE4oMEF63JTqvdbPq1rYNdQCgB4vKJ9LcY"

async def test_socketio_connection():
    """Test Socket.IO connection with query parameters"""
    print("🧪 Testing Socket.IO Connection Fix")
    print("=" * 50)
    
    # Create Socket.IO client
    sio = socketio.AsyncClient()
    
    # Test presentation ID
    presentation_id = "68fdbcd712eb41c2590d1378"
    
    # Connection event handlers
    @sio.event
    async def connect():
        print("✅ Connected to Socket.IO server")
    
    @sio.event
    async def connected(data):
        print(f"✅ Server welcome: {data}")
    
    @sio.event
    async def agent_output(message):
        print(f"📡 Received agent output: {message.get('type', 'unknown')} - {message.get('author', 'unknown')}")
        if 'html_content' in message:
            print(f"   📄 HTML Content: {message['html_content'][:100]}...")
        if 'thinking' in message:
            print(f"   💭 Thinking: {message['thinking'][:100]}...")
        if 'topic' in message:
            print(f"   🎯 Topic: {message['topic']}")
    
    @sio.event
    async def subscribed(data):
        print(f"✅ Subscribed to presentation: {data}")
    
    @sio.event
    async def error(err):
        print(f"❌ Socket error: {err}")
    
    @sio.event
    async def disconnect():
        print("🔌 Disconnected from Socket.IO server")
    
    try:
        # Connect with query parameters (as the frontend does)
        connection_url = f"{SERVER_URL}?p_id={presentation_id}&token={JWT_TOKEN}"
        print(f"🔗 Connecting to: {connection_url}")
        
        await sio.connect(connection_url)
        
        # Wait for some events
        print("⏳ Waiting for agent output events...")
        await asyncio.sleep(5)
        
        # Disconnect
        await sio.disconnect()
        print("✅ Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if sio.connected:
            await sio.disconnect()

async def test_create_and_start_presentation():
    """Test complete workflow: create presentation, start it, and connect via Socket.IO"""
    print("\n🧪 Testing Complete Workflow")
    print("=" * 50)
    
    import httpx
    
    try:
        # Step 1: Create presentation
        async with httpx.AsyncClient() as client:
            print("📝 Creating presentation...")
            create_response = await client.post(
                f"{SERVER_URL}/create-presentation",
                headers={
                    "Authorization": f"Bearer {JWT_TOKEN}",
                    "Content-Type": "application/json"
                },
                json={
                    "message": "Create a presentation about AI in Healthcare 3 slides",
                    "file_urls": []
                }
            )
            
            if create_response.status_code == 200:
                create_data = create_response.json()
                p_id = create_data.get("p_id")
                print(f"✅ Presentation created: {p_id}")
                
                # Step 2: Start presentation
                print("🚀 Starting presentation...")
                start_response = await client.post(
                    f"{SERVER_URL}/start-presentation/{p_id}",
                    headers={
                        "Authorization": f"Bearer {JWT_TOKEN}",
                        "Content-Type": "application/json"
                    }
                )
                
                if start_response.status_code == 200:
                    start_data = start_response.json()
                    print(f"✅ Presentation started: {start_data.get('status')}")
                    
                    # Step 3: Connect via Socket.IO
                    print("🔌 Connecting via Socket.IO...")
                    await test_socketio_connection_with_p_id(p_id)
                else:
                    print(f"❌ Failed to start presentation: {start_response.status_code} - {start_response.text}")
            else:
                print(f"❌ Failed to create presentation: {create_response.status_code} - {create_response.text}")
                
    except Exception as e:
        print(f"❌ Workflow test failed: {e}")
        import traceback
        traceback.print_exc()

async def test_socketio_connection_with_p_id(p_id):
    """Test Socket.IO connection with specific presentation ID"""
    sio = socketio.AsyncClient()
    
    @sio.event
    async def connect():
        print("✅ Socket.IO connected")
    
    @sio.event
    async def agent_output(message):
        print(f"📡 Agent output: {message.get('type')} - {message.get('author')}")
        if message.get('type') == 'chunk':
            if 'html_content' in message:
                print(f"   📄 HTML: {len(message['html_content'])} chars")
            if 'thinking' in message:
                print(f"   💭 Thinking: {len(message['thinking'])} chars")
    
    try:
        connection_url = f"{SERVER_URL}?p_id={p_id}&token={JWT_TOKEN}"
        await sio.connect(connection_url)
        
        # Wait for events
        await asyncio.sleep(10)
        
        await sio.disconnect()
        print("✅ Socket.IO test completed")
        
    except Exception as e:
        print(f"❌ Socket.IO test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 Starting Socket.IO Connection Fix Tests")
    print("=" * 60)
    
    # Test 1: Basic Socket.IO connection
    asyncio.run(test_socketio_connection())
    
    # Test 2: Complete workflow
    asyncio.run(test_create_and_start_presentation())
    
    print("\n🎉 All tests completed!")

