#!/usr/bin/env python3
"""
Test if Socket.IO is working properly.
"""

import socketio
import asyncio
import requests
import os
from dotenv import load_dotenv
import time

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

async def test_socketio_basic_connection():
    """Test basic Socket.IO connection without authentication"""
    print("🧪 Test 1: Basic Socket.IO Connection")
    print("=" * 50)
    
    try:
        # Try to connect without authentication
        await sio.connect('http://127.0.0.1:8060')
        print("✅ Basic connection successful!")
        await sio.disconnect()
        return True
    except Exception as e:
        print(f"❌ Basic connection failed: {e}")
        return False

async def test_socketio_with_auth():
    """Test Socket.IO connection with authentication"""
    print("\n🧪 Test 2: Socket.IO Connection with Authentication")
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
            return False
        
        # Try to connect with authentication
        print(f"🔌 Connecting with p_id={p_id} and token...")
        await sio.connect(f'http://127.0.0.1:8060?p_id={p_id}&token={VALID_TOKEN}')
        print("✅ Authenticated connection successful!")
        
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
            await asyncio.sleep(15)
        else:
            print(f"❌ Failed to start presentation: {start_response.status_code}")
        
        await sio.disconnect()
        print("✅ Disconnected successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Authenticated connection failed: {e}")
        print(f"🔍 Error type: {type(e).__name__}")
        return False

async def test_socketio_endpoints():
    """Test Socket.IO endpoints directly"""
    print("\n🧪 Test 3: Socket.IO Endpoints")
    print("=" * 50)
    
    try:
        # Test Socket.IO polling endpoint
        print("1. Testing Socket.IO polling endpoint...")
        response = requests.get("http://127.0.0.1:8060/socket.io/?EIO=4&transport=polling")
        print(f"📋 Status: {response.status_code}")
        print(f"📋 Response: {response.text[:100]}...")
        
        if response.status_code == 200:
            print("✅ Socket.IO polling endpoint working!")
        else:
            print("❌ Socket.IO polling endpoint not working!")
        
        # Test Socket.IO with parameters
        print("\n2. Testing Socket.IO with parameters...")
        response = requests.get(f"http://127.0.0.1:8060/socket.io/?p_id=test&token={VALID_TOKEN}&transport=polling&EIO=4")
        print(f"📋 Status: {response.status_code}")
        print(f"📋 Response: {response.text[:100]}...")
        
        if response.status_code == 200:
            print("✅ Socket.IO with parameters working!")
        else:
            print("❌ Socket.IO with parameters not working!")
        
        return True
        
    except Exception as e:
        print(f"❌ Socket.IO endpoints test failed: {e}")
        return False

async def test_server_health():
    """Test server health"""
    print("\n🧪 Test 4: Server Health Check")
    print("=" * 50)
    
    try:
        # Test basic server health
        response = requests.get("http://127.0.0.1:8060/")
        print(f"📋 Health check status: {response.status_code}")
        print(f"📋 Response: {response.text}")
        
        if response.status_code == 200:
            print("✅ Server is running!")
        else:
            print("❌ Server health check failed!")
        
        # Test create presentation endpoint
        print("\n2. Testing create presentation endpoint...")
        response = requests.post(
            "http://127.0.0.1:8060/create-presentation",
            json={
                "message": "Test presentation",
                "file_urls": []
            },
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {VALID_TOKEN}"
            }
        )
        
        print(f"📋 Create presentation status: {response.status_code}")
        if response.status_code == 200:
            print("✅ Create presentation endpoint working!")
        else:
            print("❌ Create presentation endpoint not working!")
        
        return True
        
    except Exception as e:
        print(f"❌ Server health check failed: {e}")
        return False

async def main():
    """Main function"""
    print("🧪 Socket.IO Working Test Suite")
    print("=" * 50)
    print("This test will verify if your Socket.IO is working properly.")
    print("=" * 50)
    
    # Run all tests
    tests = [
        ("Server Health", test_server_health),
        ("Socket.IO Endpoints", test_socketio_endpoints),
        ("Basic Connection", test_socketio_basic_connection),
        ("Authenticated Connection", test_socketio_with_auth)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"Running: {test_name}")
        print(f"{'='*60}")
        
        try:
            result = await test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ Test {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print(f"\n{'='*60}")
    print("🎯 TEST RESULTS SUMMARY")
    print(f"{'='*60}")
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    passed = sum(results.values())
    total = len(results)
    
    print(f"\n📊 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your Socket.IO is working correctly!")
    else:
        print("⚠️ Some tests failed. Check the issues above.")
    
    # Additional recommendations
    print(f"\n💡 Recommendations:")
    if not results.get("Server Health", False):
        print("   • Check if the server is running: python main.py")
    if not results.get("Socket.IO Endpoints", False):
        print("   • Check if Socket.IO is properly mounted in main.py")
    if not results.get("Basic Connection", False):
        print("   • Check Socket.IO server configuration")
    if not results.get("Authenticated Connection", False):
        print("   • Check JWT authentication in socketio_manager.py")
    
    if received_events:
        print(f"\n📨 Events Received: {len(received_events)}")
        for i, event in enumerate(received_events[:5]):  # Show first 5 events
            print(f"   {i+1}. {event.get('type', 'unknown')} from {event.get('author', 'unknown')}")
    else:
        print(f"\n📨 No events received - this might indicate agent output issues")

if __name__ == "__main__":
    asyncio.run(main())
