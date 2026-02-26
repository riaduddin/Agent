#!/usr/bin/env python3
"""
Test Simple Socket.IO Connection
Test basic Socket.IO connection without authentication
"""

import asyncio
import socketio
import json

async def test_simple_connection():
    """Test simple Socket.IO connection"""
    print("🔍 Testing Simple Socket.IO Connection")
    print("=" * 40)
    
    sio = socketio.AsyncClient()
    
    @sio.event
    async def connect():
        print("✅ Connected to Socket.IO server")
    
    @sio.event
    async def connect_error(data):
        print(f"❌ Connection error: {data}")
    
    @sio.event
    async def disconnect():
        print("🔌 Disconnected from Socket.IO server")
    
    try:
        # Test 1: Connect to root namespace without any parameters
        print("🔗 Test 1: Connecting to root namespace...")
        await sio.connect("http://127.0.0.1:8060")
        print("✅ Root namespace connected")
        await sio.disconnect()
        
        # Test 2: Connect with empty query parameters
        print("\n🔗 Test 2: Connecting with empty query parameters...")
        await sio.connect("http://127.0.0.1:8060?")
        print("✅ Empty query parameters connected")
        await sio.disconnect()
        
        # Test 3: Connect with dummy parameters
        print("\n🔗 Test 3: Connecting with dummy parameters...")
        await sio.connect("http://127.0.0.1:8060?p_id=dummy&token=dummy")
        print("✅ Dummy parameters connected")
        await sio.disconnect()
        
        print("\n🎉 All simple connection tests passed!")
        
    except Exception as e:
        print(f"❌ Simple connection test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_simple_connection())