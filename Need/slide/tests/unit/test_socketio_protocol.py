#!/usr/bin/env python3
"""
Test Socket.IO Protocol Versions
Test different Socket.IO protocol versions to find compatibility
"""

import asyncio
import socketio
import json

async def test_protocol_versions():
    """Test different Socket.IO protocol versions"""
    print("🔍 Testing Socket.IO Protocol Versions")
    print("=" * 50)
    
    # Test different client configurations
    configs = [
        {"engineio_version": 4, "socketio_version": 5},
        {"engineio_version": 3, "socketio_version": 4},
        {"engineio_version": 4, "socketio_version": 4},
    ]
    
    for i, config in enumerate(configs, 1):
        print(f"\n🔗 Test {i}: Engine.IO v{config['engineio_version']}, Socket.IO v{config['socketio_version']}")
        
        sio = socketio.AsyncClient(
            engineio_version=config['engineio_version'],
            socketio_version=config['socketio_version']
        )
        
        @sio.event
        async def connect():
            print(f"✅ Connected with config {i}")
        
        @sio.event
        async def connect_error(data):
            print(f"❌ Connection error with config {i}: {data}")
        
        try:
            connection_url = "http://127.0.0.1:8060?p_id=68fdbe3fe5a68e0e4de0fd2a&token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJfaWQiOiI2ODgxYjhjMzE2ZjViODk0MzZkYzBiNzMiLCJzdWIiOiI2ODgxYjhjMzE2ZjViODk0MzZkYzBiNzMiLCJlbWFpbCI6Im1obWFoZWRpMDAwQGdtYWlsLmNvbSIsInBhY2thZ2UiOiJ2YWx1ZV9wbGFuIiwiaXNfdmVyaWZpZWQiOnRydWUsInJvbGUiOiJ1c2VyIiwiaWF0IjoxNzYxMzg3MzUxfQ.wxPfF0xdNlE4oMEF63JTqvdbPq1rYNdQCgB4vKJ9LcY"
            
            await sio.connect(connection_url)
            print(f"✅ Config {i} connected successfully!")
            await sio.disconnect()
            break  # If successful, stop testing
            
        except Exception as e:
            print(f"❌ Config {i} failed: {e}")
            if sio.connected:
                await sio.disconnect()

if __name__ == "__main__":
    asyncio.run(test_protocol_versions())

