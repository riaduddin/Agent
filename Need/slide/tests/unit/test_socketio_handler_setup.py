#!/usr/bin/env python3
"""
Test Socket.IO Handler Setup
Test if the Socket.IO handler is properly set up
"""

import sys
import os
import asyncio
import socketio

# Add root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.socketio_manager import SocketIOConnectionManager

async def test_handler_setup():
    """Test Socket.IO handler setup"""
    print("🔍 Testing Socket.IO Handler Setup")
    print("=" * 40)
    
    try:
        # Create manager
        manager = SocketIOConnectionManager("redis://localhost:6379/0")
        print("✅ Manager created")
        
        # Check if sio is None
        print(f"🔍 sio is None: {manager.sio is None}")
        
        # Initialize manager
        await manager.initialize()
        print("✅ Manager initialized")
        
        # Check if sio is still None
        print(f"🔍 sio is None after init: {manager.sio is None}")
        
        # Check if handlers are set up
        if manager.sio:
            print("✅ Socket.IO server is available")
            
            # Try to get the connect handler
            connect_handler = getattr(manager.sio, '_connect_handler', None)
            print(f"🔍 Connect handler exists: {connect_handler is not None}")
            
            if connect_handler:
                print("✅ Connect handler is set up")
            else:
                print("❌ Connect handler is not set up")
        else:
            print("❌ Socket.IO server is None")
            
    except Exception as e:
        print(f"❌ Handler setup test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_handler_setup())

