#!/usr/bin/env python3
"""
Test script to show how to receive real-time data from Socket.IO after starting presentation generation.
"""

import asyncio
import socketio
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SocketIODataStreamer:
    def __init__(self, p_id: str, jwt_token: str, server_url: str = "http://localhost:8060"):
        self.p_id = p_id
        self.jwt_token = jwt_token
        self.server_url = server_url
        self.sio = socketio.AsyncClient()
        self.message_count = 0
        self.setup_handlers()
    
    def setup_handlers(self):
        """Setup Socket.IO event handlers"""
        
        @self.sio.event
        async def connect():
            logger.info(f"🔌 Connected to presentation {self.p_id}")
        
        @self.sio.event
        async def disconnect():
            logger.info(f"🔌 Disconnected from presentation {self.p_id}")
        
        @self.sio.event
        async def connected(data):
            logger.info(f"📨 Welcome: {data}")
        
        @self.sio.event
        async def message(data):
            self.message_count += 1
            msg_type = data.get("type", "unknown")
            author = data.get("author", "unknown")
            content = data.get("content", "")
            
            print(f"\n📨 Message #{self.message_count} [{msg_type}] from {author}")
            print(f"   Content: {content[:100]}{'...' if len(content) > 100 else ''}")
            
            # Show full content for important messages
            if msg_type in ["chunk", "event", "terminal"]:
                print(f"   Full data: {json.dumps(data, indent=2)}")
            
            # Stop after receiving terminal message or 20 messages
            if msg_type == "terminal" or self.message_count >= 20:
                return False
        
        @self.sio.event
        async def pong(data):
            logger.info(f"📨 Pong: {data}")
        
        @self.sio.event
        async def status_response(data):
            logger.info(f"📨 Status: {data}")
    
    async def connect_and_listen(self):
        """Connect and listen for real-time data"""
        try:
            print(f"🚀 Connecting to presentation {self.p_id}...")
            
            # Connect with query parameters
            await self.sio.connect(
                self.server_url,
                query={
                    'p_id': self.p_id,
                    'token': self.jwt_token
                }
            )
            
            print("✅ Connected! Listening for real-time data...")
            print("💡 Start presentation generation in another terminal or Postman")
            print("💡 Press Ctrl+C to stop listening")
            
            # Listen for messages
            await self.sio.wait()
            
        except Exception as e:
            logger.error(f"❌ Connection failed: {e}")
            return False
        
        return True
    
    async def disconnect(self):
        """Disconnect from Socket.IO"""
        await self.sio.disconnect()

async def test_data_streaming():
    """Test real-time data streaming"""
    
    # Configuration
    p_id = "68fc91847022f30b62bc95b0"  # Replace with your p_id
    jwt_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJfaWQiOiI2ODgwODdlMzA1MTk0OTc2YWVhOTk0MDMiLCJzdWIiOiI2ODgwODdlMzA1MTk0OTc2YWVhOTk0MDMiLCJlbWFpbCI6InJycmlhZHVkZGluQGdtYWlsLmNvbSIsInBhY2thZ2UiOiJ1bmxpbWl0ZWQiLCJpc192ZXJpZmllZCI6dHJ1ZSwicm9sZSI6InVzZXIiLCJpYXQiOjE3NjA5MzM4Njl9.E9cUSlM-uQRrQ48gpH47t7eUN9ioem-A63CCPVJhZnI"
    
    print("🧪 Socket.IO Data Streaming Test")
    print("=" * 60)
    print(f"📋 Presentation ID: {p_id}")
    print(f"📋 Server: http://localhost:8060")
    print("=" * 60)
    
    # Create streamer
    streamer = SocketIODataStreamer(p_id, jwt_token)
    
    try:
        # Connect and listen
        await streamer.connect_and_listen()
    except KeyboardInterrupt:
        print("\n🛑 Stopping...")
    finally:
        await streamer.disconnect()
        print("✅ Disconnected")

if __name__ == "__main__":
    print("Socket.IO Real-time Data Streaming")
    print("=" * 60)
    print("This script shows how to receive real-time data from presentation generation")
    print("=" * 60)
    print("\n💡 Instructions:")
    print("1. Start this script first")
    print("2. In another terminal/Postman, call: POST /start-presentation/{p_id}")
    print("3. Watch the real-time data stream here")
    print("=" * 60)
    
    # Run the test
    asyncio.run(test_data_streaming())
