#!/usr/bin/env python3
"""
Python test client for Socket.IO real-time communication.
This shows the actual data flow from the server.
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from dotenv import load_dotenv
import socketio

load_dotenv()

class SocketIOTestClient:
    def __init__(self, base_url, p_id, jwt_token):
        self.base_url = base_url
        self.p_id = p_id
        self.jwt_token = jwt_token
        self.sio = socketio.AsyncClient()
        self.connected = False
        
        # Setup event handlers
        self.setup_handlers()
    
    def setup_handlers(self):
        """Setup Socket.IO event handlers"""
        
        @self.sio.event
        async def connect():
            self.connected = True
            print(f"✅ Connected to Socket.IO server at {self.base_url}")
            print(f"📋 Connected with p_id: {self.p_id}")
        
        @self.sio.event
        async def disconnect():
            self.connected = False
            print("❌ Disconnected from Socket.IO server")
        
        @self.sio.event
        async def connect_error(data):
            print(f"❌ Connection error: {data}")
        
        @self.sio.event
        async def presentation_started(data):
            print(f"🚀 Presentation started: {json.dumps(data, indent=2)}")
        
        @self.sio.event
        async def presentation_progress(data):
            print(f"📊 Progress update: {json.dumps(data, indent=2)}")
        
        @self.sio.event
        async def presentation_completed(data):
            print(f"✅ Presentation completed: {json.dumps(data, indent=2)}")
        
        @self.sio.event
        async def presentation_failed(data):
            print(f"❌ Presentation failed: {json.dumps(data, indent=2)}")
        
        @self.sio.event
        async def agent_event(data):
            print(f"🤖 Agent event: {json.dumps(data, indent=2)}")
        
        @self.sio.event
        async def slide_generated(data):
            print(f"📄 Slide generated: {json.dumps(data, indent=2)}")
        
        @self.sio.event
        async def error(data):
            print(f"❌ Error: {json.dumps(data, indent=2)}")
        
        # Catch all events
        @self.sio.on('*')
        async def catch_all(event_name, *args):
            print(f"📡 Event '{event_name}': {json.dumps(args, indent=2)}")
    
    async def connect(self):
        """Connect to Socket.IO server"""
        try:
            print(f"🔌 Connecting to {self.base_url}...")
            await self.sio.connect(
                self.base_url,
                query={
                    'p_id': self.p_id,
                    'token': self.jwt_token
                },
                transports=['websocket', 'polling']
            )
            return True
        except Exception as e:
            print(f"❌ Failed to connect: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from Socket.IO server"""
        if self.connected:
            await self.sio.disconnect()
            print("🔌 Disconnected from server")
    
    async def test_connection(self):
        """Test the connection"""
        if self.connected:
            print("🧪 Testing connection...")
            await self.sio.emit('test', {'message': 'Hello from Python client!'})
        else:
            print("❌ Not connected to server")
    
    async def wait_for_events(self, duration=60):
        """Wait for events for specified duration"""
        print(f"⏳ Waiting for events for {duration} seconds...")
        print("Press Ctrl+C to stop")
        
        try:
            await asyncio.sleep(duration)
        except KeyboardInterrupt:
            print("\n🛑 Stopped by user")
    
    async def run_interactive(self):
        """Run interactive test session"""
        print("🔌 Socket.IO Real-time Test Client")
        print("=" * 50)
        
        # Connect
        if not await self.connect():
            return
        
        try:
            # Test connection
            await self.test_connection()
            
            # Wait for events
            await self.wait_for_events(300)  # 5 minutes
            
        except KeyboardInterrupt:
            print("\n🛑 Stopped by user")
        finally:
            await self.disconnect()

async def main():
    """Main function"""
    # Get configuration
    base_url = os.getenv("BASE_URL", "http://localhost:8060")
    p_id = os.getenv("P_ID", "test_presentation_id")
    jwt_token = os.getenv("JWT_TOKEN", "your_jwt_token_here")
    
    if jwt_token == "your_jwt_token_here":
        print("⚠️ Please set JWT_TOKEN in your .env file or environment")
        print("   Example: JWT_TOKEN=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...")
        return
    
    print(f"🔗 Base URL: {base_url}")
    print(f"📋 P_ID: {p_id}")
    print(f"🔑 JWT Token: {jwt_token[:20]}...")
    print()
    
    # Create and run client
    client = SocketIOTestClient(base_url, p_id, jwt_token)
    await client.run_interactive()

if __name__ == "__main__":
    print("Socket.IO Real-time Test Client")
    print("=" * 50)
    print("This client connects to your Socket.IO server and shows real-time data flow")
    print("=" * 50)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
