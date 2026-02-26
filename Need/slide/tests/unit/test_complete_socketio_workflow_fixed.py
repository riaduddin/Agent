#!/usr/bin/env python3
"""
Test the complete Socket.IO workflow with the fixed server
"""

import asyncio
import socketio
import requests
import json
import time
from datetime import datetime

# Configuration
SERVER_URL = "http://127.0.0.1:8060"
JWT_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJfaWQiOiI2ODgxYjhjMzE2ZjViODk0MzZkYzBiNzMiLCJzdWIiOiI2ODgxYjhjMzE2ZjViODk0MzZkYzBiNzMiLCJlbWFpbCI6Im1obWFoZWRpMDAwQGdtYWlsLmNvbSIsInBhY2thZ2UiOiJ2YWx1ZV9wbGFuIiwiaXNfdmVyaWZpZWQiOnRydWUsInJvbGUiOiJ1c2VyIiwiaWF0IjoxNzYxMzg3MzUxfQ.wxPfF0xdNlE4oMEF63JTqvdbPq1rYNdQCgB4vKJ9LcY"

class SocketIOTestClient:
    def __init__(self):
        self.sio = socketio.Client()
        self.presentation_id = None
        self.events_received = []
        self.setup_handlers()
    
    def setup_handlers(self):
        @self.sio.event
        def connect():
            print("✅ Connected to Socket.IO server")
        
        @self.sio.event
        def disconnect():
            print("❌ Disconnected from Socket.IO server")
        
        @self.sio.event
        def agent_output(data):
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] 🎯 Agent Output Received:")
            print(f"   Author: {data.get('author', 'unknown')}")
            print(f"   Type: {data.get('type', 'unknown')}")
            print(f"   P_ID: {data.get('p_id', 'unknown')}")
            
            if 'html_content' in data:
                print(f"   🎨 HTML Content: {data['html_content'][:100]}...")
            elif 'thinking' in data:
                print(f"   💭 Thinking: {data['thinking'][:100]}...")
            elif 'topic' in data:
                print(f"   🎯 Topic: {data['topic']}")
            elif 'text' in data:
                print(f"   📝 Text: {data['text'][:100]}...")
            else:
                print(f"   📄 Data: {json.dumps(data, indent=2)[:200]}...")
            
            self.events_received.append(data)
            print()
        
        @self.sio.event
        def message(data):
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] 📨 Message: {json.dumps(data, indent=2)}")
            self.events_received.append(data)
        
        @self.sio.event
        def connected(data):
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] ✅ Connected: {json.dumps(data, indent=2)}")
        
        @self.sio.event
        def subscribed(data):
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] ✅ Subscribed: {json.dumps(data, indent=2)}")
        
        @self.sio.event
        def error(data):
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] ❌ Error: {json.dumps(data, indent=2)}")
    
    async def create_presentation(self):
        """Create a new presentation"""
        print("🚀 Creating presentation...")
        
        response = requests.post(
            f"{SERVER_URL}/create-presentation",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {JWT_TOKEN}"
            },
            json={
                "message": "Create a presentation about AI in Healthcare with 5 slides",
                "file_urls": []
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            self.presentation_id = data["p_id"]
            print(f"✅ Presentation created: {self.presentation_id}")
            return True
        else:
            print(f"❌ Failed to create presentation: {response.text}")
            return False
    
    async def start_presentation(self):
        """Start the presentation generation"""
        if not self.presentation_id:
            print("❌ No presentation ID available")
            return False
        
        print(f"🚀 Starting presentation: {self.presentation_id}")
        
        response = requests.post(
            f"{SERVER_URL}/start-presentation/{self.presentation_id}",
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
    
    async def connect_socket(self):
        """Connect to Socket.IO server"""
        print("🔌 Connecting to Socket.IO...")
        
        try:
            # Connect with auth parameter (like the client code)
            self.sio.connect(
                SERVER_URL,
                auth={
                    "p_id": self.presentation_id,
                    "token": JWT_TOKEN
                },
                transports=['websocket']
            )
            return True
        except Exception as e:
            print(f"❌ Socket.IO connection failed: {e}")
            return False
    
    async def disconnect_socket(self):
        """Disconnect from Socket.IO server"""
        if self.sio.connected:
            self.sio.disconnect()
            print("🔌 Disconnected from Socket.IO")
    
    def get_summary(self):
        """Get summary of received events"""
        print(f"\n📊 Summary of {len(self.events_received)} events received:")
        
        event_types = {}
        authors = {}
        
        for event in self.events_received:
            event_type = event.get('type', 'unknown')
            author = event.get('author', 'unknown')
            
            event_types[event_type] = event_types.get(event_type, 0) + 1
            authors[author] = authors.get(author, 0) + 1
        
        print("📈 Event Types:")
        for event_type, count in event_types.items():
            print(f"   {event_type}: {count}")
        
        print("👥 Authors:")
        for author, count in authors.items():
            print(f"   {author}: {count}")

async def main():
    """Main test function"""
    print("🧪 Socket.IO Complete Workflow Test (Fixed)")
    print("=" * 50)
    
    client = SocketIOTestClient()
    
    try:
        # Step 1: Create presentation
        if not await client.create_presentation():
            return
        
        # Step 2: Connect to Socket.IO
        if not await client.connect_socket():
            return
        
        # Wait a moment for connection to stabilize
        await asyncio.sleep(1)
        
        # Step 3: Start presentation
        if not await client.start_presentation():
            return
        
        # Step 4: Monitor events for 30 seconds
        print("\n🎯 Monitoring Socket.IO events for 30 seconds...")
        print("   (You should see real-time agent outputs from agent_outputs_2)")
        print("-" * 50)
        
        start_time = time.time()
        while time.time() - start_time < 30:  # Monitor for 30 seconds
            await asyncio.sleep(1)
        
        # Step 5: Disconnect
        await client.disconnect_socket()
        
        # Step 6: Show summary
        client.get_summary()
        
        print("\n✅ Test completed!")
        
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
        await client.disconnect_socket()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        await client.disconnect_socket()

if __name__ == "__main__":
    asyncio.run(main())
