"""
Complete end-to-end test for Socket.IO presentation generation
This script:
1. Creates a presentation
2. Connects via Socket.IO
3. Starts the generation
4. Receives real-time updates
"""

import asyncio
import socketio
import httpx
import sys
from datetime import datetime

class PresentationTester:
    def __init__(self, api_url, token):
        self.api_url = api_url
        self.token = token
        self.p_id = None
        self.sio = socketio.AsyncClient(logger=True, engineio_logger=True)
        self.message_count = 0
        self.connected = False
        self.completed = False
        
        self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup Socket.IO event handlers"""
        
        @self.sio.event
        async def connect():
            self.connected = True
            print(f"\n{'='*60}")
            print(f"✅ Socket.IO CONNECTED")
            print(f"{'='*60}\n")
        
        @self.sio.event
        async def disconnect():
            self.connected = False
            print(f"\n{'='*60}")
            print(f"❌ Socket.IO DISCONNECTED")
            print(f"{'='*60}\n")
        
        @self.sio.event
        async def connected(data):
            print(f"\n🎉 Connection confirmed by server:")
            print(f"   Session ID: {data.get('session_id')}")
            print(f"   P_ID: {data.get('p_id')}")
            print(f"   User ID: {data.get('user_id')}")
            print(f"   Worker ID: {data.get('worker_id')}")
            print(f"   Timestamp: {data.get('timestamp')}")
        
        @self.sio.event
        async def message(data):
            self.message_count += 1
            msg_type = data.get('type', 'unknown')
            
            # Header
            print(f"\n{'─'*60}")
            print(f"📨 Message #{self.message_count} [{msg_type.upper()}]")
            print(f"{'─'*60}")
            
            # Content based on type
            if msg_type == 'heartbeat':
                print(f"💓 Heartbeat from worker: {data.get('worker_id')}")
            
            elif msg_type == 'event':
                event = data.get('event', 'N/A')
                print(f"🎯 Event: {event}")
                
                if event == 'started':
                    print(f"🚀 Presentation generation STARTED!")
                    print(f"   Status: {data.get('status')}")
                    print(f"   Worker: {data.get('worker_id')}")
                
                elif event == 'retrying':
                    print(f"🔄 Retrying...")
                    print(f"   Attempt: {data.get('attempt')}")
                    print(f"   Reason: {data.get('reason')}")
                    print(f"   Retry in: {data.get('retry_in_sec')}s")
            
            elif msg_type == 'chunk':
                content = data.get('content', '')
                author = data.get('author', 'unknown')
                
                print(f"👤 Author: {author}")
                print(f"📄 Content length: {len(content)} chars")
                
                # Show first 300 chars
                preview = content[:300]
                if len(content) > 300:
                    preview += "..."
                
                print(f"\n--- Content Preview ---")
                print(preview)
                print(f"--- End Preview ---")
            
            elif msg_type == 'function_call':
                function = data.get('function', 'unknown')
                print(f"🔧 Function called: {function}")
            
            elif msg_type == 'terminal':
                event = data.get('event', 'N/A')
                print(f"🏁 TERMINAL EVENT: {event}")
                
                if event == 'completed':
                    print(f"\n{'='*60}")
                    print(f"✅ PRESENTATION COMPLETED SUCCESSFULLY!")
                    print(f"{'='*60}")
                    self.completed = True
                
                elif event == 'failed':
                    error_msg = data.get('message', 'Unknown error')
                    print(f"\n{'='*60}")
                    print(f"❌ PRESENTATION FAILED")
                    print(f"{'='*60}")
                    print(f"Error: {error_msg}")
                    self.completed = True
            
            else:
                print(f"📦 Other message type")
                print(f"Data keys: {list(data.keys())}")
            
            # Timestamp
            timestamp = data.get('timestamp', datetime.utcnow().isoformat())
            print(f"🕐 Time: {timestamp}")
    
    async def create_presentation(self, message):
        """Step 1: Create presentation"""
        print(f"\n{'='*60}")
        print(f"STEP 1: Creating Presentation")
        print(f"{'='*60}")
        
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                f"{self.api_url}/create-presentation",
                headers={
                    "Authorization": f"Bearer {self.token}",
                    "Content-Type": "application/json"
                },
                json={"message": message}
            )
            
            if response.status_code != 200:
                print(f"❌ Failed to create presentation: {response.text}")
                return False
            
            data = response.json()
            self.p_id = data.get('p_id')
            
            print(f"✅ Presentation created successfully!")
            print(f"   P_ID: {self.p_id}")
            print(f"   Status: {data.get('status')}")
            print(f"   User ID: {data.get('user_id')}")
            
            return True
    
    async def connect_socketio(self):
        """Step 2: Connect to Socket.IO"""
        print(f"\n{'='*60}")
        print(f"STEP 2: Connecting to Socket.IO")
        print(f"{'='*60}")
        
        if not self.p_id:
            print("❌ No p_id available. Create presentation first.")
            return False
        
        # Build connection URL with query parameters
        socket_url = self.api_url.replace('http', 'ws').replace('https', 'wss')
        
        # For Socket.IO, we connect to the base URL and pass query params
        connect_url = f"{self.api_url}/socket.io"
        
        print(f"🔌 Connecting to: {connect_url}")
        print(f"   P_ID: {self.p_id}")
        print(f"   Token: {self.token[:20]}...")
        
        try:
            await self.sio.connect(
                connect_url,
                transports=['websocket', 'polling'],
                socketio_path='/socket.io',
                wait_timeout=10,
                auth=None,
                headers={},
                query={
                    'p_id': self.p_id,
                    'token': self.token
                }
            )
            
            # Wait a moment for connection to stabilize
            await asyncio.sleep(1)
            
            if self.connected:
                print(f"✅ Socket.IO connection established!")
                return True
            else:
                print(f"❌ Socket.IO connection failed (not connected)")
                return False
            
        except Exception as e:
            print(f"❌ Socket.IO connection error: {e}")
            return False
    
    async def start_generation(self):
        """Step 3: Start presentation generation"""
        print(f"\n{'='*60}")
        print(f"STEP 3: Starting Presentation Generation")
        print(f"{'='*60}")
        
        if not self.connected:
            print("❌ Not connected to Socket.IO. Connect first.")
            return False
        
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                f"{self.api_url}/start-presentation/{self.p_id}",
                headers={"Authorization": f"Bearer {self.token}"}
            )
            
            if response.status_code != 200:
                print(f"❌ Failed to start generation: {response.text}")
                return False
            
            data = response.json()
            print(f"✅ Generation started!")
            print(f"   Message: {data.get('message')}")
            print(f"   Status: {data.get('status')}")
            
            return True
    
    async def wait_for_completion(self, timeout=300):
        """Step 4: Wait for completion"""
        print(f"\n{'='*60}")
        print(f"STEP 4: Waiting for Completion")
        print(f"{'='*60}")
        print(f"⏳ Waiting up to {timeout} seconds...")
        print(f"📡 Listening for real-time updates...\n")
        
        start_time = asyncio.get_event_loop().time()
        
        while not self.completed:
            elapsed = asyncio.get_event_loop().time() - start_time
            
            if elapsed > timeout:
                print(f"\n⏰ Timeout reached after {timeout} seconds")
                return False
            
            # Check status via API every 10 seconds
            if int(elapsed) % 10 == 0 and int(elapsed) > 0:
                try:
                    async with httpx.AsyncClient(timeout=10) as client:
                        response = await client.get(
                            f"{self.api_url}/presentation/{self.p_id}/status",
                            headers={"Authorization": f"Bearer {self.token}"}
                        )
                        if response.status_code == 200:
                            status_data = response.json()
                            status = status_data.get('status')
                            print(f"\n📊 Status check: {status}")
                except:
                    pass
            
            await asyncio.sleep(1)
        
        print(f"\n✅ Completed! Total messages received: {self.message_count}")
        return True
    
    async def get_final_data(self):
        """Step 5: Get final presentation data"""
        print(f"\n{'='*60}")
        print(f"STEP 5: Retrieving Final Data")
        print(f"{'='*60}")
        
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(
                f"{self.api_url}/presentation/{self.p_id}/data",
                headers={"Authorization": f"Bearer {self.token}"}
            )
            
            if response.status_code != 200:
                print(f"❌ Failed to get data: {response.text}")
                return False
            
            data = response.json()
            print(f"✅ Data retrieved successfully!")
            print(f"   Title: {data.get('title', 'N/A')}")
            print(f"   Status: {data.get('status')}")
            print(f"   Total slides: {data.get('total_slides')}")
            print(f"   Slides in response: {len(data.get('slides', []))}")
            
            return True
    
    async def cleanup(self):
        """Cleanup connections"""
        if self.connected:
            await self.sio.disconnect()
        print(f"\n🧹 Cleanup complete")
    
    async def run_complete_test(self, message):
        """Run the complete test flow"""
        try:
            print(f"\n{'#'*60}")
            print(f"# COMPLETE SOCKET.IO TEST")
            print(f"# API: {self.api_url}")
            print(f"# Token: {self.token[:20]}...")
            print(f"{'#'*60}")
            
            # Step 1: Create presentation
            if not await self.create_presentation(message):
                return False
            
            # Step 2: Connect Socket.IO
            if not await self.connect_socketio():
                return False
            
            # Step 3: Start generation
            if not await self.start_generation():
                return False
            
            # Step 4: Wait for completion
            if not await self.wait_for_completion(timeout=300):
                print(f"⚠️ Did not complete within timeout")
            
            # Step 5: Get final data
            await self.get_final_data()
            
            print(f"\n{'='*60}")
            print(f"🎉 TEST COMPLETED!")
            print(f"{'='*60}")
            print(f"Total messages received: {self.message_count}")
            print(f"Final status: {'Completed' if self.completed else 'Incomplete'}")
            
            return True
            
        except KeyboardInterrupt:
            print(f"\n\n⏹️  Test interrupted by user")
            return False
        except Exception as e:
            print(f"\n❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            await self.cleanup()

async def main():
    # Configuration
    API_URL = "http://localhost:8060"  # Match your server port
    TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJfaWQiOiI2ODgwODdlMzA1MTk0OTc2YWVhOTk0MDMiLCJzdWIiOiI2ODgwODdlMzA1MTk0OTc2YWVhOTk0MDMiLCJlbWFpbCI6InJycmlhZHVkZGluQGdtYWlsLmNvbSIsInBhY2thZ2UiOiJ1bmxpbWl0ZWQiLCJpc192ZXJpZmllZCI6dHJ1ZSwicm9sZSI6InVzZXIiLCJpYXQiOjE3NjA5MzM4Njl9.E9cUSlM-uQRrQ48gpH47t7eUN9ioem-A63CCPVJhZnI"
    MESSAGE = "Create a presentation about artificial intelligence and machine learning"
    
    # Allow command line override
    if len(sys.argv) > 1:
        API_URL = sys.argv[1]
    if len(sys.argv) > 2:
        TOKEN = sys.argv[2]
    if len(sys.argv) > 3:
        MESSAGE = sys.argv[3]
    
    # Run test
    tester = PresentationTester(API_URL, TOKEN)
    success = await tester.run_complete_test(MESSAGE)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Socket.IO Complete Test")
    print("="*60)
    print("\nUsage: python complete_test.py [API_URL] [TOKEN] [MESSAGE]")
    print("Example: python complete_test.py http://localhost:8060 YOUR_TOKEN 'Test message'\n")
    
    asyncio.run(main())