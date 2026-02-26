"""
Test WebSocket Reconnection
============================

This script tests the WebSocket reconnection functionality.

Usage:
    python tests/test_reconnection.py --p-id YOUR_P_ID --token YOUR_JWT_TOKEN

Requirements:
    - pip install websockets
    - Server must be running
    - Valid presentation ID and JWT token
"""

import asyncio
import json
import argparse
import sys
from datetime import datetime

try:
    import websockets
except ImportError:
    print("❌ websockets package not installed!")
    print("Install with: pip install websockets")
    sys.exit(1)


class ReconnectionTester:
    def __init__(self, ws_url: str, p_id: str, token: str):
        self.ws_url = ws_url
        self.p_id = p_id
        self.token = token
        self.last_event_id = None
        self.events_received = []
        self.reconnection_events = []
    
    async def connect(self, last_event_id=None):
        """Connect to WebSocket"""
        url = f"{self.ws_url}/ws/{self.p_id}?token={self.token}"
        if last_event_id:
            url += f"&last_event_id={last_event_id}"
            print(f"🔄 Connecting with last_event_id: {last_event_id}")
        else:
            print("🆕 Initial connection...")
        
        return await websockets.connect(url)
    
    async def test_initial_connection(self):
        """Test 1: Initial connection and receive events"""
        print("\n" + "="*60)
        print("TEST 1: Initial Connection")
        print("="*60)
        
        try:
            async with await self.connect() as ws:
                print("✅ Connected successfully")
                
                # Receive events for 5 seconds
                timeout = 5
                start = asyncio.get_event_loop().time()
                
                while asyncio.get_event_loop().time() - start < timeout:
                    try:
                        message = await asyncio.wait_for(ws.recv(), timeout=1.0)
                        data = json.loads(message)
                        
                        msg_type = data.get("type")
                        
                        if msg_type == "status":
                            print(f"📊 Status: {data.get('status')}")
                            print(f"   Is reconnection: {data.get('is_reconnection', False)}")
                        
                        elif msg_type in ("history", "chunk"):
                            event_id = data.get("event_id")
                            author = data.get("author", "unknown")
                            
                            if event_id:
                                self.last_event_id = event_id
                                self.events_received.append(event_id)
                                
                                # Send ACK
                                await ws.send(json.dumps({
                                    "type": "ack",
                                    "event_id": event_id
                                }))
                                
                                print(f"📥 Event: {author[:30]} (ID: {event_id[:8]}...)")
                        
                        elif msg_type == "backfill_complete":
                            count = data.get("events_count", 0)
                            print(f"✅ Backfill complete: {count} events")
                        
                        elif msg_type == "terminal":
                            print(f"🏁 Terminal: {data.get('event')}")
                            break
                    
                    except asyncio.TimeoutError:
                        continue
                
                print(f"\n📊 Summary:")
                print(f"   Events received: {len(self.events_received)}")
                print(f"   Last event ID: {self.last_event_id[:16] if self.last_event_id else 'None'}...")
                
                return len(self.events_received) > 0
        
        except Exception as e:
            print(f"❌ Test failed: {e}")
            return False
    
    async def test_reconnection(self):
        """Test 2: Reconnect with last_event_id"""
        print("\n" + "="*60)
        print("TEST 2: Reconnection with last_event_id")
        print("="*60)
        
        if not self.last_event_id:
            print("⚠️  Skipping - no last_event_id from test 1")
            return True
        
        print(f"Last event ID from previous connection: {self.last_event_id[:16]}...")
        print("Waiting 3 seconds before reconnecting...")
        await asyncio.sleep(3)
        
        try:
            async with await self.connect(self.last_event_id) as ws:
                print("✅ Reconnected successfully")
                
                # Receive events for 5 seconds
                timeout = 5
                start = asyncio.get_event_loop().time()
                
                while asyncio.get_event_loop().time() - start < timeout:
                    try:
                        message = await asyncio.wait_for(ws.recv(), timeout=1.0)
                        data = json.loads(message)
                        
                        msg_type = data.get("type")
                        
                        if msg_type == "status":
                            print(f"📊 Status: {data.get('status')}")
                            is_recon = data.get('is_reconnection', False)
                            resuming = data.get('resuming_from')
                            print(f"   Is reconnection: {is_recon}")
                            print(f"   Resuming from: {resuming[:16] if resuming else 'None'}...")
                            
                            if not is_recon:
                                print("   ⚠️  Server didn't detect reconnection!")
                        
                        elif msg_type in ("history", "chunk"):
                            event_id = data.get("event_id")
                            author = data.get("author", "unknown")
                            
                            if event_id:
                                self.reconnection_events.append(event_id)
                                
                                # Check for duplicates
                                if event_id in self.events_received:
                                    print(f"   ❌ DUPLICATE EVENT: {event_id[:8]}...")
                                else:
                                    print(f"   📥 New event: {author[:30]} (ID: {event_id[:8]}...)")
                                
                                # Send ACK
                                await ws.send(json.dumps({
                                    "type": "ack",
                                    "event_id": event_id
                                }))
                        
                        elif msg_type == "backfill_complete":
                            count = data.get("events_count", 0)
                            is_recon = data.get("is_reconnection", False)
                            print(f"✅ Backfill complete: {count} events")
                            print(f"   Is reconnection: {is_recon}")
                            
                            if count == 0:
                                print("   ✅ No new events (as expected for quick reconnect)")
                        
                        elif msg_type == "terminal":
                            print(f"🏁 Terminal: {data.get('event')}")
                            break
                    
                    except asyncio.TimeoutError:
                        continue
                
                print(f"\n📊 Summary:")
                print(f"   Events received: {len(self.reconnection_events)}")
                
                # Check for duplicates
                duplicates = set(self.reconnection_events) & set(self.events_received)
                if duplicates:
                    print(f"   ❌ Found {len(duplicates)} duplicate events!")
                    return False
                else:
                    print(f"   ✅ No duplicate events!")
                
                return True
        
        except Exception as e:
            print(f"❌ Test failed: {e}")
            return False
    
    async def test_ping_pong(self):
        """Test 3: Ping/pong"""
        print("\n" + "="*60)
        print("TEST 3: Ping/Pong")
        print("="*60)
        
        try:
            async with await self.connect() as ws:
                print("✅ Connected")
                
                # Send ping
                timestamp = datetime.utcnow().isoformat()
                await ws.send(json.dumps({
                    "type": "ping",
                    "timestamp": timestamp
                }))
                print(f"🏓 Sent ping at {timestamp}")
                
                # Wait for pong
                for _ in range(5):
                    try:
                        message = await asyncio.wait_for(ws.recv(), timeout=2.0)
                        data = json.loads(message)
                        
                        if data.get("type") == "pong":
                            server_ts = data.get("timestamp")
                            client_ts = data.get("client_timestamp")
                            print(f"✅ Received pong")
                            print(f"   Server timestamp: {server_ts}")
                            print(f"   Client timestamp: {client_ts}")
                            print(f"   Match: {client_ts == timestamp}")
                            return client_ts == timestamp
                    
                    except asyncio.TimeoutError:
                        continue
                
                print("❌ No pong received")
                return False
        
        except Exception as e:
            print(f"❌ Test failed: {e}")
            return False
    
    async def test_status_check(self):
        """Test 4: Status check"""
        print("\n" + "="*60)
        print("TEST 4: Status Check")
        print("="*60)
        
        try:
            async with await self.connect() as ws:
                print("✅ Connected")
                
                # Send status check
                await ws.send(json.dumps({
                    "type": "status_check"
                }))
                print("📨 Sent status_check")
                
                # Wait for response
                for _ in range(5):
                    try:
                        message = await asyncio.wait_for(ws.recv(), timeout=2.0)
                        data = json.loads(message)
                        
                        if data.get("type") == "status_response":
                            print(f"✅ Received status response")
                            print(f"   Status: {data.get('status')}")
                            print(f"   P_ID: {data.get('p_id')}")
                            return True
                    
                    except asyncio.TimeoutError:
                        continue
                
                print("❌ No status response received")
                return False
        
        except Exception as e:
            print(f"❌ Test failed: {e}")
            return False
    
    async def run_all_tests(self):
        """Run all tests"""
        print("\n" + "="*60)
        print("WEBSOCKET RECONNECTION TESTS")
        print("="*60)
        print(f"URL: {self.ws_url}")
        print(f"P_ID: {self.p_id}")
        print(f"Token: {self.token[:20]}...")
        
        results = []
        
        # Test 1: Initial connection
        results.append(("Initial Connection", await self.test_initial_connection()))
        
        # Test 2: Reconnection
        results.append(("Reconnection", await self.test_reconnection()))
        
        # Test 3: Ping/Pong
        results.append(("Ping/Pong", await self.test_ping_pong()))
        
        # Test 4: Status Check
        results.append(("Status Check", await self.test_status_check()))
        
        # Summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        
        passed = 0
        failed = 0
        
        for test_name, result in results:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{status} - {test_name}")
            if result:
                passed += 1
            else:
                failed += 1
        
        print("\n" + "="*60)
        print(f"Results: {passed} passed, {failed} failed")
        print("="*60)
        
        if failed == 0:
            print("\n🎉 All reconnection tests passed!")
            return 0
        else:
            print("\n⚠️  Some tests failed.")
            return 1


async def main():
    parser = argparse.ArgumentParser(description='Test WebSocket reconnection')
    parser.add_argument('--url', default='ws://localhost:8000', help='WebSocket server URL')
    parser.add_argument('--p-id', required=True, help='Presentation ID')
    parser.add_argument('--token', required=True, help='JWT token')
    
    args = parser.parse_args()
    
    tester = ReconnectionTester(args.url, args.p_id, args.token)
    exit_code = await tester.run_all_tests()
    
    sys.exit(exit_code)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nTests cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

