#!/usr/bin/env python3
"""
Test script to verify WebSocket isolation between different p_id connections.
This script simulates multiple WebSocket connections from the same user to different presentations.
"""

import asyncio
import websockets
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_websocket_connection(p_id: str, user_id: str, token: str, server_url: str = "ws://localhost:8060"):
    """Test a single WebSocket connection to a specific presentation"""
    uri = f"{server_url}/ws/{p_id}?token={token}"
    
    try:
        async with websockets.connect(uri) as websocket:
            logger.info(f"🔌 Connected to {p_id}")
            
            # Listen for messages
            message_count = 0
            async for message in websocket:
                try:
                    data = json.loads(message)
                    message_count += 1
                    
                    # Log the message with p_id for verification
                    msg_p_id = data.get("p_id", "unknown")
                    msg_type = data.get("type", "unknown")
                    
                    logger.info(f"📨 [{p_id}] Received message #{message_count}: type={msg_type}, msg_p_id={msg_p_id}")
                    
                    # Check if message belongs to this presentation
                    if msg_p_id != p_id and msg_p_id != "unknown":
                        logger.error(f"❌ CROSS-CONTAMINATION DETECTED! Expected p_id={p_id}, got msg_p_id={msg_p_id}")
                        return False
                    
                    # Stop after receiving 10 messages or if we get a completion message
                    if message_count >= 10 or data.get("type") == "terminal":
                        break
                        
                except json.JSONDecodeError:
                    logger.warning(f"⚠️ Invalid JSON received: {message}")
                except Exception as e:
                    logger.error(f"❌ Error processing message: {e}")
            
            logger.info(f"✅ Connection to {p_id} completed successfully (received {message_count} messages)")
            return True
            
    except Exception as e:
        logger.error(f"❌ Connection to {p_id} failed: {e}")
        return False

async def test_multiple_presentations():
    """Test multiple presentations from the same user"""
    # Replace with your actual values
    user_id = "688087e305194976aea99403"
    token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJfaWQiOiI2ODgwODdlMzA1MTk0OTc2YWVhOTk0MDMiLCJzdWIiOiI2ODgwODdlMzA1MTk0OTc2YWVhOTk0MDMiLCJlbWFpbCI6InJycmlhZHVkZGluQGdtYWlsLmNvbSIsInBhY2thZ2UiOiJ1bmxpbWl0ZWQiLCJpc192ZXJpZmllZCI6dHJ1ZSwicm9sZSI6InVzZXIiLCJpYXQiOjE3NjA5MzM4Njl9.E9cUSlM-uQRrQ48gpH47t7eUN9ioem-A63CCPVJhZnI"
    
    # Test with different p_id values
    test_presentations = [
        "68fc7f6adf48c211f801c37c",  # Presentation 1
        "68fc7f87df48c211f801c381",  # Presentation 2
        "68fc7f9adf48c211f801c385",  # Presentation 3
    ]
    
    logger.info("🚀 Starting WebSocket isolation test...")
    logger.info(f"📊 Testing {len(test_presentations)} presentations for user {user_id}")
    
    # Create tasks for concurrent connections
    tasks = []
    for p_id in test_presentations:
        task = asyncio.create_task(test_websocket_connection(p_id, user_id, token))
        tasks.append((p_id, task))
    
    # Wait for all connections to complete
    results = []
    for p_id, task in tasks:
        try:
            result = await task
            results.append((p_id, result))
        except Exception as e:
            logger.error(f"❌ Task for {p_id} failed: {e}")
            results.append((p_id, False))
    
    # Report results
    logger.info("📊 Test Results:")
    for p_id, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"   {p_id}: {status}")
    
    # Check if any cross-contamination was detected
    all_passed = all(result for _, result in results)
    if all_passed:
        logger.info("🎉 All tests passed! WebSocket isolation is working correctly.")
    else:
        logger.error("💥 Some tests failed. Check the logs above for details.")
    
    return all_passed

if __name__ == "__main__":
    print("🧪 WebSocket Isolation Test")
    print("=" * 50)
    print("This test verifies that different p_id connections")
    print("receive only their own messages (no cross-contamination).")
    print("=" * 50)
    
    # Run the test
    asyncio.run(test_multiple_presentations())
