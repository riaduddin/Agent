"""
Test WebSocket Setup
====================

This script tests the WebSocket implementation to ensure everything is configured correctly.

Usage:
    python tests/test_websocket_setup.py

Requirements:
    - Redis must be running
    - REDIS_URL must be set in .env
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()


async def test_redis_connection():
    """Test 1: Verify Redis connection"""
    print("\n" + "="*60)
    print("TEST 1: Redis Connection")
    print("="*60)
    
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    print(f"Redis URL: {redis_url}")
    
    try:
        import redis.asyncio as redis
        
        client = await redis.from_url(redis_url, encoding="utf-8", decode_responses=True)
        response = await client.ping()
        
        if response:
            print("✅ Redis connection successful!")
            print(f"   Response: {response}")
            
            # Test set/get
            await client.set("test_key", "test_value", ex=10)
            value = await client.get("test_key")
            
            if value == "test_value":
                print("✅ Redis read/write operations working!")
            else:
                print("❌ Redis read/write operations failed")
                return False
            
            # Cleanup
            await client.delete("test_key")
            await client.close()
            return True
        else:
            print("❌ Redis connection failed!")
            return False
            
    except ImportError:
        print("❌ redis package not installed!")
        print("   Install with: pip install redis>=4.2.0")
        return False
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check if Redis is running: redis-cli ping")
        print("2. Verify REDIS_URL in .env file")
        print("3. Check firewall/network settings")
        return False


async def test_redis_pubsub():
    """Test 2: Verify Redis Pub/Sub"""
    print("\n" + "="*60)
    print("TEST 2: Redis Pub/Sub")
    print("="*60)
    
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    try:
        import redis.asyncio as redis
        import json
        
        # Create publisher and subscriber
        pub_client = await redis.from_url(redis_url, encoding="utf-8", decode_responses=True)
        sub_client = await redis.from_url(redis_url, encoding="utf-8", decode_responses=True)
        
        pubsub = sub_client.pubsub()
        await pubsub.psubscribe("test:*")
        
        print("✅ Subscribed to pattern: test:*")
        
        # Publish a message
        message = {"test": "data", "timestamp": "2025-10-21"}
        await pub_client.publish("test:channel", json.dumps(message))
        print(f"✅ Published message to test:channel")
        
        # Try to receive the message
        received = False
        timeout = 5
        start = asyncio.get_event_loop().time()
        
        async for msg in pubsub.listen():
            if msg["type"] == "pmessage":
                data = json.loads(msg["data"])
                if data == message:
                    print(f"✅ Received message successfully!")
                    print(f"   Channel: {msg['channel']}")
                    print(f"   Data: {data}")
                    received = True
                    break
            
            # Timeout check
            if asyncio.get_event_loop().time() - start > timeout:
                break
        
        # Cleanup
        await pubsub.close()
        await pub_client.close()
        await sub_client.close()
        
        if received:
            print("✅ Redis Pub/Sub working correctly!")
            return True
        else:
            print("❌ Did not receive message within timeout")
            return False
            
    except Exception as e:
        print(f"❌ Redis Pub/Sub test failed: {e}")
        return False


async def test_distributed_lock():
    """Test 3: Verify distributed locking"""
    print("\n" + "="*60)
    print("TEST 3: Distributed Locking")
    print("="*60)
    
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    try:
        import redis.asyncio as redis
        
        client = await redis.from_url(redis_url, encoding="utf-8", decode_responses=True)
        
        lock_key = "test:lock:12345"
        worker_id = "test-worker-1"
        
        # Try to acquire lock
        locked = await client.set(lock_key, worker_id, nx=True, ex=60)
        
        if locked:
            print(f"✅ Acquired lock: {lock_key}")
            
            # Try to acquire again (should fail)
            locked_again = await client.set(lock_key, "test-worker-2", nx=True, ex=60)
            
            if not locked_again:
                print("✅ Lock prevents duplicate acquisition!")
                
                # Check lock owner
                owner = await client.get(lock_key)
                if owner == worker_id:
                    print(f"✅ Lock owner verified: {owner}")
                else:
                    print(f"❌ Lock owner mismatch: expected {worker_id}, got {owner}")
                    await client.delete(lock_key)
                    await client.close()
                    return False
                
                # Check TTL
                ttl = await client.ttl(lock_key)
                if 0 < ttl <= 60:
                    print(f"✅ Lock TTL correct: {ttl} seconds")
                else:
                    print(f"❌ Lock TTL incorrect: {ttl} seconds")
                    await client.delete(lock_key)
                    await client.close()
                    return False
                
                # Release lock
                await client.delete(lock_key)
                print("✅ Lock released successfully!")
                
                # Verify lock is gone
                exists = await client.exists(lock_key)
                if exists == 0:
                    print("✅ Lock cleanup verified!")
                    await client.close()
                    return True
                else:
                    print("❌ Lock not properly cleaned up")
                    await client.close()
                    return False
            else:
                print("❌ Lock allowed duplicate acquisition!")
                await client.delete(lock_key)
                await client.close()
                return False
        else:
            print("❌ Failed to acquire lock!")
            await client.close()
            return False
            
    except Exception as e:
        print(f"❌ Distributed lock test failed: {e}")
        return False


async def test_environment_variables():
    """Test 4: Verify environment variables"""
    print("\n" + "="*60)
    print("TEST 4: Environment Variables")
    print("="*60)
    
    required_vars = {
        "REDIS_URL": "Redis connection URL",
        "DATABASE_URL": "MongoDB connection URL",
        "JWT_SECRET": "JWT secret key for authentication"
    }
    
    all_present = True
    
    for var, description in required_vars.items():
        value = os.getenv(var)
        if value:
            # Mask sensitive values
            if "SECRET" in var or "PASSWORD" in var:
                display_value = value[:10] + "..." if len(value) > 10 else "***"
            else:
                display_value = value[:50] + "..." if len(value) > 50 else value
            
            print(f"✅ {var}: {display_value}")
            print(f"   Description: {description}")
        else:
            print(f"❌ {var}: NOT SET")
            print(f"   Description: {description}")
            all_present = False
    
    if all_present:
        print("\n✅ All required environment variables are set!")
        return True
    else:
        print("\n❌ Some environment variables are missing!")
        print("\nAdd them to your .env file:")
        print("REDIS_URL=redis://localhost:6379")
        print("DATABASE_URL=mongodb+srv://...")
        print("JWT_SECRET=your-secret-key")
        return False


async def test_websocket_manager():
    """Test 5: Verify WebSocket manager initialization"""
    print("\n" + "="*60)
    print("TEST 5: WebSocket Manager")
    print("="*60)
    
    try:
        from websocket_manager import ConnectionManager
        
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        
        print("Creating ConnectionManager...")
        manager = ConnectionManager(redis_url)
        
        print("Initializing manager...")
        await manager.initialize()
        
        print("✅ WebSocket manager initialized successfully!")
        
        # Check worker ID
        print(f"✅ Worker ID: {manager._worker_id}")
        
        # Check Redis client
        if manager.redis_client:
            print("✅ Redis client created")
            
            # Test Redis connection
            await manager.redis_client.ping()
            print("✅ Redis client connected")
        else:
            print("❌ Redis client not created")
            await manager.shutdown()
            return False
        
        # Check background tasks
        if manager._subscriber_task:
            print("✅ Redis subscriber task running")
        else:
            print("❌ Redis subscriber task not started")
            await manager.shutdown()
            return False
        
        if manager._heartbeat_task:
            print("✅ Heartbeat task running")
        else:
            print("❌ Heartbeat task not started")
            await manager.shutdown()
            return False
        
        # Shutdown
        print("Shutting down manager...")
        await manager.shutdown()
        print("✅ WebSocket manager shut down successfully!")
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import websocket_manager: {e}")
        print("   Make sure websocket_manager.py is in the project root")
        return False
    except Exception as e:
        print(f"❌ WebSocket manager test failed: {e}")
        return False


async def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("WEBSOCKET SETUP VERIFICATION")
    print("="*60)
    print("This will test your WebSocket configuration.\n")
    
    results = []
    
    # Test 1: Redis Connection
    results.append(("Redis Connection", await test_redis_connection()))
    
    # Test 2: Redis Pub/Sub
    results.append(("Redis Pub/Sub", await test_redis_pubsub()))
    
    # Test 3: Distributed Locking
    results.append(("Distributed Locking", await test_distributed_lock()))
    
    # Test 4: Environment Variables
    results.append(("Environment Variables", await test_environment_variables()))
    
    # Test 5: WebSocket Manager
    results.append(("WebSocket Manager", await test_websocket_manager()))
    
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
        print("\n🎉 All tests passed! Your WebSocket setup is ready!")
        print("\nNext steps:")
        print("1. Start your application: uvicorn main:app --reload")
        print("2. Test WebSocket connection: wscat -c ws://localhost:8000/ws/YOUR_P_ID?token=YOUR_TOKEN")
        print("3. For production: gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("- Install Redis: brew install redis (macOS) or apt-get install redis-server (Linux)")
        print("- Start Redis: redis-server or brew services start redis")
        print("- Set environment variables in .env file")
        print("- Install dependencies: pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(run_all_tests())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nTests cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

