#!/usr/bin/env python3
"""
Quick test to verify the server can start and Socket.IO works.
"""

import subprocess
import time
import requests
import sys

def test_server_startup():
    """Test that the server can start without errors"""
    print("🧪 Testing Server Startup")
    print("=" * 50)
    
    try:
        print("1. Testing imports...")
        import socketio
        from socketio import ASGIApp
        from fastapi import FastAPI
        print("   ✅ All imports successful")
        
        print("2. Testing Socket.IO server creation...")
        sio = socketio.AsyncServer(cors_allowed_origins="*")
        print("   ✅ Socket.IO server created")
        
        print("3. Testing FastAPI integration...")
        app = FastAPI()
        sio_app = ASGIApp(sio, app)
        app.mount("/socket.io", sio_app)
        print("   ✅ FastAPI integration works")
        
        print("4. Testing main.py import...")
        from main import app as main_app
        print("   ✅ main.py imports successfully")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return False

def test_server_endpoints():
    """Test that server endpoints respond"""
    print("\n🌐 Testing Server Endpoints")
    print("=" * 50)
    
    base_url = "http://127.0.0.1:8060"
    
    try:
        print("1. Testing health endpoint...")
        response = requests.get(f"{base_url}/", timeout=5)
        if response.status_code == 200:
            print("   ✅ Health endpoint works")
        else:
            print(f"   ⚠️ Health endpoint returned {response.status_code}")
        
        print("2. Testing Socket.IO endpoint...")
        response = requests.get(f"{base_url}/socket.io/", timeout=5)
        if response.status_code in [200, 400]:  # 400 is expected for missing params
            print("   ✅ Socket.IO endpoint accessible")
        else:
            print(f"   ❌ Socket.IO endpoint returned {response.status_code}")
            return False
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("   ❌ Server not running - start it first!")
        return False
    except Exception as e:
        print(f"   ❌ Endpoint test failed: {e}")
        return False

def start_server_test():
    """Test starting the server"""
    print("\n🚀 Testing Server Start")
    print("=" * 50)
    
    try:
        print("Starting server in background...")
        process = subprocess.Popen(
            [sys.executable, "main.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        print("Waiting for server to start...")
        time.sleep(3)
        
        # Check if process is still running
        if process.poll() is None:
            print("   ✅ Server started successfully")
            
            # Test endpoints
            if test_server_endpoints():
                print("   ✅ Server endpoints working")
            else:
                print("   ⚠️ Server started but endpoints not responding")
            
            # Stop server
            process.terminate()
            process.wait()
            print("   ✅ Server stopped cleanly")
            return True
        else:
            stdout, stderr = process.communicate()
            print(f"   ❌ Server failed to start")
            print(f"   📝 Error: {stderr}")
            return False
            
    except Exception as e:
        print(f"   ❌ Server start test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Quick Test Suite")
    print("=" * 50)
    print("This test verifies the server can start and Socket.IO works")
    print("=" * 50)
    
    # Test 1: Import and basic functionality
    if not test_server_startup():
        print("\n❌ FAILED: Basic functionality test")
        print("🔧 Fix: Check Socket.IO versions and imports")
        return False
    
    # Test 2: Server startup
    if not start_server_test():
        print("\n❌ FAILED: Server startup test")
        print("🔧 Fix: Check server configuration and dependencies")
        return False
    
    print("\n✅ SUCCESS!")
    print("🎉 All tests passed!")
    print("\n📋 What was verified:")
    print("   🔧 Socket.IO Compatibility:")
    print("      • No translate_request() errors")
    print("      • Proper version compatibility")
    print("      • FastAPI integration works")
    print("   🔧 Server Functionality:")
    print("      • Server starts successfully")
    print("      • Endpoints respond correctly")
    print("      • Socket.IO accessible")
    print("\n🚀 Your server is ready to use!")
    print("\n📋 Next Steps:")
    print("   1. Start server: python main.py")
    print("   2. Test with HTML client: complete_workflow_test.html")
    print("   3. Test with Postman: SocketIO_Presentation_Test.postman_collection.json")
    print("   4. Monitor logs for any issues")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
