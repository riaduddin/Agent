#!/usr/bin/env python3
"""
Start the server with proper error handling and logging.
"""

import sys
import os
import traceback

# Add parent directory to path to allow importing modules from root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def check_dependencies():
    """Check if all required dependencies are available"""
    print("🔍 Checking Dependencies")
    print("=" * 50)
    
    try:
        print("1. Testing Socket.IO imports...")
        import socketio
        from socketio import ASGIApp
        print(f"   ✅ python-socketio: {socketio.__version__}")
        
        print("2. Testing FastAPI imports...")
        from fastapi import FastAPI
        print("   ✅ FastAPI imported successfully")
        
        print("3. Testing database imports...")
        from core.database import get_mongo_client
        print("   ✅ Database imports successful")
        
        print("4. Testing Qdrant imports...")
        from tools.qdrant_utils import get_qdrant_manager
        print("   ✅ Qdrant imports successful")
        
        print("5. Testing main.py import...")
        from main import app
        print("   ✅ main.py imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        print("🔧 Fix: Install missing dependencies")
        return False
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
        return False

def start_server():
    """Start the server with proper error handling"""
    print("\n🚀 Starting Server")
    print("=" * 50)
    
    try:
        print("Starting FastAPI server...")
        print("Server will be available at: http://127.0.0.1:8060")
        print("Socket.IO will be available at: http://127.0.0.1:8060/socket.io/")
        print("\nPress Ctrl+C to stop the server")
        print("=" * 50)
        
        # Import and run the server
        import uvicorn
        from main import app
        
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=8060,
            log_level="info",
            access_log=True
        )
        
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
        print("✅ Clean shutdown completed")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        print(f"\n🔍 Traceback:")
        traceback.print_exc()
        return False
    
    return True

def main():
    """Main function"""
    print("🚀 Presentation Generation Service")
    print("=" * 50)
    print("Starting server with Socket.IO support...")
    print("=" * 50)
    
    # Check dependencies first
    if not check_dependencies():
        print("\n❌ Dependency check failed!")
        print("🔧 Please install missing dependencies:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    
    print("\n✅ All dependencies available!")
    
    # Start the server
    if not start_server():
        print("\n❌ Server failed to start!")
        sys.exit(1)

if __name__ == "__main__":
    main()
