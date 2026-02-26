#!/usr/bin/env python3
"""
Test script to verify all imports are working correctly.
Run this before starting the main application.
"""

import sys
import os
# Add root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import traceback

def test_imports():
    """Test all critical imports"""
    print("🧪 Testing Socket.IO imports...")
    
    try:
        # Test FastAPI imports
        print("✅ Testing FastAPI imports...")
        from fastapi import APIRouter, HTTPException, Query, Depends
        print("   ✅ FastAPI imports successful")
        
        # Test Socket.IO imports
        print("✅ Testing Socket.IO imports...")
        import socketio
        print("   ✅ Socket.IO import successful")
        
        # Test Redis imports
        print("✅ Testing Redis imports...")
        import redis.asyncio as redis
        print("   ✅ Redis import successful")
        
        # Test database imports
        print("✅ Testing database imports...")
        from core.database import get_db, get_mongo_client
        print("   ✅ Database imports successful")
        
        # Test authentication imports
        print("✅ Testing authentication imports...")
        from middleware.auth import get_current_user
        print("   ✅ Authentication imports successful")
        
        # Test Socket.IO manager
        print("✅ Testing Socket.IO manager...")
        from core.socketio_manager import get_manager, cleanup_manager
        print("   ✅ Socket.IO manager imports successful")
        
        # Test Socket.IO app
        print("✅ Testing Socket.IO app...")
        from routers.socketio import router as socketio_router
        print("   ✅ Socket.IO app imports successful")
        
        # Test main app
        print("✅ Testing main app...")
        from main import app
        print("   ✅ Main app imports successful")
        
        print("\n🎉 All imports successful! Socket.IO service is ready to run.")
        return True
        
    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        print("\n💡 Missing dependencies. Install with:")
        print("   pip install -r requirements.txt")
        return False
        
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")
        print(f"\n🔍 Traceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Socket.IO Import Test")
    print("=" * 50)
    
    success = test_imports()
    
    if success:
        print("\n✅ All imports working correctly!")
        print("🚀 You can now start the service with:")
        print("   python run_windows_simple.bat")
        print("   OR")
        print("   python run_windows_python.py")
    else:
        print("\n❌ Import test failed!")
        print("🔧 Please fix the import errors before starting the service.")
        sys.exit(1)
