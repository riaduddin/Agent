#!/usr/bin/env python3
"""
Test script to verify the circular import is fixed.
"""

import sys
import os
import traceback

# Add root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def test_imports():
    """Test that all imports work without circular import errors"""
    print("🧪 Testing Import Fix")
    print("=" * 50)
    
    try:
        print("1. Testing main.py import...")
        from main import app
        print("   ✅ main.py imported successfully")
        
        print("2. Testing app_socketio.py import...")
        from routers.socketio import router as socketio_router
        print("   ✅ app_socketio.py imported successfully")
        
        print("3. Testing socketio_manager import...")
        from core.socketio_manager import get_manager
        print("   ✅ socketio_manager imported successfully")
        
        print("4. Testing complete application...")
        # Try to access the app
        print(f"   ✅ FastAPI app: {app}")
        print(f"   ✅ Socket.IO router: {socketio_router}")
        
        print("\n🎉 All imports successful! Circular import is fixed!")
        return True
        
    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        print(f"\n🔍 Traceback:")
        traceback.print_exc()
        return False
        
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")
        print(f"\n🔍 Traceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Circular Import Fix Test")
    print("=" * 50)
    print("This test verifies that the circular import error is resolved")
    print("=" * 50)
    
    success = test_imports()
    
    if success:
        print("\n✅ SUCCESS!")
        print("🎉 Circular import is fixed!")
        print("🚀 You can now start the service with: python run_windows_simple.bat")
    else:
        print("\n❌ FAILED!")
        print("🔧 There are still import issues to resolve")
        sys.exit(1)
