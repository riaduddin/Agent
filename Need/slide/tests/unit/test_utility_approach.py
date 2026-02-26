#!/usr/bin/env python3
"""
Test script to verify the utility module approach works.
"""

import sys
import os
import traceback

# Add root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def test_utility_approach():
    """Test that the utility module approach works without circular imports"""
    print("🧪 Testing Utility Module Approach")
    print("=" * 50)
    
    try:
        print("1. Testing utils.logging import...")
        from utils.logging import log_event_to_db
        print("   ✅ utils.logging imported successfully")
        
        print("2. Testing routers.socketio import...")
        from routers.socketio import router as socketio_router
        print("   ✅ routers.socketio imported successfully")
        
        print("3. Testing main.py import...")
        from main import app
        print("   ✅ main.py imported successfully")
        
        print("4. Testing complete application...")
        print(f"   ✅ FastAPI app: {app}")
        print(f"   ✅ Socket.IO router: {socketio_router}")
        
        print("\n🎉 All imports successful! No circular imports!")
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
    print("Utility Module Approach Test")
    print("=" * 50)
    print("This test verifies that the utility module approach resolves circular imports")
    print("=" * 50)
    
    success = test_utility_approach()
    
    if success:
        print("\n✅ SUCCESS!")
        print("🎉 Utility module approach works!")
        print("🚀 You can now start the service without circular import errors")
    else:
        print("\n❌ FAILED!")
        print("🔧 There are still import issues to resolve")
        sys.exit(1)
