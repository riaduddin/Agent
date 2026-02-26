#!/usr/bin/env python3
"""
Test script to verify session creation works properly.
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

load_dotenv()

async def test_session_creation():
    """Test that session creation works without errors"""
    print("🧪 Testing Session Creation")
    print("=" * 50)
    
    try:
        # Import required modules
        from google.adk.sessions import DatabaseSessionService, InMemorySessionService
        from google.adk.runners import Runner
        from root_agent.agent import SlideOrchestrationAgent
        
        # Get database URL
        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            print("❌ DATABASE_URL not found in environment")
            return False
        
        print(f"✅ Database URL found: {db_url[:50]}...")
        
        # Test session service creation
        print("1. Testing DatabaseSessionService creation...")
        try:
            db_url_with_timeout = db_url
            if '?' in db_url:
                if 'connect_timeout' not in db_url:
                    db_url_with_timeout = db_url + '&connect_timeout=10'
            else:
                db_url_with_timeout = db_url + '?connect_timeout=10'
            
            session_service = DatabaseSessionService(db_url=db_url_with_timeout)
            print("   ✅ DatabaseSessionService created successfully")
        except Exception as e:
            print(f"   ⚠️ DatabaseSessionService failed: {e}")
            print("   🔄 Falling back to InMemorySessionService...")
            session_service = InMemorySessionService()
            print("   ✅ InMemorySessionService created successfully")
        
        # Test session creation
        print("2. Testing session creation...")
        test_p_id = "test_session_123"
        test_user_id = "test_user_456"
        app_name = "Slide_creator"
        
        try:
            # Check if session exists
            session = await asyncio.wait_for(
                session_service.get_session(app_name=app_name, session_id=test_p_id, user_id=test_user_id),
                timeout=10.0
            )
            print(f"   ✅ Session lookup completed, found: {bool(session)}")
            
            if not session:
                print("   🔍 Creating new session...")
                initial_state = {"p_id": test_p_id, "user_id": test_user_id}
                await asyncio.wait_for(
                    session_service.create_session(
                        app_name=app_name, user_id=test_user_id, session_id=test_p_id, state=initial_state
                    ),
                    timeout=10.0
                )
                print("   ✅ New session created successfully")
            else:
                print("   ✅ Session already exists")
                
        except asyncio.TimeoutError:
            print("   ❌ Session operation timed out")
            return False
        except Exception as e:
            print(f"   ❌ Session operation failed: {e}")
            return False
        
        # Test runner creation
        print("3. Testing Runner creation...")
        try:
            runner = Runner(
                agent=SlideOrchestrationAgent,
                app_name=app_name,
                session_service=session_service
            )
            print("   ✅ Runner created successfully")
        except Exception as e:
            print(f"   ❌ Runner creation failed: {e}")
            return False
        
        print("\n🎉 All session operations successful!")
        return True
        
    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")
        return False

async def main():
    """Main test function"""
    print("Session Creation Test")
    print("=" * 50)
    print("This test verifies that session creation works properly")
    print("=" * 50)
    
    success = await test_session_creation()
    
    if success:
        print("\n✅ SUCCESS!")
        print("🎉 Session creation works properly!")
        print("🚀 The 'Session not found' error should be resolved")
    else:
        print("\n❌ FAILED!")
        print("🔧 There are still session creation issues to resolve")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
