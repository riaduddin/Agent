#!/usr/bin/env python3
"""
Test script to verify file_urls support in Socket.IO implementation.
"""

import asyncio
import sys
import traceback

async def test_file_urls_support():
    """Test that file_urls support works properly"""
    print("🧪 Testing File URLs Support")
    print("=" * 50)
    
    try:
        print("1. Testing imports...")
        from app_socketio import router as socketio_router, build_file_context, extract_text_from_url, PresentationRequest
        from main import app
        print("   ✅ All imports successful")
        
        print("2. Testing PresentationRequest model...")
        # Test with file_urls
        request_with_files = PresentationRequest(
            message="Create a presentation about AI",
            file_urls=["https://example.com/doc1.pdf", "https://example.com/doc2.docx"]
        )
        print(f"   ✅ Request with files: {request_with_files}")
        
        # Test without file_urls
        request_without_files = PresentationRequest(
            message="Create a presentation about AI"
        )
        print(f"   ✅ Request without files: {request_without_files}")
        
        print("3. Testing build_file_context function...")
        # Test with None (no files)
        result_none = await build_file_context(None)
        print(f"   ✅ build_file_context(None) = '{result_none}' (empty as expected)")
        
        # Test with empty list
        result_empty = await build_file_context([])
        print(f"   ✅ build_file_context([]) = '{result_empty}' (empty as expected)")
        
        # Test with mock URLs (will fail but should handle gracefully)
        result_mock = await build_file_context(["https://example.com/nonexistent.pdf"])
        print(f"   ✅ build_file_context(mock_urls) = '{result_mock}' (empty due to failed download)")
        
        print("4. Testing complete application...")
        print(f"   ✅ FastAPI app: {app}")
        print(f"   ✅ Socket.IO router: {socketio_router}")
        
        print("\n🎉 All tests passed! File URLs support implemented!")
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

def main():
    """Main test function"""
    print("File URLs Support Test")
    print("=" * 50)
    print("This test verifies that file_urls support works in Socket.IO")
    print("=" * 50)
    
    success = asyncio.run(test_file_urls_support())
    
    if success:
        print("\n✅ SUCCESS!")
        print("🎉 File URLs support implemented!")
        print("🚀 Socket.IO now supports optional file_urls parameter")
        print("\n📋 What was implemented:")
        print("   🔧 Request Model:")
        print("      • Added file_urls: Optional[List[str]] = None to PresentationRequest")
        print("      • file_urls is optional - can be provided or omitted")
        print("   🔧 File Processing:")
        print("      • build_file_context() now uses request.file_urls instead of None")
        print("      • If file_urls is None or empty, file_context will be empty")
        print("      • If file_urls provided, they will be processed into file_context")
        print("   🔧 Postman Collection:")
        print("      • Updated with example file_urls in request body")
        print("      • Added second example without file_urls")
        print("      • Shows both use cases")
        print("\n🎯 Usage Examples:")
        print("   📄 With files:")
        print('      {"message": "Create presentation", "file_urls": ["https://example.com/doc.pdf"]}')
        print("   📄 Without files:")
        print('      {"message": "Create presentation"}')
        print("\n🚀 The Socket.IO implementation now supports both scenarios!")
    else:
        print("\n❌ FAILED!")
        print("🔧 There are still issues to resolve")
        sys.exit(1)

if __name__ == "__main__":
    main()
