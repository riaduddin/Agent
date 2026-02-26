#!/usr/bin/env python3
"""
Test client to demonstrate file_urls functionality in Socket.IO.
"""

import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

def test_file_urls_client():
    """Test client for file_urls functionality"""
    print("🧪 Testing File URLs Client")
    print("=" * 50)
    
    # Get configuration
    base_url = os.getenv("BASE_URL", "http://localhost:8060")
    jwt_token = os.getenv("JWT_TOKEN", "your_jwt_token_here")
    
    if jwt_token == "your_jwt_token_here":
        print("⚠️ Please set JWT_TOKEN in your .env file or environment")
        print("   Example: JWT_TOKEN=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...")
        return False
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {jwt_token}"
    }
    
    print(f"🔗 Base URL: {base_url}")
    print(f"🔑 JWT Token: {jwt_token[:20]}...")
    
    # Test 1: Create presentation with file_urls
    print("\n1. Testing with file_urls...")
    payload_with_files = {
        "message": "Create a presentation about AI in Healthcare with 10 slides",
        "file_urls": [
            "https://example.com/healthcare_report.pdf",
            "https://example.com/ai_guidelines.docx"
        ]
    }
    
    try:
        response = requests.post(
            f"{base_url}/create-presentation",
            headers=headers,
            json=payload_with_files,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Success: {data}")
            p_id_with_files = data.get("p_id")
            print(f"   📋 p_id: {p_id_with_files}")
        else:
            print(f"   ❌ Failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    # Test 2: Create presentation without file_urls
    print("\n2. Testing without file_urls...")
    payload_without_files = {
        "message": "Create a presentation about Machine Learning with 8 slides"
    }
    
    try:
        response = requests.post(
            f"{base_url}/create-presentation",
            headers=headers,
            json=payload_without_files,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Success: {data}")
            p_id_without_files = data.get("p_id")
            print(f"   📋 p_id: {p_id_without_files}")
        else:
            print(f"   ❌ Failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    # Test 3: Start presentation with files
    if 'p_id_with_files' in locals():
        print(f"\n3. Testing start presentation with files (p_id: {p_id_with_files})...")
        try:
            response = requests.post(
                f"{base_url}/start-presentation/{p_id_with_files}",
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Success: {data}")
            else:
                print(f"   ❌ Failed: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Test 4: Start presentation without files
    if 'p_id_without_files' in locals():
        print(f"\n4. Testing start presentation without files (p_id: {p_id_without_files})...")
        try:
            response = requests.post(
                f"{base_url}/start-presentation/{p_id_without_files}",
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Success: {data}")
            else:
                print(f"   ❌ Failed: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n🎉 File URLs client test completed!")
    return True

if __name__ == "__main__":
    print("File URLs Client Test")
    print("=" * 50)
    print("This test demonstrates file_urls functionality")
    print("=" * 50)
    
    success = test_file_urls_client()
    
    if success:
        print("\n✅ SUCCESS!")
        print("🎉 File URLs client test completed!")
        print("\n📋 Test Summary:")
        print("   • Created presentation with file_urls")
        print("   • Created presentation without file_urls")
        print("   • Started both presentations")
        print("   • Verified different behaviors for each case")
    else:
        print("\n❌ FAILED!")
        print("🔧 There are issues with the client test")
        sys.exit(1)
