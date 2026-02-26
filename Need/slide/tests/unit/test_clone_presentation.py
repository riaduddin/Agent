"""
Test script for Clone Presentation Feature
This script demonstrates how to clone a shared presentation
"""

import requests
import json
import time
from typing import Optional

# Configuration
BASE_URL = "http://localhost:8000"

# Test tokens - Replace with actual tokens
USER_A_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."  # User who owns original
USER_B_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."  # User who clones


def print_section(title: str):
    """Print a section header"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")


def test_clone_presentation(original_p_id: str, user_b_token: str) -> Optional[dict]:
    """
    Test cloning a presentation
    
    Args:
        original_p_id: The p_id of the presentation to clone
        user_b_token: JWT token for User B
        
    Returns:
        Clone response data or None if failed
    """
    print_section("TEST: Clone Presentation")
    
    print(f"📋 Original p_id: {original_p_id}")
    print(f"👤 Cloning as User B (from token)")
    
    headers = {
        "Authorization": f"Bearer {user_b_token}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "original_p_id": original_p_id
    }
    
    print(f"\n📤 Sending clone request to: {BASE_URL}/clone-presentation")
    
    try:
        response = requests.post(
            f"{BASE_URL}/clone-presentation",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        print(f"📥 Response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("\n✅ Clone successful!")
            print(f"   Original p_id: {data.get('original_p_id')}")
            print(f"   New p_id: {data.get('new_p_id')}")
            print(f"   User ID: {data.get('user_id')}")
            print(f"   Email: {data.get('email')}")
            print(f"   Total slides: {data.get('total_slides')}")
            print(f"   Can edit: {data.get('can_edit')}")
            return data
        else:
            print(f"\n❌ Clone failed!")
            print(f"   Error: {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        print("\n❌ Request timed out")
        return None
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Request failed: {e}")
        return None


def test_view_cloned_slides(p_id: str, token: str) -> bool:
    """
    Test viewing slides from cloned presentation
    
    Args:
        p_id: The cloned presentation p_id
        token: JWT token
        
    Returns:
        True if successful, False otherwise
    """
    print_section("TEST: View Cloned Slides")
    
    print(f"📋 Cloned p_id: {p_id}")
    
    headers = {
        "Authorization": f"Bearer {token}"
    }
    
    try:
        response = requests.get(
            f"{BASE_URL}/slides/",
            params={"p_id": p_id},
            headers=headers,
            timeout=30
        )
        
        print(f"📥 Response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            slides = data.get('slides', [])
            print(f"\n✅ Successfully retrieved slides!")
            print(f"   Total slides: {len(slides)}")
            print(f"   Status: {data.get('status')}")
            print(f"   Title: {data.get('title')}")
            
            if slides:
                print(f"\n   First slide preview:")
                first_slide = slides[0]
                print(f"   - Index: {first_slide.get('slide_index')}")
                print(f"   - Plan: {first_slide.get('slide_plan', '')[:100]}...")
            
            return True
        else:
            print(f"\n❌ Failed to retrieve slides!")
            print(f"   Error: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Request failed: {e}")
        return False


def test_view_original_presentation(p_id: str, token: str) -> bool:
    """
    Test viewing the original presentation (to verify it exists)
    
    Args:
        p_id: The original presentation p_id
        token: JWT token of owner
        
    Returns:
        True if successful, False otherwise
    """
    print_section("TEST: Verify Original Presentation")
    
    print(f"📋 Original p_id: {p_id}")
    
    headers = {
        "Authorization": f"Bearer {token}"
    }
    
    try:
        response = requests.get(
            f"{BASE_URL}/slides/",
            params={"p_id": p_id},
            headers=headers,
            timeout=30
        )
        
        print(f"📥 Response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n✅ Original presentation exists!")
            print(f"   Status: {data.get('status')}")
            print(f"   Title: {data.get('title')}")
            print(f"   Total slides: {data.get('total_slides')}")
            return True
        else:
            print(f"\n❌ Original presentation not found or inaccessible!")
            print(f"   Error: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Request failed: {e}")
        return False


def test_ownership_verification(original_p_id: str, user_b_token: str) -> bool:
    """
    Test that User B cannot access User A's original presentation
    
    Args:
        original_p_id: User A's presentation p_id
        user_b_token: User B's JWT token
        
    Returns:
        True if access is properly denied, False otherwise
    """
    print_section("TEST: Ownership Verification")
    
    print(f"📋 Testing if User B can access User A's original presentation")
    print(f"   Original p_id: {original_p_id}")
    
    headers = {
        "Authorization": f"Bearer {user_b_token}"
    }
    
    try:
        response = requests.get(
            f"{BASE_URL}/slides/",
            params={"p_id": original_p_id},
            headers=headers,
            timeout=30
        )
        
        print(f"📥 Response status: {response.status_code}")
        
        if response.status_code == 403:
            print(f"\n✅ Access properly denied!")
            print(f"   User B cannot access User A's presentation (as expected)")
            return True
        else:
            print(f"\n⚠️ Unexpected response!")
            print(f"   Expected 403 Forbidden, got {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Request failed: {e}")
        return False


def run_full_test_suite(original_p_id: str, user_a_token: str, user_b_token: str):
    """
    Run complete test suite for clone functionality
    
    Args:
        original_p_id: The p_id to clone
        user_a_token: JWT token for User A (original owner)
        user_b_token: JWT token for User B (cloner)
    """
    print("\n" + "🚀" * 30)
    print("  CLONE PRESENTATION TEST SUITE")
    print("🚀" * 30)
    
    results = {
        "original_exists": False,
        "clone_success": False,
        "view_cloned_slides": False,
        "ownership_verified": False
    }
    
    # Test 1: Verify original presentation exists
    results["original_exists"] = test_view_original_presentation(
        original_p_id, 
        user_a_token
    )
    
    if not results["original_exists"]:
        print("\n❌ Original presentation doesn't exist. Stopping tests.")
        return results
    
    time.sleep(1)
    
    # Test 2: Clone the presentation
    clone_data = test_clone_presentation(original_p_id, user_b_token)
    if clone_data:
        results["clone_success"] = True
        new_p_id = clone_data.get('new_p_id')
        
        time.sleep(1)
        
        # Test 3: View cloned slides
        if new_p_id:
            results["view_cloned_slides"] = test_view_cloned_slides(
                new_p_id,
                user_b_token
            )
            
            time.sleep(1)
    
    # Test 4: Verify ownership (User B should not access User A's original)
    results["ownership_verified"] = test_ownership_verification(
        original_p_id,
        user_b_token
    )
    
    # Print summary
    print_section("TEST SUMMARY")
    print(f"✅ Original presentation exists: {results['original_exists']}")
    print(f"✅ Clone successful: {results['clone_success']}")
    print(f"✅ View cloned slides: {results['view_cloned_slides']}")
    print(f"✅ Ownership verified: {results['ownership_verified']}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print("\n⚠️ SOME TESTS FAILED")
    
    return results


def quick_clone_test(original_p_id: str, user_b_token: str):
    """
    Quick test - just clone and view
    
    Args:
        original_p_id: The p_id to clone
        user_b_token: JWT token for User B
    """
    print("\n" + "🔥" * 30)
    print("  QUICK CLONE TEST")
    print("🔥" * 30)
    
    # Clone
    clone_data = test_clone_presentation(original_p_id, user_b_token)
    
    if clone_data:
        new_p_id = clone_data.get('new_p_id')
        
        time.sleep(1)
        
        # View
        if new_p_id:
            test_view_cloned_slides(new_p_id, user_b_token)
            
            print("\n✅ Quick test complete!")
            print(f"\n📌 Your cloned presentation ID: {new_p_id}")
            print(f"📌 You can now access it at: {BASE_URL}/slides/?p_id={new_p_id}")


if __name__ == "__main__":
    print("\n🔐 CLONE PRESENTATION TEST SCRIPT\n")
    
    # Check if tokens are set
    if "..." in USER_A_TOKEN or "..." in USER_B_TOKEN:
        print("⚠️ WARNING: Please update USER_A_TOKEN and USER_B_TOKEN in the script")
        print("   with actual JWT tokens before running tests.\n")
    
    # Interactive mode
    print("Choose a test mode:")
    print("1. Quick test (clone + view)")
    print("2. Full test suite (all tests)")
    print("3. Manual entry")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        p_id = input("Enter original p_id to clone: ").strip()
        token = input("Enter User B's JWT token: ").strip() or USER_B_TOKEN
        quick_clone_test(p_id, token)
        
    elif choice == "2":
        p_id = input("Enter original p_id to clone: ").strip()
        token_a = input("Enter User A's JWT token (original owner): ").strip() or USER_A_TOKEN
        token_b = input("Enter User B's JWT token (cloner): ").strip() or USER_B_TOKEN
        run_full_test_suite(p_id, token_a, token_b)
        
    elif choice == "3":
        print("\nManual Test:")
        p_id = input("Enter original p_id: ").strip()
        token = input("Enter JWT token: ").strip()
        
        print("\nWhat would you like to test?")
        print("1. Clone presentation")
        print("2. View slides")
        
        test_choice = input("Enter choice (1-2): ").strip()
        
        if test_choice == "1":
            test_clone_presentation(p_id, token)
        elif test_choice == "2":
            test_view_cloned_slides(p_id, token)
    
    else:
        print("❌ Invalid choice")
    
    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60 + "\n")

