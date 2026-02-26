"""
Test the image search tool
"""
import sys
import os
# Add root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tools.image_search import search_images

print("🧪 Testing Image Search Tool...\n")

# Test 1: Single query
print("1️⃣ Testing single image search...")
try:
    result = search_images(
        search_queries=["Tesla electric car"],
        count_per_query=3
    )
    print(result)
    print("\n✅ Single query test passed!")
except Exception as e:
    print(f"❌ Failed: {e}")

# Test 2: Multiple queries
print("\n2️⃣ Testing multiple image searches...")
try:
    result = search_images(
        search_queries=["AI technology concept", "healthcare innovation"],
        count_per_query=2
    )
    print(result)
    print("\n✅ Multiple query test passed!")
except Exception as e:
    print(f"❌ Failed: {e}")

print("\n" + "="*60)
print("✅ Image search tool testing complete!")
print("="*60)

