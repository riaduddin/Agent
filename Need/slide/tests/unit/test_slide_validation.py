"""
Quick test script to verify slide validation database queries
"""
from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

# Connect to MongoDB
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
client = MongoClient(MONGODB_URI)
db = client["slide_creator_db"]

# Test p_id
p_id = "68f39e5c5f69cc247959e307"

print("=" * 60)
print("SLIDE VALIDATION TEST")
print("=" * 60)
print(f"\nTesting p_id: {p_id}")
print()

# Test 1: Count total slides
print("1. Counting slides...")
total_slides = db.slide_html.count_documents({"p_id": p_id})
print(f"   Total slides found: {total_slides}")
print()

# Test 2: List all slides
print("2. Listing all slides:")
slides = list(db.slide_html.find(
    {"p_id": p_id},
    {"slide_number": 1, "slide_index": 1, "_id": 0}
).sort("slide_number", 1))

for slide in slides:
    print(f"   - Slide {slide.get('slide_number')} (index: {slide.get('slide_index')})")
print()

# Test 3: Check if slide 3 exists
print("3. Checking if slide 3 exists...")
slide_3 = db.slide_html.find_one({
    "p_id": p_id,
    "slide_number": 3
})
if slide_3:
    print(f"   ✅ Slide 3 EXISTS")
    print(f"   - slide_number: {slide_3.get('slide_number')}")
    print(f"   - slide_index: {slide_3.get('slide_index')}")
else:
    print(f"   ❌ Slide 3 NOT FOUND")
print()

# Test 4: Check presentation document
print("4. Checking presentation document...")
presentation = db.presentations.find_one({"p_id": p_id})
if presentation:
    print(f"   ✅ Presentation EXISTS")
    print(f"   - p_id: {presentation.get('p_id')}")
    print(f"   - title: {presentation.get('title')}")
    print(f"   - total_slides field: {presentation.get('total_slides', 'NOT SET')}")
else:
    print(f"   ❌ Presentation NOT FOUND")
print()

# Test 5: Validation logic simulation
print("5. Simulating validation for slide [3]...")
slide_numbers = [3]
invalid_slides = []

for slide_num in slide_numbers:
    slide_exists = db.slide_html.find_one({
        "p_id": p_id,
        "slide_number": slide_num
    })
    if not slide_exists:
        invalid_slides.append(slide_num)

if invalid_slides:
    print(f"   ❌ VALIDATION FAILED")
    print(f"   Invalid slides: {invalid_slides}")
    print(f"   Error: Slide(s) {invalid_slides} not found. Presentation has {total_slides} slides (1-{total_slides}).")
else:
    print(f"   ✅ VALIDATION PASSED")
    print(f"   All requested slides exist")

print()
print("=" * 60)
print("TEST COMPLETE")
print("=" * 60)


