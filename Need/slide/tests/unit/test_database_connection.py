#!/usr/bin/env python3
"""
Test database connection with environment variables.
"""

import os
from dotenv import load_dotenv
from core.database import get_mongo_client

# Load environment variables
load_dotenv()

print("🔍 Testing Database Connection")
print("=" * 50)

# Check environment variables
mongodb_url = os.getenv("MONGODB_URL")
print(f"📋 MONGODB_URL: {mongodb_url[:50]}..." if mongodb_url else "❌ MONGODB_URL not set")

try:
    # Test database connection
    client = get_mongo_client()
    db = client["slide_creator_db"]
    
    # Test basic connection
    collections = db.list_collection_names()
    print(f"✅ Database connection successful!")
    print(f"📋 Collections: {collections}")
    
    # Test creating a test document
    test_doc = {
        "test": True,
        "timestamp": "2025-01-25T10:30:00Z"
    }
    
    result = db.test_collection.insert_one(test_doc)
    print(f"✅ Test document created: {result.inserted_id}")
    
    # Clean up
    db.test_collection.delete_one({"_id": result.inserted_id})
    print(f"✅ Test document cleaned up")
    
except Exception as e:
    print(f"❌ Database connection failed: {e}")
    print(f"🔍 Error type: {type(e).__name__}")
