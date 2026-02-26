#!/usr/bin/env python3
"""
Test environment variable loading.
"""

import os
from dotenv import load_dotenv

print("🔍 Testing Environment Variable Loading")
print("=" * 50)

# Load .env file
load_dotenv()

# Check environment variables
env_vars = [
    "MONGODB_URL",
    "JWT_SECRET", 
    "REDIS_URL",
    "POSTGRES_URL"
]

for var in env_vars:
    value = os.getenv(var)
    if value:
        print(f"✅ {var}: {value[:50]}...")
    else:
        print(f"❌ {var}: NOT SET")

print("\n🔍 Checking .env file content:")
try:
    with open('.env', 'r') as f:
        content = f.read()
        print(f"📋 .env file exists and has {len(content)} characters")
        print(f"📋 First 200 characters: {content[:200]}...")
except Exception as e:
    print(f"❌ Error reading .env file: {e}")
