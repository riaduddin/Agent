#!/usr/bin/env python3
"""
Test JWT secret loading in socketio_manager.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("🔍 Testing JWT Secret Loading")
print("=" * 50)

# Test 1: Direct environment variable access
jwt_secret_direct = os.getenv("JWT_SECRET")
print(f"1. Direct JWT_SECRET: {jwt_secret_direct[:50]}..." if jwt_secret_direct else "❌ JWT_SECRET not found")

# Test 2: Test the exact code from socketio_manager
jwt_secret_manager = os.getenv("JWT_SECRET", "your-secret-key-here")
print(f"2. Manager JWT_SECRET: {jwt_secret_manager[:50]}..." if jwt_secret_manager else "❌ JWT_SECRET not found")

# Test 3: Test JWT token validation
import jwt

token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJfaWQiOiI2ODgwODdlMzA1MTk0OTc2YWVhOTk0MDMiLCJzdWIiOiI2ODgwODdlMzA1MTk0OTc2YWVhOTk0MDMiLCJlbWFpbCI6InJycmlhZHVkZGluQGdtYWlsLmNvbSIsInBhY2thZ2UiOiJ1bmxpbWl0ZWQiLCJpc192ZXJpZmllZCI6dHJ1ZSwicm9sZSI6InVzZXIiLCJpYXQiOjE3NjEzNzczMzAsImV4cCI6MTc2MTQ2MzczMH0.dUeYJu4jNbaSfN8jRloiFiRTie1WkJ1prWtMe-_RQLg"

try:
    payload = jwt.decode(token, jwt_secret_manager, algorithms=["HS256"])
    print(f"3. JWT validation: ✅ SUCCESS")
    print(f"   User ID: {payload.get('sub')}")
except Exception as e:
    print(f"3. JWT validation: ❌ FAILED - {e}")

print(f"\n💡 The server needs to be restarted to pick up the environment variable change!")
print(f"   Stop the server (Ctrl+C) and restart it with: python main.py")
