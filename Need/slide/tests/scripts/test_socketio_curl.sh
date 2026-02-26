#!/bin/bash

# Socket.IO cURL Test Script
# This tests the Socket.IO endpoint using cURL

echo "🧪 Socket.IO cURL Test"
echo "======================"

# Configuration
BASE_URL="http://localhost:8060"
P_ID="test_presentation_id"
JWT_TOKEN="your_jwt_token_here"

# Check if JWT token is set
if [ "$JWT_TOKEN" = "your_jwt_token_here" ]; then
    echo "⚠️ Please set JWT_TOKEN environment variable"
    echo "   Example: export JWT_TOKEN=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
    exit 1
fi

echo "🔗 Base URL: $BASE_URL"
echo "📋 P_ID: $P_ID"
echo "🔑 JWT Token: ${JWT_TOKEN:0:20}..."
echo ""

# Test 1: Socket.IO polling transport
echo "📡 Test 1: Socket.IO Polling Transport"
echo "--------------------------------------"
curl -v "$BASE_URL/socket.io/?p_id=$P_ID&token=$JWT_TOKEN&EIO=4&transport=polling"
echo ""

# Test 2: Socket.IO WebSocket transport
echo "📡 Test 2: Socket.IO WebSocket Transport"
echo "----------------------------------------"
curl -v "$BASE_URL/socket.io/?p_id=$P_ID&token=$JWT_TOKEN&EIO=4&transport=websocket"
echo ""

# Test 3: Create presentation
echo "📝 Test 3: Create Presentation"
echo "------------------------------"
curl -X POST "$BASE_URL/create-presentation" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $JWT_TOKEN" \
  -d '{
    "message": "Create a presentation about AI in Healthcare with 5 slides",
    "file_urls": [
      "https://example.com/healthcare_report.pdf",
      "https://example.com/ai_guidelines.docx"
    ]
  }'
echo ""

# Test 4: Start presentation (if p_id is available)
if [ "$P_ID" != "test_presentation_id" ]; then
    echo "🚀 Test 4: Start Presentation"
    echo "-----------------------------"
    curl -X POST "$BASE_URL/start-presentation/$P_ID" \
      -H "Authorization: Bearer $JWT_TOKEN"
    echo ""
fi

echo "✅ cURL tests completed!"
echo ""
echo "📋 Notes:"
echo "  • Socket.IO uses WebSocket protocol, so cURL can only test the initial handshake"
echo "  • For real-time data, use the browser client or Python client"
echo "  • The polling transport should return a session ID"
echo "  • The WebSocket transport should return a 101 Switching Protocols"
