#!/bin/bash

echo "🧪 Running Client Tests..."

echo "1. HTML Client Test:"
echo "   Open tests/client/client_test.html in browser"

echo "2. Python Client Test:"
python tests/unit/client_test.py

echo "3. Node.js Client Test:"
node tests/client/client_test.js

echo "✅ All client tests completed"
