"""
Test script for JWT Authentication
Tests the authentication middleware with the provided token
"""

import jwt
import os
from dotenv import load_dotenv

load_dotenv()

# Test token from user
TEST_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJfaWQiOiI2ODgxYjhjMzE2ZjViODk0MzZkYzBiNzMiLCJzdWIiOiI2ODgxYjhjMzE2ZjViODk0MzZkYzBiNzMiLCJlbWFpbCI6Im1obWFoZWRpMDAwQGdtYWlsLmNvbSIsInBhY2thZ2UiOiJ2YWx1ZV9wbGFuIiwiaXNfdmVyaWZpZWQiOnRydWUsInJvbGUiOiJ1c2VyIiwiaWF0IjoxNzYwNzYzNDc5fQ.PTrxANu4OxcuPbY2wMEBoX6RU9KSVg19nxzvELKNN04"

# Expected payload
EXPECTED_PAYLOAD = {
    "_id": "6881b8c316f5b89436dc0b73",
    "sub": "6881b8c316f5b89436dc0b73",
    "email": "mhmahedi000@gmail.com",
    "package": "value_plan",
    "is_verified": True,
    "role": "user",
    "iat": 1760763479
}


def test_decode_token():
    """Test JWT token decoding"""
    print("=" * 60)
    print("JWT Authentication Test")
    print("=" * 60)
    
    # Check if JWT_SECRET is configured
    jwt_secret = os.getenv("JWT_SECRET")
    jwt_algorithm = os.getenv("JWT_ALGORITHM", "HS256")
    
    print(f"\n1. Configuration Check:")
    print(f"   JWT_SECRET: {'✓ Configured' if jwt_secret else '✗ NOT CONFIGURED'}")
    print(f"   JWT_ALGORITHM: {jwt_algorithm}")
    
    if not jwt_secret:
        print("\n❌ ERROR: JWT_SECRET not configured!")
        print("   Please add JWT_SECRET to your .env file")
        print("   See JWT_SETUP_GUIDE.md for details")
        return False
    
    print(f"\n2. Test Token:")
    print(f"   Token (first 50 chars): {TEST_TOKEN[:50]}...")
    
    try:
        # Decode without verification first to see the payload
        print(f"\n3. Decoding Token (without verification):")
        unverified = jwt.decode(TEST_TOKEN, options={"verify_signature": False})
        print(f"   Payload:")
        for key, value in unverified.items():
            print(f"     {key}: {value}")
        
        # Now decode with verification
        print(f"\n4. Decoding Token (with verification):")
        payload = jwt.decode(
            TEST_TOKEN,
            jwt_secret,
            algorithms=[jwt_algorithm]
        )
        print(f"   ✓ Token verified successfully!")
        
        # Verify expected fields
        print(f"\n5. Validating Payload Fields:")
        user_id = payload.get("_id") or payload.get("sub")
        email = payload.get("email")
        is_verified = payload.get("is_verified")
        package = payload.get("package")
        role = payload.get("role")
        
        print(f"   User ID: {user_id}")
        print(f"   Email: {email}")
        print(f"   Verified: {is_verified}")
        print(f"   Package: {package}")
        print(f"   Role: {role}")
        
        # Validation checks
        print(f"\n6. Validation Results:")
        checks = [
            ("User ID present", bool(user_id)),
            ("Email present", bool(email)),
            ("User is verified", is_verified == True),
            ("Has package info", bool(package)),
            ("Has role info", bool(role)),
        ]
        
        all_passed = True
        for check_name, passed in checks:
            status = "✓" if passed else "✗"
            print(f"   {status} {check_name}")
            if not passed:
                all_passed = False
        
        if all_passed:
            print(f"\n✅ All checks passed! Authentication will work correctly.")
            return True
        else:
            print(f"\n⚠️ Some checks failed. Authentication may not work as expected.")
            return False
            
    except jwt.ExpiredSignatureError:
        print(f"\n❌ ERROR: Token has expired!")
        print("   You'll need a new token from the authentication service")
        return False
    
    except jwt.InvalidSignatureError:
        print(f"\n❌ ERROR: Invalid token signature!")
        print("   Possible reasons:")
        print("   1. JWT_SECRET in .env doesn't match the secret used to create the token")
        print("   2. Token has been tampered with")
        print("\n   Please verify you're using the correct JWT_SECRET")
        return False
    
    except jwt.InvalidTokenError as e:
        print(f"\n❌ ERROR: Invalid token - {str(e)}")
        return False
    
    except Exception as e:
        print(f"\n❌ ERROR: Unexpected error - {str(e)}")
        return False


def print_api_usage_examples():
    """Print example API calls"""
    print("\n" + "=" * 60)
    print("API Usage Examples")
    print("=" * 60)
    
    print("\n1. Create Presentation (curl):")
    print("""
curl -X POST "http://localhost:8000/create-presentation-sse" \\
  -H "Authorization: Bearer YOUR_TOKEN_HERE" \\
  -H "Content-Type: application/json" \\
  -d '{
    "message": "Create a presentation about AI",
    "file_urls": []
  }'
    """)
    
    print("\n2. Stream Presentation (curl):")
    print("""
curl -N "http://localhost:8000/stream/PRESENTATION_ID" \\
  -H "Authorization: Bearer YOUR_TOKEN_HERE"
    """)
    
    print("\n3. List Presentations (curl):")
    print("""
curl "http://localhost:8000/presentations/" \\
  -H "Authorization: Bearer YOUR_TOKEN_HERE"
    """)
    
    print("\n4. Upload File (curl):")
    print("""
curl -X POST "http://localhost:8000/upload-file" \\
  -H "Authorization: Bearer YOUR_TOKEN_HERE" \\
  -F "files=@/path/to/file.pdf"
    """)
    
    print("\n5. JavaScript/Fetch Example:")
    print("""
const token = 'YOUR_TOKEN_HERE';

// Create presentation
const response = await fetch('http://localhost:8000/create-presentation-sse', {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    message: 'Create a presentation about AI',
    file_urls: []
  })
});

const data = await response.json();
console.log('Presentation ID:', data.presentation_id);

// Stream events
const eventSource = new EventSource(
  `http://localhost:8000/stream/${data.presentation_id}`,
  {
    headers: {
      'Authorization': `Bearer ${token}`
    }
  }
);

eventSource.addEventListener('chunk', (event) => {
  const data = JSON.parse(event.data);
  console.log('Received chunk:', data);
});

eventSource.addEventListener('completed', (event) => {
  console.log('Presentation completed!');
  eventSource.close();
});
    """)


if __name__ == "__main__":
    print("\n🔐 JWT Authentication Test Script\n")
    
    success = test_decode_token()
    
    if success:
        print_api_usage_examples()
    
    print("\n" + "=" * 60)
    print("Test Complete")
    print("=" * 60)
    
    if not success:
        print("\n⚠️ Please fix the issues above before using the API")
        print("See JWT_SETUP_GUIDE.md for setup instructions")
    else:
        print("\n✅ Ready to use! Start the server and make authenticated requests")

