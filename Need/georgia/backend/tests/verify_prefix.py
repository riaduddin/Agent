import os
import sys

# Add the backend directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set the environment variable
os.environ['ROUTE_PREFIX'] = '/test-prefix'

from app import create_app

def test_routes():
    app = create_app()
    prefix = '/test-prefix'
    
    expected_routes = [
        f"{prefix}/health",
        f"{prefix}/debug/routes",
        f"{prefix}/backend/api/v1/auth/login",
        f"{prefix}/backend/api/v2/system/health" # Assuming this exists based on register_blueprint
    ]
    
    actual_routes = [str(rule) for rule in app.url_map.iter_rules()]
    
    print(f"Checking for prefix: {prefix}")
    all_passed = True
    for expected in expected_routes:
        found = any(expected in actual for actual in actual_routes)
        if found:
            print(f"✅ Found expected route: {expected}")
        else:
            print(f"❌ Could NOT find expected route: {expected}")
            all_passed = False
            
    if all_passed:
        print("\n✨ All prefix tests passed!")
    else:
        print("\n⚠️ Some prefix tests failed.")
        print("\nAll registered routes:")
        for rule in actual_routes:
            print(f"  - {rule}")

if __name__ == "__main__":
    test_routes()