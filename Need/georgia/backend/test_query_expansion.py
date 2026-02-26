import sys
import os
import time

# Adjust path to include the app directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services import vertex_ai_service
from app import config

def test_query_variations():
    original_query = "What is the policy for remote work?"
    print(f"\n--- Testing Query Expansion for: '{original_query}' ---")
    
    try:
        if not config.GEMINI_MODEL_NAME:
             print("Skipping test: GEMINI_MODEL_NAME not set.")
             return

        # First Call (Expect Miss)
        print("\n1️⃣  First Call (Should trigger LLM generation)...")
        start_t = time.time()
        variations = vertex_ai_service.generate_query_variations(original_query)
        print(f"Time: {time.time() - start_t:.4f}s")
        print(f"Result: {variations}")

        # Second Call (Expect Hit)
        print("\n2️⃣  Second Call (Should be instant from Redis)...")
        start_t = time.time()
        variations_cached = vertex_ai_service.generate_query_variations(original_query)
        print(f"Time: {time.time() - start_t:.4f}s")
        print(f"Result: {variations_cached}")
            
        if len(variations) >= 1 and variations == variations_cached:
             print("\n✅ TEST PASSED: Variations generated and cached correctly.")
        else:
            print("\n❌ TEST FAILED: Results differ or empty.")

    except Exception as e:
        print(f"❌ TEST FAILED with exception: {e}")

if __name__ == "__main__":
    test_query_variations()
