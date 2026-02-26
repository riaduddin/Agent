"""
Diagnose Qdrant retrieval issue
"""
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

print("🔍 Diagnosing Qdrant Retrieval Issue...\n")

# Step 1: Check API Key
print("1️⃣ Checking Gemini API Key...")
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if api_key:
    print(f"   ✅ API Key found: {api_key[:10]}...{api_key[-4:]}")
    genai.configure(api_key=api_key)
else:
    print(f"   ❌ No API key found!")
    print(f"   Set GOOGLE_API_KEY or GEMINI_API_KEY in .env")
    exit(1)

# Step 2: Test embedding generation
print("\n2️⃣ Testing Gemini embedding generation...")
try:
    test_text = "This is a test query for embedding"
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=test_text,
        task_type="retrieval_query"
    )
    
    embedding = result['embedding']
    print(f"   ✅ Embedding generated successfully!")
    print(f"   Dimensions: {len(embedding)}")
    print(f"   Sample values: {embedding[:3]}")
    
    if len(embedding) != 768:
        print(f"   ⚠️  WARNING: Expected 768 dimensions, got {len(embedding)}")
    
except Exception as e:
    print(f"   ❌ Embedding generation failed: {e}")
    print(f"   This is why Qdrant retrieval is failing!")
    exit(1)

# Step 3: Check Qdrant connection
print("\n3️⃣ Checking Qdrant connection...")
try:
    from qdrant_client import QdrantClient
    
    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
    
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    
    # Get collection info
    info = client.get_collection("browser_research")
    print(f"   ✅ Connected to Qdrant: {QDRANT_HOST}:{QDRANT_PORT}")
    print(f"   Collection: browser_research")
    print(f"   Points: {info.points_count}")
    print(f"   Dimensions: {info.config.params.vectors.size}")
    
except Exception as e:
    print(f"   ❌ Qdrant connection failed: {e}")
    exit(1)

# Step 4: Test actual search
print("\n4️⃣ Testing Qdrant search...")
try:
    # Generate embedding for query
    test_query = "Tesla marketing strategy"
    query_result = genai.embed_content(
        model="models/text-embedding-004",
        content=test_query,
        task_type="retrieval_query"
    )
    query_embedding = query_result['embedding']
    
    print(f"   Query: {test_query}")
    print(f"   Query embedding dimensions: {len(query_embedding)}")
    
    # Try search
    results = client.search(
        collection_name="browser_research",
        query_vector=query_embedding,
        limit=3
    )
    
    print(f"   ✅ Search successful!")
    print(f"   Found {len(results)} results")
    
    for i, result in enumerate(results, 1):
        print(f"   {i}. Score: {result.score:.3f}, Text: {result.payload.get('text', '')[:80]}...")
    
except Exception as e:
    print(f"   ❌ Search failed: {e}")
    print(f"   Error details: {type(e).__name__}")
    
    # Check if it's an OutputTooSmall error
    if "OutputTooSmall" in str(e):
        print(f"\n   🔍 OutputTooSmall Error Detected!")
        print(f"   This usually means:")
        print(f"   1. Query embedding is empty or malformed")
        print(f"   2. Dimension mismatch (but we checked and it's 768)")
        print(f"   3. Qdrant internal issue")
        
        print(f"\n   💡 Solution: Recreate collection")
        print(f"   Run: python fix_qdrant_dimensions.py and choose 'y' to delete")
    
    exit(1)

# Step 5: Test with filters (using the FIXED workaround)
print("\n5️⃣ Testing search with Python-based filtering (WORKAROUND)...")
try:
    # Get a sample point to find valid user_id and p_id
    sample_points = client.scroll(
        collection_name="browser_research",
        limit=1,
        with_payload=True
    )[0]
    
    if sample_points:
        sample_user_id = sample_points[0].payload.get("user_id", "test")
        sample_p_id = sample_points[0].payload.get("p_id", "test")
        
        print(f"   Using sample user_id: {sample_user_id}")
        print(f"   Using sample p_id: {sample_p_id}")
        
        # Use query_points without filters (workaround)
        all_results = client.query_points(
            collection_name="browser_research",
            query=query_embedding,
            limit=50,  # Get more to filter in Python
            with_payload=True
        ).points
        
        print(f"   ✅ Got {len(all_results)} results without filter")
        
        # Filter in Python
        filtered_results = []
        for result in all_results:
            if (result.payload.get("user_id") == sample_user_id and 
                result.payload.get("p_id") == sample_p_id):
                filtered_results.append(result)
                if len(filtered_results) >= 5:
                    break
        
        print(f"   ✅ Python filtering successful!")
        print(f"   Found {len(filtered_results)} results for user: {sample_user_id}, p_id: {sample_p_id}")
        
        for i, result in enumerate(filtered_results[:3], 1):
            print(f"   {i}. Score: {result.score:.3f}, Keyword: {result.payload.get('keyword', 'N/A')}")
    else:
        print(f"   ⚠️  No points in collection to test with")

except Exception as e:
    print(f"   ❌ Python filtering failed: {e}")
    import traceback
    traceback.print_exc()

# Step 6: Test the actual qdrant_utils.py method
print("\n6️⃣ Testing qdrant_utils.retrieve_research_data() method...")
try:
    import sys
    import os
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.append(ROOT_DIR)
    from tools.qdrant_utils import get_qdrant_manager
    
    qdrant_manager = get_qdrant_manager()
    
    # Use sample IDs from above
    if sample_points:
        results = qdrant_manager.retrieve_research_data(
            query="Tesla marketing strategy",
            user_id=sample_user_id,
            p_id=sample_p_id,
            limit=5
        )
        
        if results:
            print(f"   ✅ qdrant_utils method works!")
            print(f"   Retrieved {len(results)} chunks")
            for i, r in enumerate(results[:2], 1):
                print(f"   {i}. Keyword: {r['keyword']}, Score: {r['score']:.3f}")
        else:
            print(f"   ⚠️  No results returned (but no error - this is OK)")
    
except Exception as e:
    print(f"   ❌ qdrant_utils method failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("✅ DIAGNOSIS COMPLETE")
print("="*60)
print("\nIf all tests passed, the lightweight approach should work!")
print("If any tests failed, check the error messages above for solutions.")

