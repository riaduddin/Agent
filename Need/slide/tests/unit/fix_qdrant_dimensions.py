"""
Fix Qdrant dimension mismatch by recreating collection
"""
from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv

load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = "browser_research"

print("🔧 Fixing Qdrant Collection Dimensions...\n")

try:
    # Connect to Qdrant
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    print(f"✅ Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
    
    # Check if collection exists
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]
    
    if COLLECTION_NAME in collection_names:
        # Get collection info
        info = client.get_collection(COLLECTION_NAME)
        print(f"\n📊 Current Collection Info:")
        print(f"   Name: {COLLECTION_NAME}")
        print(f"   Vectors: {info.vectors_count}")
        print(f"   Points: {info.points_count}")
        print(f"   Dimensions: {info.config.params.vectors.size}")
        
        # Check dimension
        current_dim = info.config.params.vectors.size
        expected_dim = 3072  # Gemini text-embedding-004
        
        if current_dim != expected_dim:
            print(f"\n⚠️  DIMENSION MISMATCH DETECTED!")
            print(f"   Current: {current_dim} dimensions")
            print(f"   Expected: {expected_dim} dimensions (Gemini text-embedding-004)")
            print(f"\n🗑️  Deleting collection...")
            
            # Delete collection
            client.delete_collection(COLLECTION_NAME)
            print(f"   ✅ Collection deleted")
        else:
            print(f"\n✅ Dimensions are correct ({current_dim})")
            print(f"   Collection has {info.points_count} points")
            
            response = input("\n❓ Do you want to delete and recreate anyway? (y/N): ")
            if response.lower() == 'y':
                print(f"🗑️  Deleting collection...")
                client.delete_collection(COLLECTION_NAME)
                print(f"   ✅ Collection deleted")
            else:
                print(f"   ℹ️  Keeping existing collection")
                exit(0)
    else:
        print(f"\nℹ️  Collection '{COLLECTION_NAME}' does not exist yet")
    
    # Create new collection with correct dimensions
    from qdrant_client.models import Distance, VectorParams
    
    print(f"\n🆕 Creating new collection with 3072 dimensions...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=3072,  # Gemini text-embedding-004
            distance=Distance.COSINE
        )
    )
    
    # Verify
    info = client.get_collection(COLLECTION_NAME)
    print(f"✅ Collection created successfully!")
    print(f"   Name: {COLLECTION_NAME}")
    print(f"   Dimensions: {info.config.params.vectors.size}")
    print(f"   Distance: COSINE")
    
    print(f"\n🎉 Qdrant collection is ready for Gemini embeddings!")
    print(f"   You can now run your presentation generation service.")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    print(f"\n💡 Troubleshooting:")
    print(f"   1. Make sure Qdrant is running: docker run -p 6333:6333 qdrant/qdrant")
    print(f"   2. Check QDRANT_HOST and QDRANT_PORT in .env")
    print(f"   3. Verify Qdrant is accessible: curl http://localhost:6333/")

