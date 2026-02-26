"""
Qdrant Vector Database Utilities for Browser Research Storage and Retrieval
"""
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import List, Dict, Optional, Tuple
import os
from dotenv import load_dotenv
import uuid
import logging
import google.generativeai as genai

load_dotenv()
logger = logging.getLogger(__name__)

# Qdrant Configuration
QDRANT_URL = os.getenv("QDRANT_URL", None)  # For cloud instances (e.g., https://xxx.cloud.qdrant.io:6333)
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")  # For local instances
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))  # For local instances
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
COLLECTION_NAME = "browser_research"

# Gemini Embedding Model Configuration
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")  # Latest Gemini embedding (768 dimensions)
EMBEDDING_DIM = os.getenv("EMBEDDING_DIM", 3072)  # Dimension for text-embedding-004

# Chunking Configuration 
CHUNK_SIZE = 1000  # Characters per chunk - optimal for retrieval
CHUNK_OVERLAP = 200  # Overlap between chunks to maintain context


class QdrantManager:
    """Manages Qdrant vector database operations for browser research storage"""
    
    def __init__(self):
        """Initialize Qdrant client and Gemini embedding model"""
        try:
            # Initialize Qdrant client with support for both cloud and local instances
            if QDRANT_URL:
                # Cloud instance - use URL parameter
                if QDRANT_API_KEY:
                    self.client = QdrantClient(
                        url=QDRANT_URL,
                        api_key=QDRANT_API_KEY,
                        timeout=60,  # 60 second timeout to handle slow connections
                        prefer_grpc=False  # Use HTTP instead of gRPC for better compatibility
                    )
                else:
                    self.client = QdrantClient(
                        url=QDRANT_URL,
                        timeout=60,
                        prefer_grpc=False
                    )
                logger.info(f"✅ Using Qdrant Cloud: {QDRANT_URL}")
            else:
                # Local instance - use host and port
                if QDRANT_API_KEY:
                    self.client = QdrantClient(
                        host=QDRANT_HOST,
                        port=QDRANT_PORT,
                        api_key=QDRANT_API_KEY,
                        timeout=60,  # 60 second timeout to handle slow connections
                        prefer_grpc=False  # Use HTTP instead of gRPC for better Windows compatibility
                    )
                else:
                    self.client = QdrantClient(
                        host=QDRANT_HOST, 
                        port=QDRANT_PORT,
                        timeout=60,
                        prefer_grpc=False
                    )
                logger.info(f"✅ Using Qdrant Local: {QDRANT_HOST}:{QDRANT_PORT}")
            
            # Initialize Gemini API for embeddings
            if GEMINI_API_KEY:
                genai.configure(api_key=GEMINI_API_KEY)
            else:
                logger.warning("⚠️ No Gemini API key found. Embeddings may fail.")
            
            self.embedding_model = EMBEDDING_MODEL
            self.embedding_dim = EMBEDDING_DIM
            
            # Ensure collection exists
            self._ensure_collection()
            
            connection_info = QDRANT_URL if QDRANT_URL else f"{QDRANT_HOST}:{QDRANT_PORT}"
            logger.info(f"✅ Qdrant initialized: {connection_info}, Model: {EMBEDDING_MODEL}, Collection: {COLLECTION_NAME}")
        except Exception as e:
            logger.error(f"❌ Qdrant initialization failed: {e}")
            raise
    
    def _ensure_collection(self):
        """Create collection if it doesn't exist or has wrong dimensions"""
        try:
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if COLLECTION_NAME in collection_names:
                # Check dimensions of existing collection
                collection_info = self.client.get_collection(COLLECTION_NAME)
                current_dim = collection_info.config.params.vectors.size
                
                if current_dim != self.embedding_dim:
                    logger.warning(f"⚠️ Dimension mismatch in '{COLLECTION_NAME}': expected {self.embedding_dim}, found {current_dim}. Recreating...")
                    self.client.delete_collection(COLLECTION_NAME)
                    collection_names.remove(COLLECTION_NAME)
                else:
                    logger.info(f"✅ Qdrant collection exists with correct dimensions: {COLLECTION_NAME} ({current_dim})")
            
            if COLLECTION_NAME not in collection_names:
                self.client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"✅ Created Qdrant collection: {COLLECTION_NAME} with {self.embedding_dim} dimensions")
        except Exception as e:
            logger.error(f"❌ Collection setup failed: {e}")
            raise
    
    def health_check(self) -> Tuple[bool, str]:
        """
        Check if Qdrant is accessible and working
        
        Returns:
            Tuple of (is_healthy: bool, message: str)
        """
        try:
            # Try to get collections - this tests the connection
            collections = self.client.get_collections()
            
            # Check if our collection exists
            collection_names = [c.name for c in collections.collections]
            if COLLECTION_NAME in collection_names:
                # Try to get collection info to verify it's accessible
                collection_info = self.client.get_collection(COLLECTION_NAME)
                return True, f"✅ Qdrant is healthy. Collection '{COLLECTION_NAME}' exists with {collection_info.points_count} points."
            else:
                return True, f"✅ Qdrant is accessible. Collection '{COLLECTION_NAME}' will be created on first use."
        except Exception as e:
            error_msg = str(e)
            connection_info = QDRANT_URL if QDRANT_URL else f"{QDRANT_HOST}:{QDRANT_PORT}"
            return False, f"❌ Qdrant health check failed ({connection_info}): {error_msg}"
    
    def chunk_text(self, text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
        """
        Split text into overlapping chunks for optimal retrieval
        
        Args:
            text: The text to chunk
            chunk_size: Maximum characters per chunk
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of text chunks
        """
        if not text or len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Get chunk
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary if possible
            if end < len(text):
                # Look for sentence endings near the chunk boundary
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                last_question = chunk.rfind('?')
                last_exclaim = chunk.rfind('!')
                
                # Use the latest sentence boundary
                boundary = max(last_period, last_newline, last_question, last_exclaim)
                if boundary > chunk_size * 0.7:  # Only break if boundary is not too early
                    chunk = chunk[:boundary + 1]
                    end = start + boundary + 1
            
            chunks.append(chunk.strip())
            
            # Move to next chunk with overlap
            start = end - overlap
            
            # Avoid infinite loop
            if start >= len(text):
                break
        
        return chunks
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using Gemini API with retry logic"""
        import time
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                result = genai.embed_content(
                    model=self.embedding_model,
                    content=text,
                    task_type="retrieval_document"  # Optimized for retrieval
                )
                return result['embedding']
            except (ConnectionError, ConnectionResetError, OSError) as e:
                if attempt < max_retries - 1:
                    wait_time = 1 * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"⚠️ Embedding API connection error (attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"❌ Failed to generate embedding after {max_retries} attempts: {e}")
                    raise
            except Exception as e:
                logger.error(f"❌ Failed to generate embedding: {e}")
                raise
    
    def store_research_data(
        self,
        text: str,
        user_id: str,
        p_id: str,
        keyword: str,
        sources: List[Dict[str, str]] = None,
        metadata: Dict = None
        ) -> int:
        """
        Store browser research data in Qdrant with chunking
        
        Args:
            text: The research text to store
            user_id: User ID for filtering
            p_id: Presentation ID for filtering
            keyword: The search keyword/topic
            sources: List of source dicts with 'title' and 'url'
            metadata: Additional metadata to store
            
        Returns:
            Number of chunks stored
        """
        try:
            # Chunk the text
            chunks = self.chunk_text(text)
            
            # Prepare points for Qdrant
            points = []
            
            for idx, chunk in enumerate(chunks):
                # Generate embedding using Gemini
                embedding = self.generate_embedding(chunk)
                
                # Prepare payload (metadata)
                payload = {
                    "user_id": user_id,
                    "p_id": p_id,
                    "keyword": keyword,
                    "text": chunk,
                    "chunk_index": idx,
                    "total_chunks": len(chunks),
                    "sources": sources or [],
                }
                
                # Add any additional metadata
                if metadata:
                    payload.update(metadata)
                
                # Create point
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload=payload
                )
                points.append(point)
            
            # Upload to Qdrant
            self.client.upsert(
                collection_name=COLLECTION_NAME,
                points=points
            )
            
            logger.info(f"✅ Stored {len(chunks)} chunks for keyword '{keyword}' (user: {user_id}, p_id: {p_id})")
            return len(chunks)
            
        except Exception as e:
            logger.error(f"❌ Failed to store research data: {e}")
            raise

    def store_file_context(
        self,
        text: str,
        user_id: str,
        p_id: str
        ) -> int:
        """
        Store file_context text with high-priority metadata for retrieval.
        Uses keyword="file_context" and adds file_context metadata for easy filtering.
        """
        return self.store_research_data(
            text=text,
            user_id=user_id,
            p_id=p_id,
            keyword="file_context",
            sources=[{"title": "uploaded_file_context", "url": ""}],
            metadata={
                "source": "file_context",
                "priority": "high",
                "file_context": True  # Metadata field for easy filtering during retrieval
            }
        )
    
    def retrieve_research_data(
        self,
        query: str,
        user_id: str,
        p_id: str,
        limit: int = 10
        ) -> List[Dict]:
        """
        Retrieve relevant research data based on query and filters
        
        Args:
            query: The search query
            user_id: User ID to filter by
            p_id: Presentation ID to filter by
            limit: Maximum number of results
            
        Returns:
            List of relevant research chunks with metadata
        """
        try:
            # Generate query embedding using Gemini with query task type and retry logic
            logger.info(f"🔍 Generating embedding for query: '{query}'")
            
            import time
            max_retries = 3
            query_embedding = None
            
            for attempt in range(max_retries):
                try:
                    result = genai.embed_content(
                        model=self.embedding_model,
                        content=query,
                        task_type="retrieval_query"  # Optimized for query
                    )
                    query_embedding = result['embedding']
                    logger.info(f"✅ Query embedding generated: {len(query_embedding)} dimensions")
                    break  # Success, exit retry loop
                except (ConnectionError, ConnectionResetError, OSError) as e:
                    if attempt < max_retries - 1:
                        wait_time = 1 * (2 ** attempt)
                        logger.warning(f"⚠️ Query embedding connection error (attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"❌ Failed to generate query embedding after {max_retries} attempts")
                        raise
            
            if query_embedding is None:
                raise Exception("Failed to generate query embedding")
            
            # Try search WITHOUT filters first (as workaround for Qdrant bug)
            # Then filter in Python
            logger.info(f"🔍 Searching Qdrant (without filters as workaround)...")
            
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            
            # Use query_points instead of deprecated search
            results = self.client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_embedding,
                limit=limit * 5,  # Get more results to filter in Python
                with_payload=True
            ).points
            
            logger.info(f"✅ Got {len(results)} results from Qdrant")
            
            # Filter results by user_id and p_id in Python (workaround for Qdrant filter bug)
            filtered_results = []
            for result in results:
                result_user_id = result.payload.get("user_id", "")
                result_p_id = result.payload.get("p_id", "")
                
                if result_user_id == user_id and result_p_id == p_id:
                    filtered_results.append(result)
                    if len(filtered_results) >= limit:
                        break
            
            logger.info(f"✅ Filtered to {len(filtered_results)} results for user: {user_id}, p_id: {p_id}")
            
            # Format results
            retrieved_data = []
            for result in filtered_results:
                retrieved_data.append({
                    "text": result.payload.get("text", ""),
                    "keyword": result.payload.get("keyword", ""),
                    "sources": result.payload.get("sources", []),
                    "score": result.score,
                    "chunk_index": result.payload.get("chunk_index", 0),
                    "file_context": result.payload.get("file_context", False),  # Include file_context metadata for filtering
                })
            
            logger.info(f"✅ Retrieved {len(retrieved_data)} chunks for query '{query}' (user: {user_id}, p_id: {p_id})")
            return retrieved_data
            
        except Exception as e:
            logger.error(f"❌ Failed to retrieve research data: {e}")
            logger.exception("Full traceback:")
            return []
    
    def delete_presentation_data(self, user_id: str, p_id: str):
        """
        Delete all research data for a specific presentation
        
        Args:
            user_id: User ID
            p_id: Presentation ID to delete
        """
        try:
            self.client.delete(
                collection_name=COLLECTION_NAME,
                points_selector={
                    "filter": {
                        "must": [
                            {"key": "user_id", "match": {"value": user_id}},
                            {"key": "p_id", "match": {"value": p_id}}
                        ]
                    }
                }
            )
            logger.info(f"✅ Deleted research data for p_id: {p_id} (user: {user_id})")
        except Exception as e:
            logger.error(f"❌ Failed to delete research data: {e}")
            raise


# Global instance
qdrant_manager = None

def get_qdrant_manager() -> QdrantManager:
    """Get or create Qdrant manager instance"""
    global qdrant_manager
    if qdrant_manager is None:
        qdrant_manager = QdrantManager()
    return qdrant_manager

