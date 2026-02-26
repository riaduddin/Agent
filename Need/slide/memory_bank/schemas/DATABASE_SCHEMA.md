# Database Schemas

Complete database schema documentation for MongoDB and Qdrant.

---

## 🗄️ MongoDB Collections

### **presentations**

Stores presentation metadata and generated slides.

**Schema**:
```javascript
{
  _id: ObjectId("..."),                    // MongoDB auto-generated ID
  p_id: "abc123...",                       // Unique presentation ID (string)
  user_id: "user_id_from_jwt",             // User who created it
  title: "Presentation Title",             // Inferred or user-provided
  message: "Original user query...",       // User's original request
  specs: {                                 // Presentation specifications
    presentation_type: "Business Pitch",
    audience_type: "Investors",
    tone: "Professional",
    slide_count: 6,
    color_theme: {
      background_color: "#FFFFFF",
      text_color: "#1F2937",
      heading_color: "#3B82F6",
      accent_color: "#F6823B",
      secondary_color: "#2F68C5"
    }
  },
  slides: [                                // Array of slides
    {
      slide_number: 1,
      html: "<!DOCTYPE html>...",          // Complete HTML with inline CSS
      created_at: ISODate("2025-10-21T12:00:00Z")
    },
    {
      slide_number: 2,
      html: "<!DOCTYPE html>...",
      created_at: ISODate("2025-10-21T12:00:01Z")
    }
  ],
  file_urls: [                             // Optional uploaded files
    "https://storage.googleapis.com/..."
  ],
  status: "completed",                     // pending | processing | completed | failed
  created_at: ISODate("2025-10-21T12:00:00Z"),
  updated_at: ISODate("2025-10-21T12:05:00Z")
}
```

**Indexes**:
```javascript
// Recommended indexes
db.presentations.createIndex({ "p_id": 1 }, { unique: true })
db.presentations.createIndex({ "user_id": 1, "created_at": -1 })
db.presentations.createIndex({ "status": 1 })
```

**Size**: ~50KB - 500KB per presentation (depends on slide count and HTML complexity)

---

### **agent_logs**

Stores detailed agent execution logs for debugging and monitoring.

**Schema**:
```javascript
{
  _id: ObjectId("..."),                    // MongoDB auto-generated ID
  p_id: "abc123...",                       // Presentation ID (links to presentations collection)
  user_id: "user_id_from_jwt",             // User who triggered the agent
  agent_name: "KeywordResearchAgent",      // Name of the agent
  event_type: "chunk",                     // chunk | source | done | error
  message: "Generated 10 search queries",  // Human-readable message
  data: {                                  // Optional structured data
    search_queries: ["query1", "query2"],
    execution_time_ms: 1500
  },
  level: "info",                           // debug | info | warning | error
  timestamp: ISODate("2025-10-21T12:00:00Z")
}
```

**Indexes**:
```javascript
// Recommended indexes
db.agent_logs.createIndex({ "p_id": 1, "timestamp": 1 })
db.agent_logs.createIndex({ "user_id": 1 })
db.agent_logs.createIndex({ "agent_name": 1 })
db.agent_logs.createIndex({ "level": 1, "timestamp": -1 })
```

**Size**: ~1KB - 5KB per log entry  
**Retention**: Consider TTL index for automatic cleanup:
```javascript
// Auto-delete logs older than 30 days
db.agent_logs.createIndex(
  { "timestamp": 1 },
  { expireAfterSeconds: 2592000 }  // 30 days
)
```

---

## 🔍 Qdrant Collections

### **browser_research**

Stores research text chunks with vector embeddings for semantic search.

**Collection Config**:
```python
{
  "vectors": {
    "size": 768,                           # Gemini text-embedding-004 dimensions
    "distance": "Cosine"                   # Similarity metric
  }
}
```

**Point Schema**:
```python
{
  "id": "uuid-or-hash",                    # Unique ID for this vector
  "vector": [0.123, -0.456, ...],          # 768-dimensional embedding
  "payload": {                             # Metadata
    "text": "Research content chunk...",   # Original text (800 chars)
    "user_id": "user_id",                  # For filtering
    "p_id": "presentation_id",             # For filtering
    "keyword": "AI healthcare",            # Search keyword used
    "source_url": "https://...",           # Source URL (optional)
    "title": "Article Title",              # Source title (optional)
    "timestamp": "2025-10-21T12:00:00Z",   # When stored
    "chunk_index": 0                       # Chunk number in original text
  }
}
```

**Payload Fields**:

| Field | Type | Required | Purpose |
|-------|------|----------|---------|
| `text` | string | Yes | The actual text content (chunked) |
| `user_id` | string | Yes | Filter by user |
| `p_id` | string | Yes | Filter by presentation |
| `keyword` | string | No | Original search query |
| `source_url` | string | No | Where this came from |
| `title` | string | No | Source title |
| `timestamp` | string | No | When added |
| `chunk_index` | integer | No | Chunk position |

**Indexing**:
- Qdrant automatically indexes vectors (HNSW)
- Payload fields automatically indexed for filtering

**Size**: 
- Vector: ~3KB (768 floats * 4 bytes)
- Payload: ~1KB - 2KB
- **Total**: ~4KB - 5KB per point

**Cleanup**:
```python
# Delete all data for a presentation
from qdrant_utils import QdrantManager

manager = QdrantManager()
manager.delete_presentation_data(user_id="...", p_id="...")
```

---

## 📊 Data Flow

### **Presentation Creation**

```
1. User Request
   ↓
2. Create MongoDB Document
   {
     p_id: "abc123",
     user_id: "user123",
     status: "processing",
     slides: []
   }
   ↓
3. Research Phase
   ├─ Web search results → Qdrant
   ├─ Store text chunks with embeddings
   └─ Associate with user_id + p_id
   ↓
4. Generation Phase
   ├─ Retrieve from Qdrant (filtered by user_id + p_id)
   ├─ Generate slides
   └─ Update MongoDB
       {
         slides: [{slide_number: 1, html: "..."}],
         status: "completed"
       }
   ↓
5. Agent Logs
   └─ Store in MongoDB agent_logs collection
```

### **Presentation Retrieval**

```
1. GET /slides/?p_id=abc123
   ↓
2. Query MongoDB
   db.presentations.findOne({
     p_id: "abc123",
     user_id: "user123"  // From JWT
   })
   ↓
3. Return slides array
```

### **Presentation Deletion**

```
1. DELETE /presentations/abc123
   ↓
2. Delete from MongoDB
   db.presentations.deleteOne({
     p_id: "abc123",
     user_id: "user123"
   })
   ↓
3. Delete from Qdrant
   qdrant.delete({
     filter: {
       "user_id": "user123",
       "p_id": "abc123"
     }
   })
   # Note: Currently uses Python-based filtering workaround
   ↓
4. Delete agent logs
   db.agent_logs.deleteMany({
     p_id: "abc123"
   })
```

---

## 🔄 Data Lifecycle

### **Typical Presentation Lifecycle**

1. **Creation** (t=0):
   - MongoDB: Status = "processing"
   - Qdrant: No vectors yet

2. **Research Phase** (t=30s):
   - Qdrant: Add 50-200 vectors (research chunks)
   - MongoDB: agent_logs grows

3. **Generation Phase** (t=60s):
   - MongoDB: slides array populates
   - MongoDB: agent_logs continues

4. **Completion** (t=90s):
   - MongoDB: Status = "completed"
   - MongoDB: slides = [6 slides]
   - Qdrant: ~100-300 vectors stored

5. **Retrieval** (t+hours/days):
   - MongoDB: Read slides
   - Qdrant: Vectors remain (for regeneration or similar presentations)

6. **Deletion** (optional):
   - MongoDB: Delete document + logs
   - Qdrant: Delete vectors

---

## 💾 Storage Estimates

### **Per Presentation**

| Component | Size | Notes |
|-----------|------|-------|
| MongoDB (presentation) | 50KB - 500KB | Depends on slide count/complexity |
| MongoDB (agent_logs) | 10KB - 100KB | ~50-200 log entries |
| Qdrant (vectors) | 400KB - 1.5MB | ~100-300 vectors |
| **Total** | **460KB - 2.1MB** | Per presentation |

### **1000 Presentations**

| Component | Size |
|-----------|------|
| MongoDB | 50MB - 600MB |
| Qdrant | 400MB - 1.5GB |
| **Total** | **450MB - 2.1GB** |

---

## 🔍 Query Patterns

### **Common MongoDB Queries**

```javascript
// Get user's presentations
db.presentations.find({
  user_id: "user123"
}).sort({ created_at: -1 })

// Get presentation by p_id
db.presentations.findOne({
  p_id: "abc123",
  user_id: "user123"
})

// Get recent presentations
db.presentations.find({
  created_at: { $gte: ISODate("2025-10-20T00:00:00Z") }
})

// Get failed presentations
db.presentations.find({
  status: "failed"
})

// Count slides
db.presentations.aggregate([
  { $project: { slide_count: { $size: "$slides" } } }
])
```

### **Common Qdrant Queries**

```python
# Semantic search (via qdrant_utils.py)
from qdrant_utils import QdrantManager

manager = QdrantManager()
results = manager.retrieve_research_data(
    query="AI in healthcare applications",
    user_id="user123",
    p_id="abc123",
    top_k=10
)

# Direct Qdrant API (without filters)
from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)
results = client.query_points(
    collection_name="browser_research",
    query=[0.123, -0.456, ...],  # Query vector
    limit=10
)
```

---

## 🛠️ Maintenance

### **MongoDB Maintenance**

```javascript
// Check collection sizes
db.stats()
db.presentations.stats()
db.agent_logs.stats()

// Rebuild indexes
db.presentations.reIndex()
db.agent_logs.reIndex()

// Clean up old logs (if no TTL index)
db.agent_logs.deleteMany({
  timestamp: { $lt: ISODate("2025-09-21T00:00:00Z") }
})
```

### **Qdrant Maintenance**

```bash
# Check collection info
curl http://localhost:6333/collections/browser_research

# Backup (Docker)
docker exec qdrant_container tar czf /qdrant/backup.tar.gz /qdrant/storage

# Optimize collection (rebuild index)
curl -X POST http://localhost:6333/collections/browser_research/index
```

---

## 📈 Scaling Considerations

### **MongoDB**

- **Replica Sets**: For high availability
- **Sharding**: For >100GB data
- **Indexes**: Ensure p_id, user_id, created_at indexed
- **Archiving**: Move old presentations to cold storage

### **Qdrant**

- **Cluster Mode**: For >10M vectors
- **Quantization**: Reduce memory usage (u8 quantization)
- **Replication**: For high availability
- **Partitioning**: Separate collections per tenant

---

**Related Documents**:
- `SYSTEM_OVERVIEW.md` - Architecture overview
- `API_REFERENCE.md` - API endpoints that interact with these schemas
- `CURRENT_STATE.md` - Current database status

