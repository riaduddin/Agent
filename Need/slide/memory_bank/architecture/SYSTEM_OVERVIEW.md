# System Overview - Presentation Generation Service

## 🎯 Purpose

An AI-powered service that generates professional slide presentations from user queries and document uploads using multi-agent AI architecture.

## 🏗️ High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Backend                           │
│                     (Python + SSE Streaming)                     │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌──────────────┐    ┌──────────────────┐    ┌──────────────┐
│   MongoDB    │    │  Qdrant Vector   │    │  Google AI   │
│  (Storage)   │    │      DB          │    │  (Gemini)    │
└──────────────┘    └──────────────────┘    └──────────────┘
```

## 🔄 Request Flow

```
1. User Request (SSE)
   ├── JWT Authentication
   ├── Extract query + files
   └── Create session (user_id, p_id)
   
2. Research Phase
   ├── KeywordResearchAgent
   │   └── Generate 8-10 search queries
   ├── BrowserAgent (Parallel)
   │   ├── browser_worker_0..N search web
   │   ├── Extract text + sources
   │   └── Store in Qdrant
   └── Yield sources to SSE
   
3. Planning Phase
   ├── LightweightPlanningAgent
   │   └── Generate slide outline JSON
   └── Parse JSON (4 fallback methods)
   
4. Generation Phase (Parallel)
   ├── EnhancedSlideGenerator_0
   │   ├── Retrieve from Qdrant
   │   ├── Search images (optional)
   │   └── Generate HTML slide
   ├── EnhancedSlideGenerator_1
   └── EnhancedSlideGenerator_N
   
5. Quality Verification (Optional)
   ├── SlideQualityVerifierAgent
   │   ├── Analyze rendering
   │   ├── Fix text overflow
   │   └── Enhance CSS
   └── Return enhanced HTML
   
6. Storage & Response
   ├── Save to MongoDB
   ├── Store agent logs
   └── Stream final HTML via SSE
```

## 🤖 Multi-Agent System

### **Agent Types**

1. **Orchestration Agents** - Coordinate overall workflow
   - `SlideOrchestrationAgent` - Main entry point

2. **Research Agents** - Gather information
   - `KeywordResearchAgent` - Generate search queries
   - `BrowserAgent` - Web search and data collection
   - `comprehensive_research_agent` - Deep research

3. **Planning Agents** - Create presentation structure
   - `LightweightPlanningAgent` - Generate slide outlines
   - `VibeEstimatorAgent` - Infer presentation specs
   - `PresentationSpecExtractorAgent` - Extract requirements

4. **Generation Agents** - Create slide content
   - `EnhancedSlideGenerator` - Generate HTML slides
   - `SlideGeneratorAgent` - Alternative generator

5. **Quality Agents** - Improve output
   - `SlideQualityVerifierAgent` - Analyze and fix slides
   - `ContentRefinerAgent` - Refine content

## 🗄️ Data Storage

### **MongoDB Collections**

```javascript
// presentations
{
  _id: ObjectId,
  p_id: "unique_presentation_id",
  user_id: "user_id_from_jwt",
  title: "Presentation Title",
  slides: [
    {
      slide_number: 1,
      html: "<html>...</html>",
      created_at: DateTime
    }
  ],
  created_at: DateTime,
  updated_at: DateTime
}

// agent_logs
{
  _id: ObjectId,
  p_id: "presentation_id",
  user_id: "user_id",
  agent_name: "agent_name",
  event_type: "chunk|source|done",
  message: "Event message",
  timestamp: DateTime
}
```

### **Qdrant Collection**

```python
# browser_research
{
  "id": "unique_vector_id",
  "vector": [768 dimensions],  # Gemini embedding
  "payload": {
    "text": "Research content chunk",
    "user_id": "user_id",
    "p_id": "presentation_id",
    "keyword": "search_keyword",
    "source_url": "https://...",
    "title": "Source title",
    "timestamp": "ISO datetime"
  }
}
```

## 🔐 Authentication Flow

```
Request with JWT
    ↓
Bearer Token Extraction
    ↓
JWT Validation (JWT_SECRET)
    ↓
Extract User Info (_id, email, role, package, is_verified)
    ↓
Verify User (is_verified == true)
    ↓
Create AuthenticatedUser object
    ↓
Pass to endpoint handler
```

## 🌊 SSE Event Types

| Event Type | Purpose | Example Data |
|------------|---------|--------------|
| `chunk` | Progress updates, agent messages | `{"author": "agent_name", "text": "message"}` |
| `source` | Research sources (title + URL) | `{"title": "...", "url": "..."}` |
| `slide` | Generated slide HTML | `{"slide_number": 1, "html": "..."}` |
| `done` | Completion signal | `{"status": "completed"}` |

## 🔧 Technology Stack

### **Core**
- **Framework**: FastAPI 0.104+
- **Language**: Python 3.9+
- **AI Platform**: Google Vertex AI Agent Builder (ADK)
- **LLM**: Google Gemini 2.5 Flash
- **Authentication**: JWT (PyJWT)

### **Storage**
- **Database**: MongoDB (pymongo)
- **Vector DB**: Qdrant (qdrant-client)
- **Embeddings**: Google `text-embedding-004` (768D)

### **External APIs**
- **Web Search**: Brave Search API, Google Search API
- **Image Search**: Custom image search service
- **Cloud Storage**: Google Cloud Storage (GCS)

### **Async & Streaming**
- **HTTP Client**: httpx, aiohttp
- **SSE**: sse-starlette
- **Async Runtime**: asyncio

## 📊 Performance Characteristics

| Metric | Value |
|--------|-------|
| **Avg. Research Time** | 30-60 seconds (parallel) |
| **Avg. Slide Generation** | 15-30 seconds (parallel) |
| **Total Time (6 slides)** | 1-2 minutes |
| **Qdrant Query Time** | <500ms |
| **MongoDB Write Time** | <100ms |
| **Max Concurrent Agents** | 10+ (configurable) |

## 🔄 Design Patterns

### **1. Parallel Execution**
Multiple agents run simultaneously for speed:
- Browser workers search in parallel
- Slide generators work concurrently

### **2. Semantic Search**
Uses vector embeddings for intelligent retrieval:
- Research stored with embeddings
- Slide generators query by meaning, not keywords

### **3. Progressive Enhancement**
Quality verification improves output:
- Initial generation
- Automated analysis
- CSS/HTML enhancement

### **4. Graceful Degradation**
Multiple fallback mechanisms:
- 4 JSON parsing methods
- Keyword fallbacks in search
- Default theme if detection fails

## 🎨 Presentation Themes

Supports dynamic theme detection:
- **Modern Professional** - Blue/gray palette
- **Creative Vibrant** - Colorful, energetic
- **Minimal Clean** - White space, simple
- **Corporate Formal** - Dark, serious
- **Brand-specific** - Auto-detected from queries

## 🚀 Scalability Considerations

### **Current Architecture**
- Single-instance deployment
- In-memory session management
- Local Qdrant instance

### **Production Scaling**
1. **Horizontal Scaling**
   - Multiple FastAPI instances
   - Load balancer (nginx/ALB)
   - Distributed session store (Redis)

2. **Database Scaling**
   - MongoDB replica sets
   - Qdrant cluster
   - Read replicas

3. **Caching**
   - Redis for browser results
   - CDN for generated presentations
   - API response caching

4. **Rate Limiting**
   - Per-user API limits
   - Queue-based processing
   - Background task workers

## 🔍 Monitoring & Observability

### **Logging**
- Structured logging (Python logging)
- Agent execution traces
- Error stack traces

### **Metrics** (Potential)
- Request latency
- Agent execution time
- Qdrant query performance
- Error rates

### **Tracing** (Potential)
- OpenTelemetry instrumentation
- Distributed tracing
- Request flow visualization

## 📝 Key Files

| File | Purpose |
|------|---------|
| `main.py` | Main FastAPI application |
| `app_sse.py` | SSE endpoints |
| `root_agent/agent.py` | SlideOrchestrationAgent |
| `qdrant_utils.py` | Vector DB management |
| `auth_middleware.py` | JWT authentication |
| `db.py` | MongoDB operations |

## 🎓 Learning Resources

- **Google ADK Docs**: https://cloud.google.com/vertex-ai/docs
- **Qdrant Docs**: https://qdrant.tech/documentation/
- **FastAPI Docs**: https://fastapi.tiangolo.com/
- **Gemini API**: https://ai.google.dev/docs

---

**Related Documents**:
- `AGENT_FLOW.md` - Detailed agent workflows
- `DATA_FLOW.md` - Data movement and transformations
- `DEPLOYMENT.md` - Production deployment guide

