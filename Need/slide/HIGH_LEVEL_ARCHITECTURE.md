# High-Level Architecture - Presentation Generation Service

## Executive Summary

AI-powered presentation generation service built on Google Vertex AI, featuring a multi-agent orchestration system, real-time WebSocket streaming, and vector-based semantic search. The system processes user queries to generate professional slide presentations with research, planning, and generation phases executed in parallel for optimal performance.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CLIENT LAYER                                      │
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                │
│  │   Web App    │    │  Mobile App  │    │   Postman    │                │
│  └──────────────┘    └──────────────┘    └──────────────┘                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ HTTPS / WebSocket
                                    │ JWT Authentication
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      API GATEWAY LAYER (FastAPI)                            │
│                                                                              │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐        │
│  │  REST Endpoints  │  │  WebSocket (SIO) │  │  SSE Streaming   │        │
│  │  - /upload       │  │  /ws/{p_id}      │  │  /sse/present.   │        │
│  │  - /slides       │  │  Real-time events│  │  /stream/{p_id}  │        │
│  │  - /templates    │  │                  │  │                  │        │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘        │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────┐          │
│  │           Authentication Middleware (JWT)                    │          │
│  │           - Token validation                                 │          │
│  │           - User verification                                │          │
│  │           - Multi-tenancy support                            │          │
│  └────────────────────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ORCHESTRATION LAYER                                       │
│                                                                              │
│                    ┌──────────────────────────┐                            │
│                    │ SlideOrchestrationAgent   │                            │
│                    │  (Root Agent)             │                            │
│                    │                          │                            │
│                    │  • Topic Detection        │                            │
│                    │  • Query Enhancement      │                            │
│                    │  • Intent Classification │                            │
│                    │  • Task Routing          │                            │
│                    └──────────────────────────┘                            │
│                               │                                             │
│                ┌──────────────┼──────────────┐                              │
│                │              │              │                              │
│                ▼              ▼              ▼                              │
│    ┌──────────────┐ ┌──────────────┐ ┌──────────────┐                    │
│    │   Creation    │ │ Modification │ │  Insertion    │                    │
│    │   Pipeline    │ │ Orchestrator │ │ Orchestrator  │                    │
│    └──────────────┘ └──────────────┘ └──────────────┘                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PROCESSING AGENTS LAYER                                   │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                    RESEARCH PHASE (Parallel)                        │    │
│  │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐   │    │
│  │  │ KeywordResearch  │  │   BrowserAgent   │  │ File Extractor│   │    │
│  │  │                  │  │  (Multi Workers) │  │              │   │    │
│  │  │ • Generate       │  │  • Web Search     │  │ • PDF/DOCX   │   │    │
│  │  │   8-10 queries   │  │  • Content Extract│ │ • Text Extract│  │    │
│  │  └──────────────────┘  └──────────────────┘  └──────────────┘   │    │
│  │                          │                                          │    │
│  │                          ▼                                          │    │
│  │              ┌──────────────────────┐                              │    │
│  │              │   Qdrant Vector DB   │                              │    │
│  │              │  (Semantic Storage)  │                              │    │
│  │              │  • Store embeddings  │                              │    │
│  │              │  • Vector search     │                              │    │
│  │              └──────────────────────┘                              │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                    PLANNING PHASE                                  │    │
│  │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐      │    │
│  │  │ Spec Extractor  │  │  Vibe Estimator │  │   Planning   │      │    │
│  │  │                 │  │                 │  │   Agent      │      │    │
│  │  │ • Presentation  │  │  • Theme/Colors │  │              │      │    │
│  │  │   type/audience │  │  • Tone/Style   │  │ • Slide      │      │    │
│  │  │                 │  │                 │  │   outline     │      │    │
│  │  └──────────────────┘  └──────────────────┘  └──────────────┘      │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                  GENERATION PHASE (Parallel)                        │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐            │    │
│  │  │ Slide Gen 0  │  │ Slide Gen 1 │  │ Slide Gen N  │            │    │
│  │  │              │  │              │  │              │            │    │
│  │  │ • Retrieve   │  │ • Retrieve   │  │ • Retrieve   │            │    │
│  │  │   from Qdrant│  │   from Qdrant│  │   from Qdrant│            │    │
│  │  │ • Generate   │  │ • Generate   │  │ • Generate   │            │    │
│  │  │   HTML slide │  │   HTML slide │  │   HTML slide │            │    │
│  │  └──────────────┘  └──────────────┘  └──────────────┘            │    │
│  │                          │                                          │    │
│  │                          ▼                                          │    │
│  │              ┌──────────────────────┐                              │    │
│  │              │ Quality Verifier     │                              │    │
│  │              │ • CSS fixes          │                              │    │
│  │              │ • Text overflow      │                              │    │
│  │              │ • Rendering checks   │                              │    │
│  │              └──────────────────────┘                              │    │
│  └────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DATA STORAGE LAYER                                  │
│                                                                              │
│  ┌──────────────────┐              ┌──────────────────┐                    │
│  │     MongoDB      │              │      Redis       │                    │
│  │                  │              │                  │                    │
│  │ • presentations  │              │ • Pub/Sub        │                    │
│  │ • slides         │              │ • Distributed    │                    │
│  │ • agent_logs     │              │   locks          │                    │
│  │ • agent_outputs  │              │ • Connection       │                    │
│  │ • users          │              │   tracking       │                    │
│  │ • templates      │              │ • Session state  │                    │
│  └──────────────────┘              └──────────────────┘                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      EXTERNAL SERVICES LAYER                                 │
│                                                                              │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐          │
│  │  Google Gemini   │  │  Brave/Google   │  │ Google Cloud     │          │
│  │  AI Platform     │  │  Search APIs    │  │ Storage (GCS)    │          │
│  │                  │  │                  │  │                  │          │
│  │ • LLM (Gemini    │  │ • Web search     │  │ • File uploads   │          │
│  │   2.5 Flash)     │  │ • Content        │  │ • Static assets  │          │
│  │ • Embeddings     │  │   scraping       │  │ • Signed URLs    │          │
│  │   (text-embed-  │  │                  │  │                  │          │
│  │    004)         │  │                  │  │                  │          │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘          │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Key Components

### 1. **Client Layer**
- **Web Applications**: Browser-based clients
- **Mobile Applications**: Native mobile apps
- **API Clients**: Postman, curl, etc.
- **Protocols**: HTTPS for REST, WebSocket/Socket.IO for real-time

### 2. **API Gateway Layer**
- **Framework**: FastAPI (Python)
- **Authentication**: JWT-based middleware
- **Communication Methods**:
  - **REST API**: Standard CRUD operations
  - **WebSocket (Socket.IO)**: Real-time bidirectional communication
  - **SSE**: Server-Sent Events for streaming

### 3. **Orchestration Layer**
- **Root Agent**: `SlideOrchestrationAgent`
  - Routes requests to appropriate pipeline
  - Manages session state
  - Handles multi-turn conversations
- **Sub-Orchestrators**:
  - **Slide Creation Pipeline**: New presentation generation
  - **Modification Orchestrator**: Edit existing slides
  - **Insertion Orchestrator**: Add new slides

### 4. **Processing Agents Layer**

#### Research Phase
- **KeywordResearchAgent**: Generates 8-10 search queries
- **BrowserAgent**: Parallel web workers for content extraction
- **File Extractor**: Extracts text from PDF/DOCX/TXT uploads
- **Qdrant Vector DB**: Stores and retrieves semantic embeddings (768D)

#### Planning Phase
- **PresentationSpecExtractorAgent**: Extracts requirements
- **VibeEstimatorAgent**: Determines theme, tone, visual style
- **PlanningAgent**: Generates structured slide outline (JSON)

#### Generation Phase
- **EnhancedSlideGenerator**: Parallel generators (0..N workers)
  - Retrieves context from Qdrant
  - Generates HTML slides with CSS
  - Uses dynamic layouts based on content
- **SlideQualityVerifierAgent**: Validates and fixes output

### 5. **Data Storage Layer**
- **MongoDB**: Primary database
  - `presentations`: Presentation metadata
  - `slides`: Slide content and plans
  - `agent_outputs_2`: Real-time event stream
  - `users`: User accounts and verification
  - `templates`: Reusable slide templates
- **Redis**: Distributed coordination
  - Pub/Sub for cross-worker messaging
  - Distributed locks for agent execution
  - Connection tracking for WebSocket

### 6. **External Services Layer**
- **Google Gemini AI**: LLM and embeddings
- **Search APIs**: Brave Search, Google Search
- **Google Cloud Storage**: File uploads and asset storage

---

## 🔄 Request Flow

### **Create Presentation Flow**

```
1. Client Request
   ├─ POST /ws/{p_id}?token={jwt}
   ├─ Query: "Create presentation about AI"
   └─ Files: [optional PDF/DOCX URLs]
   │
2. Authentication & Validation
   ├─ Verify JWT token
   ├─ Extract user_id
   └─ Create/validate presentation record
   │
3. Root Agent (SlideOrchestrationAgent)
   ├─ Topic Checker: Verify topic exists
   ├─ Query Enhancer: Enhance with context
   ├─ Query Classifier: Determine intent (create_presentation)
   └─ Route to: SlideCreationPipeline
   │
4. Research Phase (Parallel)
   ├─ KeywordResearchAgent → 8-10 queries
   ├─ BrowserAgent (Workers 0..N) → Web search
   ├─ File Extractor → Extract uploaded files
   └─ Store in Qdrant (with embeddings)
   │
5. Planning Phase
   ├─ SpecExtractor → Requirements
   ├─ VibeEstimator → Theme/colors/tone
   └─ PlanningAgent → Slide outline (JSON)
   │
6. Generation Phase (Parallel)
   ├─ For each slide:
   │  ├─ EnhancedSlideGenerator_N
   │  │  ├─ Query Qdrant (semantic search)
   │  │  ├─ Generate HTML + CSS
   │  │  └─ Apply dynamic layout
   │  └─ QualityVerifier → Fix issues
   │
7. Storage & Streaming
   ├─ Save to MongoDB
   ├─ Emit via Socket.IO (real-time)
   └─ Stream to all connected clients
```

### **Modify Slide Flow**

```
1. Client Request
   ├─ Query: "Change slide 3 to focus on revenue"
   ├─ p_id: Presentation ID
   └─ Authentication
   │
2. Root Agent
   ├─ Route to: MultiSlideModificationOrchestrator
   │
3. Modification Process
   ├─ RequestParser → Extract slide_number & edit_request
   ├─ DataFetcher → Get original slide data
   ├─ PlanModifier → Modify slide plan
   ├─ SlideGenerator → Regenerate HTML
   └─ DatabaseUpdater → Save changes
   │
4. Real-time Update
   └─ Emit updated slide via Socket.IO
```

---

## 🔐 Security Architecture

```
Request → JWT Token Extraction
         │
         ├─ Verify Signature (JWT_SECRET)
         │
         ├─ Extract Claims
         │  ├─ user_id
         │  ├─ email
         │  ├─ role
         │  └─ is_verified
         │
         ├─ Validate User (is_verified == true)
         │
         └─ Authorization Check
            ├─ Verify user owns presentation (p_id)
            └─ Check user permissions
```

---

## ⚡ Performance Characteristics

| Phase | Time | Parallelization |
|-------|------|-----------------|
| **Research** | 30-60s | 8-10 parallel browser workers |
| **Planning** | 5-10s | Sequential (spec → vibe → plan) |
| **Generation** | 15-30s per slide | N parallel slide generators |
| **Total (6 slides)** | ~1-2 minutes | Fully optimized pipeline |

**Scalability**:
- Single worker: 1,000+ concurrent connections
- 4 workers: 4,000+ concurrent connections
- Production (12 workers): 12,000+ concurrent connections

---

## 🚀 Deployment Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Load Balancer (NGINX)                     │
└─────────────────────────────────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          │               │               │
          ▼               ▼               ▼
    ┌──────────┐   ┌──────────┐   ┌──────────┐
    │ Worker 1 │   │ Worker 2 │   │ Worker N │
    │ FastAPI  │   │ FastAPI  │   │ FastAPI  │
    │ Gunicorn │   │ Gunicorn │   │ Gunicorn │
    └──────────┘   └──────────┘   └──────────┘
          │               │               │
          └───────────────┼───────────────┘
                          │
          ┌───────────────┼───────────────┐
          │               │               │
          ▼               ▼               ▼
    ┌──────────┐   ┌──────────┐   ┌──────────┐
    │  MongoDB │   │  Redis   │   │  Qdrant  │
    │ (Primary)│   │ (Pub/Sub)│   │ (Vector) │
    └──────────┘   └──────────┘   └──────────┘
```

---

## 📦 Technology Stack Summary

| Category | Technology | Purpose |
|----------|-----------|---------|
| **Framework** | FastAPI | HTTP/WebSocket server |
| **AI Platform** | Google Vertex AI Agent Builder | Agent orchestration |
| **LLM** | Gemini 2.5 Flash | Content generation |
| **Embeddings** | text-embedding-004 (768D) | Vector embeddings |
| **Database** | MongoDB | Primary data store |
| **Message Broker** | Redis | Pub/Sub, locks, state |
| **Vector DB** | Qdrant | Semantic search |
| **Cloud Storage** | Google Cloud Storage | File storage |
| **Search APIs** | Brave Search, Google Search | Web research |
| **Auth** | JWT (PyJWT) | Authentication |

---

## 🎯 Key Design Decisions

1. **Multi-Agent Architecture**: Specialized agents for each task enable parallelization and maintainability
2. **Vector-Based Search**: Qdrant provides semantic retrieval superior to keyword matching
3. **Real-Time Streaming**: WebSocket enables live updates during generation
4. **Distributed Coordination**: Redis Pub/Sub allows horizontal scaling
5. **Parallel Processing**: Research and generation phases run concurrently
6. **Quality Verification**: Automated checks ensure output quality

---

## 📈 Scalability Considerations

- **Horizontal Scaling**: Multiple FastAPI workers via Gunicorn
- **Stateless Workers**: Session state in Redis/MongoDB
- **Distributed Locks**: Prevent duplicate agent execution
- **Connection Pooling**: MongoDB and Redis connection pools
- **CDN Integration**: Static assets via Google Cloud Storage

---

## 🔍 Monitoring Points

- Active WebSocket connections per worker
- Agent execution time (avg/p95/p99)
- Redis Pub/Sub message rate
- MongoDB query latency
- Lock acquisition time
- Failed agent runs
- Connection errors/disconnects

---

**Document Version**: 1.0  
**Last Updated**: 2025  
**Maintained By**: Architecture Team


