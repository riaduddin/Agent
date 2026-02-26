# High-Level Architecture Diagram

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT LAYER                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │  Web Client  │    │ Mobile App   │    │  Postman     │                  │
│  │  (Browser)   │    │  (Native)    │    │  (Testing)   │                  │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘                  │
│         │                   │                     │                          │
│         │ WebSocket (ws://) │ REST API (http://)  │                          │
│         └───────────────────┴─────────────────────┘                          │
│                              │                                                │
└──────────────────────────────┼────────────────────────────────────────────────┘
                               │
                               │ JWT Authentication
                               │
┌──────────────────────────────▼────────────────────────────────────────────────┐
│                         LOAD BALANCER / NGINX                                 │
│                    (Optional - for production)                                │
└──────────────────────────────┬────────────────────────────────────────────────┘
                               │
                               │ Round Robin
                               │
┌──────────────────────────────▼────────────────────────────────────────────────┐
│                         FASTAPI APPLICATION                                   │
│                      (main.py - Multiple Workers)                             │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │
│  │  Worker 1       │  │  Worker 2       │  │  Worker N       │              │
│  │  (Gunicorn)     │  │  (Gunicorn)     │  │  (Gunicorn)     │              │
│  │                 │  │                 │  │                 │              │
│  │ ┌─────────────┐ │  │ ┌─────────────┐ │  │ ┌─────────────┐ │              │
│  │ │ WebSocket   │ │  │ │ WebSocket   │ │  │ │ WebSocket   │ │              │
│  │ │ Manager     │ │  │ │ Manager     │ │  │ │ Manager     │ │              │
│  │ └─────┬───────┘ │  │ └─────┬───────┘ │  │ └─────┬───────┘ │              │
│  │       │         │  │       │         │  │       │         │              │
│  │ ┌─────▼───────┐ │  │ ┌─────▼───────┐ │  │ ┌─────▼───────┐ │              │
│  │ │ REST API    │ │  │ │ REST API    │ │  │ │ REST API    │ │              │
│  │ │ Endpoints   │ │  │ │ Endpoints   │ │  │ │ Endpoints   │ │              │
│  │ └─────────────┘ │  │ └─────────────┘ │  │ └─────────────┘ │              │
│  └────┬────────┬───┘  └────┬────────┬───┘  └────┬────────┬───┘              │
│       │        │            │        │            │        │                  │
└───────┼────────┼────────────┼────────┼────────────┼────────┼──────────────────┘
        │        │            │        │            │        │
        │        └────────────┴────────┴────────────┘        │
        │                     │                              │
        │              ┌──────▼──────┐                       │
        │              │   REDIS     │                       │
        │              │  (In-Memory) │                      │
        │              ├─────────────┤                       │
        │              │ Pub/Sub     │◄──────────────────────┘
        │              │ Channels    │   Cross-worker messaging
        │              ├─────────────┤
        │              │ Distributed │   Agent lock
        │              │ Locking     │   Connection registry
        │              ├─────────────┤
        │              │ Session     │   Last ack tracking
        │              │ State       │   Heartbeat data
        │              └─────────────┘
        │
        │
        │
┌───────▼───────────────────────────────────────────────────────────────────────┐
│                              MONGODB                                          │
│                         (Persistent Storage)                                  │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐                 │
│  │ presentations  │  │ agent_outputs_2│  │    users       │                 │
│  ├────────────────┤  ├────────────────┤  ├────────────────┤                 │
│  │ - p_id         │  │ - _id (event)  │  │ - user_id      │                 │
│  │ - user_id      │  │ - p_id         │  │ - email        │                 │
│  │ - status       │  │ - author       │  │ - package      │                 │
│  │ - title        │  │ - content      │  │ - is_verified  │                 │
│  │ - slides[]     │  │ - timestamp    │  │ - role         │                 │
│  │ - timeline[]   │  │ - html_content │  └────────────────┘                 │
│  │ - created_at   │  └────────────────┘                                       │
│  └────────────────┘                                                           │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
                               │
                               │
┌──────────────────────────────▼────────────────────────────────────────────────┐
│                         AGENT EXECUTION LAYER                                 │
│                    (Background Processing - Async)                            │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  ┌─────────────────────────────────────────────────────────────┐              │
│  │           Root Agent (Orchestrator)                         │              │
│  │         (root_agent/slide_creation_agent/)                  │              │
│  └───────────────────────┬─────────────────────────────────────┘              │
│                          │                                                    │
│          ┌───────────────┼───────────────┐                                    │
│          │               │               │                                    │
│  ┌───────▼──────┐ ┌──────▼──────┐ ┌─────▼──────────┐                        │
│  │Presentation  │ │  Keyword    │ │  Slide         │                        │
│  │Spec Extract  │ │  Research   │ │  Generator     │                        │
│  │Agent         │ │  Agent      │ │  Agent         │                        │
│  └──────┬───────┘ └──────┬──────┘ └─────┬──────────┘                        │
│         │                │               │                                    │
│         │    ┌───────────▼───────────────▼───────────┐                        │
│         │    │      Template Retrieval Tool          │                        │
│         │    │      (Qdrant Vector Search)           │                        │
│         │    └───────────────────────────────────────┘                        │
│         │                                                                     │
│         │    ┌───────────────────────────────────────┐                        │
│         └────►   Image/Logo Search Tools             │                        │
│              │   (External APIs)                     │                        │
│              └───────────────────────────────────────┘                        │
│                                                                                │
│  Each agent writes output to:                                                 │
│  1. MongoDB (agent_outputs_2 collection)                                      │
│  2. Redis Pub/Sub (presentation:{p_id} channel)                               │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
```

## Data Flow Diagram

### 1. Fresh WebSocket Connection (Queued Presentation)

```
┌──────────┐
│  Client  │
└────┬─────┘
     │ 1. Connect WebSocket
     │    ws://host/ws/{p_id}?token={jwt}
     ▼
┌────────────────┐
│  Worker 1      │
│  (WebSocket    │
│   Manager)     │
└────┬───────────┘
     │ 2. Verify JWT Token
     │
     ▼
┌────────────────┐
│  MongoDB       │  3. Check presentation status
│  presentations │     → "queued"
└────┬───────────┘
     │
     ▼
┌────────────────┐
│  Client        │  4. Send status message
│                │     {"type": "status", "status": "queued",
│                │      "message": "fresh connection - only new data"}
└────────────────┘
     │
     ▼
┌────────────────┐
│  Client        │  5. Send backfill_complete
│                │     {"type": "backfill_complete", "events_count": 0}
└────────────────┘
     │
     ▼
┌────────────────┐
│  Redis         │  6. Try acquire lock
│  agent:lock:   │     SET NX agent:lock:{p_id} = worker_id
│  {p_id}        │     TTL: 600 seconds
└────┬───────────┘
     │ Lock acquired ✓
     ▼
┌────────────────┐
│  MongoDB       │  7. Update status to "processing"
│  presentations │
└────────────────┘
     │
     ▼
┌────────────────┐
│  Agent Layer   │  8. Start async agent execution
│  (Background)  │     - Presentation Spec Extractor
│                │     - Keyword Research
│                │     - Slide Generators
└────┬───────────┘
     │
     │ 9. For each agent output:
     │
     ▼
┌────────────────┐
│  MongoDB       │  a. Save to agent_outputs_2
│  agent_outputs │     {p_id, author, content, timestamp}
└────────────────┘
     │
     ▼
┌────────────────┐
│  Redis Pub/Sub │  b. Publish to channel
│  presentation: │     PUBLISH presentation:{p_id} {message}
│  {p_id}        │
└────┬───────────┘
     │
     ├──────────┬──────────┬──────────┐
     ▼          ▼          ▼          ▼
┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
│Worker 1│ │Worker 2│ │Worker 3│ │Worker N│  c. All workers receive message
│(Active)│ │        │ │        │ │        │
└───┬────┘ └────────┘ └────────┘ └────────┘
    │
    ▼
┌────────────────┐
│  Client        │  d. Send to connected client
│  (WebSocket)   │     {"type": "chunk", "author": "...",
│                │      "content": "...", "html_content": "..."}
└────────────────┘
     │
     │ 10. Agent completes
     ▼
┌────────────────┐
│  MongoDB       │  11. Update status to "completed"
│  presentations │      Add all slides, set completion_date
└────────────────┘
     │
     ▼
┌────────────────┐
│  Client        │  12. Send terminal message
│  (WebSocket)   │      {"type": "terminal", "status": "completed"}
└────────────────┘
     │
     ▼
┌────────────────┐
│  Redis         │  13. Release lock
│  agent:lock:   │      DEL agent:lock:{p_id}
│  {p_id}        │
└────────────────┘
```

### 2. Reconnection Flow

```
┌──────────┐
│  Client  │
└────┬─────┘
     │ 1. Reconnect with last_event_id
     │    ws://host/ws/{p_id}?token={jwt}&last_event_id={event_id}
     ▼
┌────────────────┐
│  Worker 2      │  2. Verify JWT, check status
│  (WebSocket    │     → "processing"
│   Manager)     │
└────┬───────────┘
     │
     ▼
┌────────────────┐
│  Client        │  3. Send status
│                │     {"type": "status", "status": "processing",
│                │      "is_reconnection": true,
│                │      "message": "reconnecting - will backfill missed events"}
└────────────────┘
     │
     ▼
┌────────────────┐
│  MongoDB       │  4. Query agent_outputs_2
│  agent_outputs │     WHERE p_id = {p_id} AND _id > {last_event_id}
│                │     → Get missed events
└────┬───────────┘
     │
     ▼
┌────────────────┐
│  Client        │  5. Send backfilled events
│  (WebSocket)   │     {"type": "chunk", ...} x N events
│                │     {"type": "backfill_complete", "events_count": N}
└────────────────┘
     │
     ▼
┌────────────────┐
│  Client        │  6. Continue streaming new events
│  (WebSocket)   │     (via Redis Pub/Sub)
└────────────────┘
```

### 3. REST API Flow (Completed Presentation)

```
┌──────────┐
│  Client  │
└────┬─────┘
     │ 1. GET /presentation/{p_id}/data?token={jwt}
     ▼
┌────────────────┐
│  Worker N      │  2. Verify JWT Token
│  (REST API)    │
└────┬───────────┘
     │
     ▼
┌────────────────┐
│  MongoDB       │  3. Query presentation
│  presentations │     WHERE p_id = {p_id}
│                │     → status = "completed"
└────┬───────────┘
     │
     ▼
┌────────────────┐
│  MongoDB       │  4. Query all events
│  agent_outputs │     WHERE p_id = {p_id}
│                │     → Get complete timeline
└────┬───────────┘
     │
     ▼
┌────────────────┐
│  Client        │  5. Return complete data
│  (HTTP 200)    │     {
│                │       "p_id": "...",
│                │       "status": "completed",
│                │       "slides": [...],
│                │       "timeline": [...]
│                │     }
└────────────────┘
```

## Component Details

### 1. **WebSocket Manager** (`websocket_manager.py`)
- Manages active WebSocket connections per worker
- Handles Redis Pub/Sub subscriptions
- Maintains connection registry in Redis
- Sends heartbeats to keep connections alive
- Routes messages to specific users/presentations

**Key Features:**
- Cross-worker message broadcasting via Redis Pub/Sub
- Connection tracking: `ws:connection:{user_id}:{socket_id}`
- Last acknowledged event: `ws:last_ack:{user_id}:{p_id}`
- Worker coordination: Each worker subscribes to `presentation:*`

### 2. **WebSocket Endpoint** (`app_websocket.py`)
- JWT authentication via query parameter
- Status checking (queued/processing/completed/failed)
- Backfill logic for reconnections
- Distributed locking for agent execution
- Client message handling (ping/pong, ack, status_check)

**Key Features:**
- **Fresh connections**: No backfill, only new real-time data
- **Reconnections**: Backfill missed events using `last_event_id`
- **Agent triggering**: Automatic start for "queued" presentations
- **Lock management**: Prevents duplicate agent runs

### 3. **REST API Endpoints** (`app_websocket.py`)
- `/presentation/{p_id}/data` - Get complete presentation data
- `/presentation/{p_id}/status` - Get current status
- JWT authentication via query parameter
- Read-only access to completed presentations

### 4. **Redis Components**

**Pub/Sub Channels:**
- `presentation:{p_id}` - Agent output messages for specific presentation

**Keys:**
- `agent:lock:{p_id}` - Distributed lock (TTL: 600s)
- `ws:connection:{user_id}:{socket_id}` - Active connection (TTL: 60s)
- `ws:last_ack:{user_id}:{p_id}` - Last acknowledged event (TTL: 3600s)

### 5. **MongoDB Collections**

**presentations:**
- Main presentation metadata
- Status tracking (queued → processing → completed/failed)
- Slide data (html_content, resources)
- Timeline events

**agent_outputs_2:**
- Agent execution logs
- Incremental outputs (chunks)
- Event stream for reconnections
- Indexed by: `p_id`, `_id` (timestamp order)

**users:**
- User authentication data
- Package/role information

### 6. **Agent Layer** (`root_agent/`)
- Asynchronous execution
- Multi-agent orchestration
- External tool integration (Qdrant, image search)
- Streaming output to MongoDB + Redis

## Scaling Considerations

### Horizontal Scaling
- ✅ Multiple Gunicorn workers supported
- ✅ Redis Pub/Sub enables cross-worker communication
- ✅ Distributed locking prevents race conditions
- ✅ Stateless workers (state in Redis/MongoDB)

### Production Deployment
```
┌─────────────────┐
│   Load Balancer │
│   (Nginx/ALB)   │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
┌───▼───┐ ┌──▼────┐
│ App 1 │ │ App 2 │  (Docker containers)
│ 4 wrk │ │ 4 wrk │
└───┬───┘ └──┬────┘
    │        │
    └───┬────┘
        │
   ┌────▼─────┐
   │  Redis   │  (Managed service)
   │ Cluster  │
   └──────────┘
        │
   ┌────▼─────┐
   │ MongoDB  │  (Atlas/managed)
   │ Replica  │
   └──────────┘
```

### Performance Metrics
- **WebSocket connections**: 1000+ per worker
- **Agent processing**: 1-2 presentations per worker concurrently
- **Lock TTL**: 600 seconds (prevents stuck locks)
- **Connection TTL**: 60 seconds (heartbeat renewal)
- **Message throughput**: 100+ msg/sec via Redis Pub/Sub

## Security Features

1. **JWT Authentication**
   - Token verification on every connection
   - User authorization checks
   - Package/role validation

2. **Presentation Ownership**
   - User can only access their own presentations
   - Token contains user_id for validation

3. **Rate Limiting** (Recommended)
   - Connection limits per user
   - Request throttling
   - DDoS protection

## Monitoring & Observability

### Logs
- WebSocket connection/disconnection events
- Agent execution lifecycle
- Lock acquisition/release
- Error tracking with stack traces

### Metrics (To Implement)
- Active WebSocket connections
- Agent processing time
- Message delivery latency
- Lock contention
- Error rates

## Technology Stack

| Component | Technology |
|-----------|------------|
| Web Framework | FastAPI |
| WebSocket | FastAPI WebSockets |
| ASGI Server | Uvicorn + Gunicorn |
| Message Broker | Redis Pub/Sub |
| Database | MongoDB |
| Authentication | JWT (PyJWT) |
| Vector Search | Qdrant |
| Language | Python 3.9+ |
| Agent Framework | LangGraph / Custom |

## API Endpoints Summary

### WebSocket
- `ws://host/ws/{p_id}?token={jwt}&last_event_id={optional}`

### REST
- `GET /presentation/{p_id}/data?token={jwt}`
- `GET /presentation/{p_id}/status?token={jwt}`

## Environment Variables

```bash
# MongoDB
MONGO_URI=mongodb://localhost:27017
MONGO_DB_NAME=presentation_db

# Redis
REDIS_URL=redis://localhost:6379/0

# JWT
JWT_SECRET=your-secret-key
JWT_ALGORITHM=HS256

# Agent
AGENT_LOCK_TTL=600

# Google Cloud (for service account)
GOOGLE_APPLICATION_CREDENTIALS=service-account.json
```

## Deployment

```bash
# Development
uvicorn main:app --port 8060

# Production (4 workers)
gunicorn main:app -k uvicorn.workers.UvicornWorker \
  --workers 4 \
  --bind 0.0.0.0:8060 \
  --timeout 600
```

---

**Last Updated:** October 22, 2025
**Version:** 1.0.0

