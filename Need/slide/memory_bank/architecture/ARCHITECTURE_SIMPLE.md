# Presentation Generation Service - Architecture Overview

## Quick Visual

```
                    ┌─────────────────────┐
                    │   CLIENT DEVICES    │
                    │  (Browser/Mobile)   │
                    └──────────┬──────────┘
                               │
                     ┌─────────┴─────────┐
                     │                   │
                 WebSocket           REST API
              (Real-time data)    (Static data)
                     │                   │
                     └─────────┬─────────┘
                               │
                    ┌──────────▼──────────┐
                    │   FASTAPI SERVER    │
                    │  (Multiple Workers) │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
     ┌────────▼────────┐  ┌───▼────┐  ┌────────▼────────┐
     │  WEBSOCKET MGR  │  │  REDIS │  │   REST ROUTES   │
     │  (Real-time)    │◄─┤ Pub/Sub├─►│  (HTTP Endpoints)│
     └────────┬────────┘  │ +Locks │  └────────┬────────┘
              │           └────────┘           │
              │                                │
              └────────────┬───────────────────┘
                           │
                  ┌────────▼────────┐
                  │    MONGODB      │
                  │ (Presentations, │
                  │ Events, Users)  │
                  └────────┬────────┘
                           │
                  ┌────────▼────────┐
                  │  AGENT LAYER    │
                  │  (AI Processing)│
                  └─────────────────┘
```

## Core Components

### 1. **Client Layer**
- Connects via WebSocket for real-time streaming
- Uses REST API for completed presentations
- JWT token authentication

### 2. **FastAPI Server (Multiple Workers)**
Each worker runs independently with:
- WebSocket Manager (handles connections)
- REST API Routes (handles HTTP requests)
- Shared Redis connection (for coordination)

### 3. **Redis (Message Broker + State)**
- **Pub/Sub**: Broadcasts agent messages to all workers
- **Locks**: Prevents duplicate agent execution
- **State**: Tracks connections and last events

### 4. **MongoDB (Persistent Storage)**
- Stores presentations, user data, events
- Enables reconnection with event history
- Powers REST API responses

### 5. **Agent Layer (AI Processing)**
- Generates slides based on user prompts
- Runs asynchronously in background
- Sends updates via MongoDB + Redis

## Key Flows

### Fresh Connection
```
Client → WebSocket → Worker → Check Status (queued)
                            ↓
                     Start Agent Processing
                            ↓
              Agent writes to MongoDB + Redis
                            ↓
               Redis broadcasts to all Workers
                            ↓
          Worker sends to connected Client (real-time)
```

### Reconnection
```
Client → WebSocket (with last_event_id) → Worker
                                        ↓
                            Check MongoDB for missed events
                                        ↓
                            Send backfilled events to Client
                                        ↓
                            Continue real-time streaming
```

### Completed Presentation
```
Client → REST API → Worker → MongoDB (get full data)
                            ↓
                    Return complete JSON response
```

## Multi-Worker Coordination

```
┌──────────┐  ┌──────────┐  ┌──────────┐
│ Worker 1 │  │ Worker 2 │  │ Worker 3 │
└─────┬────┘  └─────┬────┘  └─────┬────┘
      │             │             │
      └─────────────┼─────────────┘
                    │
              ┌─────▼─────┐
              │   REDIS   │
              │  Pub/Sub  │
              └───────────┘
```

**How it works:**
1. Worker 1 acquires lock in Redis
2. Worker 1 starts agent processing
3. Agent writes output to MongoDB
4. Agent publishes to Redis channel
5. **All workers** receive the message
6. Each worker sends to **their connected clients**

## Data Flow Timeline

```
Time  │ Action
──────┼─────────────────────────────────────────
T0    │ Client connects via WebSocket
T1    │ Server checks status: "queued"
T2    │ Server sends status + backfill_complete
T3    │ Worker acquires lock in Redis
T4    │ Agent starts processing
T5    │ Agent: "Extracting presentation specs..."
      │ ↓ Write to MongoDB + Publish to Redis
T6    │ Worker receives from Redis → Client
T7    │ Agent: "Generated slide 1"
      │ ↓ Write to MongoDB + Publish to Redis
T8    │ Worker receives from Redis → Client
...   │ ... (continues for each slide)
T99   │ Agent: "Presentation completed"
      │ ↓ Write to MongoDB + Update status
T100  │ Worker sends terminal message → Client
T101  │ Worker releases lock in Redis
```

## Message Types

### Status Messages
```json
{
  "type": "status",
  "p_id": "123",
  "status": "queued|processing|completed|failed",
  "is_reconnection": false,
  "message": "Connected to presentation (fresh connection - only new data will be sent)"
}
```

### Backfill Complete
```json
{
  "type": "backfill_complete",
  "p_id": "123",
  "events_count": 0,
  "is_reconnection": false
}
```

### Agent Output (Chunk)
```json
{
  "type": "chunk",
  "event_id": "67890",
  "p_id": "123",
  "author": "enhanced_slide_generator_0",
  "message": "Generated slide 1",
  "html_content": "<!DOCTYPE html>...",
  "timestamp": "2025-10-22T10:30:00"
}
```

### Terminal Message
```json
{
  "type": "terminal",
  "event": "completed",
  "status": "completed",
  "p_id": "123"
}
```

## Security Model

```
Client → JWT Token → Server → Verify Token → Extract user_id
                                           ↓
                              Check presentation ownership
                                           ↓
                              user_id matches presentation.user_id?
                                           ↓
                                    Allow connection
```

## Scaling Strategy

### Current (Development)
- Single server, 1 worker
- Redis + MongoDB on localhost

### Small Scale (Staging)
- Single server, 4 workers
- Redis + MongoDB on cloud

### Production (Large Scale)
```
                ┌─────────────┐
                │ LOAD BALANCER│
                └──────┬──────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
    ┌───▼───┐      ┌───▼───┐     ┌───▼───┐
    │ App 1 │      │ App 2 │     │ App 3 │
    │4 wrkrs│      │4 wrkrs│     │4 wrkrs│
    └───┬───┘      └───┬───┘     └───┬───┘
        │              │              │
        └──────────────┼──────────────┘
                       │
            ┌──────────┴──────────┐
            │                     │
       ┌────▼────┐          ┌─────▼─────┐
       │ Redis   │          │  MongoDB  │
       │ Cluster │          │  Replica  │
       └─────────┘          └───────────┘
```

**Capacity:**
- 12 workers total (3 apps × 4 workers)
- 12,000+ concurrent WebSocket connections
- 12-24 concurrent agent processes
- Load balanced by connection count

## Monitoring Points

1. **Connection Health**
   - Active WebSocket connections per worker
   - Connection duration
   - Disconnection rate

2. **Agent Performance**
   - Processing time per presentation
   - Success/failure rate
   - Queue depth

3. **Infrastructure**
   - Redis Pub/Sub message rate
   - MongoDB query latency
   - Worker CPU/Memory usage

4. **Lock Management**
   - Lock acquisition time
   - Lock contention rate
   - Expired locks (failures)

## File Structure

```
presentation-gen-service/
├── main.py                      # FastAPI app entry point
├── app_websocket.py             # WebSocket + REST endpoints
├── websocket_manager.py         # WebSocket connection manager
├── auth_middleware.py           # JWT authentication
├── database.py / db.py          # MongoDB connection
├── root_agent/                  # Agent processing layer
│   └── slide_creation_agent/    # Slide generation agents
├── requirements.txt             # Python dependencies
├── docker-compose.yml           # Docker setup
└── ARCHITECTURE_DIAGRAM.md      # This file
```

## Quick Start

```bash
# 1. Start dependencies
docker-compose up -d redis mongodb

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set environment variables
export MONGO_URI="mongodb://localhost:27017"
export REDIS_URL="redis://localhost:6379/0"
export JWT_SECRET="your-secret-key"

# 4. Run development server
uvicorn main:app --port 8060

# 5. Connect via WebSocket
# ws://localhost:8060/ws/{p_id}?token={jwt_token}
```

---

**Key Takeaways:**
1. ✅ WebSocket for real-time streaming (new data only)
2. ✅ REST API for completed presentations (full data)
3. ✅ Redis coordinates multiple workers
4. ✅ MongoDB stores all persistent data
5. ✅ Distributed locking prevents duplicate work
6. ✅ JWT authentication secures all endpoints
7. ✅ Reconnection support with event backfilling

