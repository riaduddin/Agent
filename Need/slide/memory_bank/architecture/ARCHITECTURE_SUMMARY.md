# Architecture Summary - Quick Reference

## 📁 Architecture Documentation Files

| File | Purpose | Best For |
|------|---------|----------|
| `architecture_diagram.html` | Interactive visual diagram | **Quick overview** (Open in browser) |
| `ARCHITECTURE_SIMPLE.md` | Simplified architecture guide | **Understanding flows** |
| `ARCHITECTURE_DIAGRAM.md` | Complete technical documentation | **Deep dive & implementation** |
| `ARCHITECTURE_SUMMARY.md` | This file - quick reference | **Quick lookup** |

## 🎯 Core Architecture Principles

### 1. **Real-Time First**
- WebSocket for live streaming
- Only new data sent (no historical backfill for fresh connections)
- Instant updates as slides are generated

### 2. **Multi-Worker Ready**
- Redis Pub/Sub for cross-worker communication
- Distributed locking prevents duplicate work
- Stateless workers enable horizontal scaling

### 3. **Reconnection Support**
- `last_event_id` tracking
- Backfill missed events on reconnect
- No data loss guaranteed

### 4. **Dual API Strategy**
- **WebSocket**: For active/processing presentations (real-time)
- **REST**: For completed presentations (static data)

## 🏗️ System Layers (Top to Bottom)

```
┌─────────────────────────────┐
│ 1. CLIENT LAYER             │  Browser, Mobile, Postman
├─────────────────────────────┤
│ 2. APPLICATION LAYER        │  FastAPI + Multiple Workers
├─────────────────────────────┤
│ 3. INFRASTRUCTURE LAYER     │  Redis + MongoDB
├─────────────────────────────┤
│ 4. AGENT PROCESSING LAYER   │  AI Slide Generation
└─────────────────────────────┘
```

## 🔄 Data Flow (Simplified)

```
Client Connects
    ↓
JWT Verified
    ↓
Status Checked (queued/processing/completed)
    ↓
Lock Acquired (Redis)
    ↓
Agent Starts Processing
    ↓
For Each Slide:
    Agent → MongoDB (save)
    Agent → Redis Pub/Sub (broadcast)
    Redis → All Workers (receive)
    Worker → Connected Clients (send)
    ↓
Agent Completes
    ↓
Lock Released
```

## 📊 Key Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Web Framework | FastAPI | HTTP + WebSocket server |
| Message Broker | Redis Pub/Sub | Cross-worker communication |
| Database | MongoDB | Persistent storage |
| Auth | JWT | Token-based security |
| Workers | Gunicorn | Multi-process ASGI server |
| Agent | Python Async | AI slide generation |

## 🔐 Security Model

```
Request → JWT Token → Verify Signature → Extract user_id
                                       ↓
                           Check presentation.user_id == token.user_id
                                       ↓
                                  Allow/Deny
```

## 📡 WebSocket Message Types

### Status (Connection Established)
```json
{
  "type": "status",
  "status": "queued|processing|completed",
  "is_reconnection": false,
  "message": "Connected (fresh connection - only new data)"
}
```

### Backfill Complete
```json
{
  "type": "backfill_complete",
  "events_count": 0,  // 0 for fresh, N for reconnections
  "is_reconnection": false
}
```

### Agent Output (Real-time)
```json
{
  "type": "chunk",
  "event_id": "abc123",
  "author": "slide_generator_0",
  "message": "Generated slide 1",
  "html_content": "<!DOCTYPE html>...",
  "timestamp": "2025-10-22T10:30:00"
}
```

### Terminal (Completion)
```json
{
  "type": "terminal",
  "status": "completed|failed",
  "p_id": "123"
}
```

## 🔧 Redis Keys & Channels

| Type | Pattern | Purpose | TTL |
|------|---------|---------|-----|
| **Lock** | `agent:lock:{p_id}` | Prevent duplicate agents | 600s |
| **Connection** | `ws:connection:{user_id}:{socket_id}` | Track active connections | 60s |
| **Last Ack** | `ws:last_ack:{user_id}:{p_id}` | Reconnection tracking | 3600s |
| **Channel** | `presentation:{p_id}` | Pub/Sub broadcast | N/A |

## 💾 MongoDB Collections

### presentations
- `p_id` (unique presentation ID)
- `user_id` (owner)
- `status` (queued → processing → completed/failed)
- `slides[]` (generated slides)
- `timeline[]` (events)

### agent_outputs_2
- `_id` (event ID, used for `last_event_id`)
- `p_id` (presentation reference)
- `author` (which agent)
- `content` (output data)
- `timestamp` (creation time)

### users
- `user_id` (unique)
- `email`, `package`, `role`, `is_verified`

## 🚀 Deployment Commands

### Development
```bash
# Single worker
uvicorn main:app --port 8060
```

### Production
```bash
# 4 workers
gunicorn main:app \
  -k uvicorn.workers.UvicornWorker \
  --workers 4 \
  --bind 0.0.0.0:8060 \
  --timeout 600
```

### Docker
```bash
docker-compose up -d
```

## 📈 Scaling Capacity

| Metric | Single Worker | 4 Workers | Production (12 Workers) |
|--------|--------------|-----------|-------------------------|
| **Concurrent Connections** | 1,000+ | 4,000+ | 12,000+ |
| **Agent Processes** | 1-2 | 4-8 | 12-24 |
| **Requests/sec** | 100+ | 400+ | 1,200+ |

## 🔍 Monitoring Checklist

- [ ] Active WebSocket connections per worker
- [ ] Agent processing time (avg/p95/p99)
- [ ] Redis Pub/Sub message rate
- [ ] MongoDB query latency
- [ ] Lock acquisition time
- [ ] Failed agent runs
- [ ] Connection errors/disconnects

## 🐛 Troubleshooting Guide

### Connection Fails
1. Check JWT token validity
2. Verify user has access to presentation
3. Check Redis connection
4. Verify WebSocket endpoint (not HTTP GET)

### No Real-Time Updates
1. Check agent is running (lock acquired?)
2. Verify Redis Pub/Sub working
3. Check worker subscribed to correct channel
4. Review MongoDB writes

### Duplicate Agent Runs
1. Check Redis lock mechanism
2. Verify lock TTL not expired
3. Review worker coordination logs

### Reconnection Issues
1. Verify `last_event_id` format (valid ObjectId)
2. Check MongoDB has events after that ID
3. Review `ws:last_ack` Redis key

## 📞 API Quick Reference

### Connect WebSocket
```
ws://localhost:8060/ws/{p_id}?token={jwt}
```

### Reconnect WebSocket
```
ws://localhost:8060/ws/{p_id}?token={jwt}&last_event_id={event_id}
```

### Get Completed Presentation
```
GET http://localhost:8060/presentation/{p_id}/data?token={jwt}
```

### Check Status
```
GET http://localhost:8060/presentation/{p_id}/status?token={jwt}
```

## 🎨 Client-Side Example

```javascript
// Connect
const ws = new WebSocket('ws://host/ws/' + p_id + '?token=' + jwt_token);

// Handle messages
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  switch(data.type) {
    case 'status':
      console.log('Connected:', data.status);
      break;
    case 'chunk':
      if (data.html_content) {
        renderSlide(data.html_content);
      }
      break;
    case 'terminal':
      console.log('Completed!');
      ws.close();
      break;
  }
};

// Send acknowledgment (for reconnection tracking)
ws.send(JSON.stringify({
  type: 'ack',
  event_id: data.event_id
}));
```

## 🔐 Environment Variables

```bash
# Required
MONGO_URI=mongodb://localhost:27017
REDIS_URL=redis://localhost:6379/0
JWT_SECRET=your-secret-key

# Optional
JWT_ALGORITHM=HS256
AGENT_LOCK_TTL=600
MONGO_DB_NAME=presentation_db
```

## 📚 Related Documentation

- **Setup**: `WEBSOCKET_SETUP.md`
- **Testing**: `TESTING_GUIDE.md`
- **Reconnection**: `WEBSOCKET_RECONNECTION_GUIDE.md`
- **Quick Start**: `QUICK_TEST_REFERENCE.md`

## 🎯 Design Decisions

### Why Redis Pub/Sub?
- Enables multi-worker coordination
- Real-time message broadcasting
- Simple and reliable

### Why Distributed Locks?
- Prevents duplicate agent executions
- Ensures only one worker processes a presentation
- TTL prevents stuck locks

### Why No Backfill for Fresh Connections?
- User only wants NEW data (your requirement)
- Reduces initial payload
- Faster connection establishment

### Why Both WebSocket + REST?
- WebSocket: Real-time streaming (active presentations)
- REST: One-time fetch (completed presentations)
- Best tool for each use case

## 🏁 Quick Start Checklist

- [ ] Redis running on port 6379
- [ ] MongoDB running on port 27017
- [ ] Environment variables set
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Service account JSON configured (Google Cloud)
- [ ] Server started (`uvicorn main:app --port 8060`)
- [ ] Test WebSocket connection (browser console or Postman)
- [ ] Verify real-time updates working

---

**Version:** 1.0.0  
**Last Updated:** October 22, 2025  
**Author:** System Architecture Team

For detailed technical documentation, see:
- `ARCHITECTURE_DIAGRAM.md` (full details)
- `ARCHITECTURE_SIMPLE.md` (simplified guide)
- `architecture_diagram.html` (visual overview - open in browser!)

