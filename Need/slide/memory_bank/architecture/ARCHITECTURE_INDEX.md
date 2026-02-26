# 🏗️ Architecture Documentation Index

Welcome to the Presentation Generation Service architecture documentation! This index helps you find the right documentation for your needs.

---

## 📖 Quick Navigation

### 🎯 **Start Here**
- **[architecture_diagram.html](architecture_diagram.html)** - 🌟 **BEST FOR VISUAL LEARNERS** - Open in your browser for an interactive, beautiful architecture diagram

### 📚 **Complete Documentation**

| Document | Best For | Length |
|----------|----------|--------|
| **[ARCHITECTURE_SUMMARY.md](ARCHITECTURE_SUMMARY.md)** | Quick reference & lookup | ⚡ Quick |
| **[ARCHITECTURE_SIMPLE.md](ARCHITECTURE_SIMPLE.md)** | Understanding flows & components | 📄 Medium |
| **[ARCHITECTURE_DIAGRAM.md](ARCHITECTURE_DIAGRAM.md)** | Deep technical dive | 📚 Complete |
| **[architecture_diagram.html](architecture_diagram.html)** | Visual overview | 🎨 Interactive |

---

## 🎯 Choose Your Path

### "I want a quick visual overview"
👉 **Open `architecture_diagram.html` in your browser**

Beautiful, interactive diagram showing:
- All system layers
- Component interactions
- Data flow
- Tech stack
- Key features

### "I need to look something up quickly"
👉 **Read `ARCHITECTURE_SUMMARY.md`**

Quick reference for:
- Message types
- API endpoints
- Redis keys
- MongoDB collections
- Environment variables
- Troubleshooting

### "I want to understand how it works"
👉 **Read `ARCHITECTURE_SIMPLE.md`**

Covers:
- Core components
- Data flows
- Multi-worker coordination
- Message types
- Scaling strategy

### "I need complete technical details"
👉 **Read `ARCHITECTURE_DIAGRAM.md`**

Comprehensive documentation:
- Detailed architecture diagrams
- Complete data flow scenarios
- Component specifications
- Performance metrics
- Deployment strategies
- Monitoring & observability

---

## 🚀 Architecture Overview (Quick Summary)

```
CLIENT (Browser/Mobile)
    ↓
JWT Authentication
    ↓
FASTAPI (Multiple Workers)
    ├── WebSocket Manager (Real-time)
    └── REST API (Static data)
    ↓
REDIS (Pub/Sub + Locks)
    ↓
MONGODB (Persistent Storage)
    ↓
AGENT LAYER (AI Processing)
```

**Key Features:**
- ✅ Real-time WebSocket streaming
- ✅ Multi-worker support (Redis Pub/Sub)
- ✅ Distributed locking (no duplicate work)
- ✅ Reconnection with backfill
- ✅ Only new data sent to fresh connections
- ✅ JWT authentication
- ✅ Horizontal scaling ready

---

## 📡 API Endpoints

### WebSocket (Real-time)
```
ws://localhost:8060/ws/{p_id}?token={jwt}&last_event_id={optional}
```

### REST (Static data)
```
GET /presentation/{p_id}/data?token={jwt}
GET /presentation/{p_id}/status?token={jwt}
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Framework | FastAPI + Uvicorn |
| WebSocket | FastAPI WebSockets |
| Message Broker | Redis Pub/Sub |
| Database | MongoDB |
| Authentication | JWT (PyJWT) |
| Agent Runtime | Python Async |
| Deployment | Docker + Gunicorn |

---

## 📊 Key Design Decisions

### Why WebSocket + REST?
- **WebSocket**: Real-time streaming for active presentations
- **REST**: One-time fetch for completed presentations
- Best tool for each use case

### Why Redis Pub/Sub?
- Enables multi-worker communication
- Broadcasts agent updates to all workers
- Simple and reliable

### Why Distributed Locks?
- Prevents duplicate agent execution
- Only one worker processes each presentation
- TTL prevents stuck locks (600s)

### Why No Backfill for Fresh Connections?
- **User requirement**: Only want NEW data
- Faster connection establishment
- Reduced initial payload
- Backfill only on reconnections with `last_event_id`

---

## 🔄 Data Flow (Simplified)

```
1. Client connects via WebSocket
2. Server verifies JWT token
3. Server sends status (queued/processing/completed)
4. Server sends backfill_complete (0 events for fresh)
5. Worker acquires lock in Redis
6. Agent starts processing in background
7. For each slide:
   a. Agent writes to MongoDB
   b. Agent publishes to Redis Pub/Sub
   c. All workers receive message
   d. Each worker sends to their connected clients
8. Agent completes
9. Server sends terminal message
10. Worker releases lock
```

---

## 🎨 Message Types

### Status
```json
{
  "type": "status",
  "status": "queued",
  "is_reconnection": false,
  "message": "Connected (fresh connection - only new data)"
}
```

### Agent Output (Real-time)
```json
{
  "type": "chunk",
  "event_id": "abc123",
  "author": "slide_generator_0",
  "html_content": "<!DOCTYPE html>...",
  "timestamp": "2025-10-22T10:30:00"
}
```

### Terminal (Completion)
```json
{
  "type": "terminal",
  "status": "completed"
}
```

---

## 🧪 Testing Documentation

- **[TESTING_GUIDE.md](TESTING_GUIDE.md)** - Complete testing guide with Postman
- **[QUICK_TEST_REFERENCE.md](QUICK_TEST_REFERENCE.md)** - Quick testing checklist
- **[tests/test_websocket_setup.py](tests/test_websocket_setup.py)** - Automated setup verification
- **[tests/test_reconnection.py](tests/test_reconnection.py)** - Reconnection testing

---

## 🔧 Implementation Files

### Core WebSocket Implementation
- **[websocket_manager.py](websocket_manager.py)** - Connection management & Redis Pub/Sub
- **[app_websocket.py](app_websocket.py)** - WebSocket & REST endpoints
- **[main.py](main.py)** - FastAPI app with lifespan management
- **[auth_middleware.py](auth_middleware.py)** - JWT authentication

### Agent Layer
- **[root_agent/slide_creation_agent/](root_agent/slide_creation_agent/)** - AI slide generation agents

### Configuration
- **[requirements.txt](requirements.txt)** - Python dependencies
- **[docker-compose.yml](docker-compose.yml)** - Docker setup (Redis + MongoDB)
- **[Dockerfile](Dockerfile)** - Container image

---

## 📈 Scaling

### Development (Current)
```bash
uvicorn main:app --port 8060
```
- 1 worker
- ~1,000 concurrent connections

### Production (Recommended)
```bash
gunicorn main:app -k uvicorn.workers.UvicornWorker \
  --workers 4 \
  --bind 0.0.0.0:8060 \
  --timeout 600
```
- 4 workers per instance
- ~4,000+ concurrent connections per instance
- Horizontal scaling with load balancer

### Large Scale
```
Load Balancer
    ↓
┌─────┬─────┬─────┐
│ App1│ App2│ App3│  (3 instances × 4 workers)
└─────┴─────┴─────┘
    ↓
Redis Cluster + MongoDB Replica
```
- 12 workers total
- 12,000+ concurrent connections
- 12-24 concurrent agent processes

---

## 🔐 Security

1. **JWT Authentication**
   - Token verification on all endpoints
   - User ownership validation
   - Package/role checking

2. **Presentation Access Control**
   - Users can only access their own presentations
   - Token contains `user_id` for validation

3. **Environment Variables**
   ```bash
   JWT_SECRET=your-secret-key  # Change in production!
   ```

---

## 🐛 Troubleshooting

### WebSocket Connection Fails
1. Check JWT token validity
2. Verify `ws://` protocol (not `http://`)
3. Ensure Redis is running
4. Check user has access to presentation

### No Real-Time Updates
1. Verify agent is running (check logs for lock)
2. Check Redis Pub/Sub connection
3. Verify MongoDB writes
4. Check worker subscribed to correct channel

### Detailed Troubleshooting
See **[ARCHITECTURE_SUMMARY.md](ARCHITECTURE_SUMMARY.md)** - Troubleshooting section

---

## 📚 Additional Documentation

### Memory Bank (Legacy Documentation)
- **[memory_bank/INDEX.md](memory_bank/INDEX.md)** - Complete memory bank index
- **[memory_bank/architecture/SYSTEM_OVERVIEW.md](memory_bank/architecture/SYSTEM_OVERVIEW.md)** - Original system overview
- **[memory_bank/guides/QUICK_START.md](memory_bank/guides/QUICK_START.md)** - Quick start guide

### WebSocket-Specific
- **[WEBSOCKET_SETUP.md](WEBSOCKET_SETUP.md)** - Setup guide
- **[WEBSOCKET_IMPLEMENTATION_SUMMARY.md](WEBSOCKET_IMPLEMENTATION_SUMMARY.md)** - Implementation summary
- **[WEBSOCKET_RECONNECTION_GUIDE.md](WEBSOCKET_RECONNECTION_GUIDE.md)** - Reconnection guide

---

## 🎓 Learning Path

### Beginner (30 minutes)
1. Open **architecture_diagram.html** in browser
2. Skim **ARCHITECTURE_SIMPLE.md**
3. Test WebSocket connection

### Intermediate (2 hours)
1. Read **ARCHITECTURE_SIMPLE.md** completely
2. Read **ARCHITECTURE_SUMMARY.md** for reference
3. Follow **TESTING_GUIDE.md** examples
4. Understand message types and flows

### Advanced (4+ hours)
1. Read **ARCHITECTURE_DIAGRAM.md** completely
2. Study **websocket_manager.py** implementation
3. Review agent layer code
4. Implement custom client
5. Deploy with multiple workers
6. Monitor and optimize

---

## ✅ Quick Start Checklist

- [ ] Redis running (`docker-compose up redis`)
- [ ] MongoDB running (`docker-compose up mongodb`)
- [ ] Environment variables set (`JWT_SECRET`, `REDIS_URL`, `MONGO_URI`)
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Server started (`uvicorn main:app --port 8060`)
- [ ] WebSocket test successful
- [ ] Real-time updates working

---

## 📞 Support Resources

1. **Architecture Questions**: Read the appropriate documentation above
2. **Implementation Issues**: Check troubleshooting sections
3. **Testing Help**: See **TESTING_GUIDE.md**
4. **API Questions**: See **ARCHITECTURE_SUMMARY.md** - API Reference

---

## 🔄 Documentation Updates

**Last Updated:** October 22, 2025  
**Version:** 1.0.0  
**Status:** ✅ Complete & Current

### Recent Changes
- ✅ Created comprehensive architecture documentation
- ✅ Added interactive HTML diagram
- ✅ Documented WebSocket implementation
- ✅ Added testing guides
- ✅ Included troubleshooting sections

---

## 🌟 Quick Links Summary

**Visual:** [architecture_diagram.html](architecture_diagram.html) 🎨  
**Quick Ref:** [ARCHITECTURE_SUMMARY.md](ARCHITECTURE_SUMMARY.md) ⚡  
**Understanding:** [ARCHITECTURE_SIMPLE.md](ARCHITECTURE_SIMPLE.md) 📄  
**Deep Dive:** [ARCHITECTURE_DIAGRAM.md](ARCHITECTURE_DIAGRAM.md) 📚  
**Testing:** [TESTING_GUIDE.md](TESTING_GUIDE.md) 🧪

---

**Happy coding! 🚀**

