# Quick Test Reference Card

## 🚀 Quick Start Testing

### 1. Import Postman Collection
```bash
# File created: Postman_Collection.json
# Import into Postman → Set variables: base_url, jwt_token
```

### 2. Test REST API (Postman)

```http
# Health Check
GET http://localhost:8000/health

# Create Presentation
POST http://localhost:8000/create-presentation-sse
Authorization: Bearer YOUR_JWT_TOKEN
Body: {"message": "Create a presentation about AI"}

# Check Status
GET http://localhost:8000/presentation/{p_id}/status?token=YOUR_JWT_TOKEN

# Get Data (if completed)
GET http://localhost:8000/presentation/{p_id}/data?token=YOUR_JWT_TOKEN
```

### 3. Test WebSocket (Postman)

**Initial Connection:**
```
ws://localhost:8000/ws/YOUR_P_ID?token=YOUR_JWT_TOKEN
```

**Reconnection:**
```
ws://localhost:8000/ws/YOUR_P_ID?token=YOUR_JWT_TOKEN&last_event_id=LAST_ID
```

**Send Messages:**
```json
// Ping
{"type":"ping","timestamp":"2025-10-21T12:00:00.000Z"}

// Acknowledge
{"type":"ack","event_id":"507f1f77bcf86cd799439011"}

// Status Check
{"type":"status_check"}
```

## 📊 Expected Message Flow

```
1. Connected → {"type":"connected"}
2. Status → {"type":"status","is_reconnection":false}
3. History Events → {"type":"history"} (multiple)
4. Backfill Complete → {"type":"backfill_complete"}
5. Real-time Events → {"type":"chunk"} (ongoing)
6. Heartbeats → {"type":"heartbeat"} (every 30s)
7. Terminal → {"type":"terminal","event":"completed"}
```

## 🔄 Test Reconnection

```javascript
// 1. Connect normally
ws://localhost:8000/ws/P_ID?token=TOKEN

// 2. Note last event_id (e.g., "507f...")

// 3. Disconnect

// 4. Reconnect with last_event_id
ws://localhost:8000/ws/P_ID?token=TOKEN&last_event_id=507f...

// 5. Verify: is_reconnection: true
// 6. Verify: Only NEW events (no duplicates)
```

## 🛠️ Tools Comparison

| Tool | Best For | WebSocket Support |
|------|----------|-------------------|
| **Postman** | API + WebSocket | ✅ Built-in |
| **Browser Console** | Quick tests | ✅ Native |
| **wscat** | CLI testing | ✅ Dedicated |
| **Python** | Automation | ✅ Via library |

## 🐛 Quick Troubleshooting

| Issue | Check |
|-------|-------|
| Connection refused | Server running? `curl localhost:8000/health` |
| No messages | Agent started? Redis running? |
| Duplicate events | Pass `last_event_id` parameter |
| Auth failed | JWT token valid? |

## 📝 Message Types Reference

### Server → Client

| Type | Purpose | When |
|------|---------|------|
| `connected` | Connection confirm | On connect |
| `status` | Current status | After connect |
| `history` | Historical event | During backfill |
| `chunk` | Real-time event | During generation |
| `backfill_complete` | Backfill done | After history |
| `terminal` | Done/failed | At end |
| `heartbeat` | Keep-alive | Every 30s |
| `pong` | Ping response | After ping |
| `status_response` | Status reply | After status_check |

### Client → Server

| Type | Purpose | Example |
|------|---------|---------|
| `ping` | Check latency | `{"type":"ping","timestamp":"..."}` |
| `ack` | Confirm receipt | `{"type":"ack","event_id":"..."}` |
| `status_check` | Request status | `{"type":"status_check"}` |

## 💡 Pro Tips

### Track Events
```javascript
// Save in localStorage
localStorage.setItem('last_event', eventId);

// Use on reconnect
const lastId = localStorage.getItem('last_event');
```

### Monitor Health
```javascript
// Send ping every 15s
setInterval(() => {
  ws.send(JSON.stringify({
    type: 'ping',
    timestamp: new Date().toISOString()
  }));
}, 15000);
```

### Handle Disconnections
```javascript
ws.onclose = () => {
  console.log('Reconnecting...');
  setTimeout(() => reconnect(), 2000);
};
```

## 🔍 Redis Inspection

```bash
# Check connections
redis-cli KEYS ws:connection:*

# Check ACKs
redis-cli KEYS ws:last_ack:*

# Get specific ACK
redis-cli GET "ws:last_ack:USER_ID:P_ID"

# Check agent locks
redis-cli KEYS agent:lock:*

# Monitor all commands
redis-cli MONITOR
```

## 📦 Files Reference

- `Postman_Collection.json` - Import into Postman
- `TESTING_GUIDE.md` - Detailed testing guide
- `WEBSOCKET_SETUP.md` - Setup documentation
- `WEBSOCKET_RECONNECTION_GUIDE.md` - Reconnection details
- `tests/test_websocket_setup.py` - Setup test script
- `tests/test_reconnection.py` - Reconnection test script

## ✅ Test Checklist

- [ ] Import Postman collection
- [ ] Set jwt_token variable
- [ ] Test health endpoint
- [ ] Create presentation
- [ ] Check status
- [ ] Connect WebSocket
- [ ] Receive events
- [ ] Send ping
- [ ] Send ACK
- [ ] Disconnect
- [ ] Reconnect with last_event_id
- [ ] Verify no duplicates
- [ ] Test status_check
- [ ] Verify terminal event

## 🚀 One-Line Tests

```bash
# wscat test
wscat -c "ws://localhost:8000/ws/P_ID?token=TOKEN"

# curl test
curl http://localhost:8000/health

# Redis test
redis-cli ping

# Python test
python tests/test_websocket_setup.py
```

---

**Need Help?** See `TESTING_GUIDE.md` for detailed instructions.

