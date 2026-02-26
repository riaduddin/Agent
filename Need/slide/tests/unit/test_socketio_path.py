"""
Test script to verify Socket.IO path configuration.

This script creates a minimal FastAPI app with Socket.IO mounted at /presentations/socket.io
and tests that connections work correctly.

Run this script and then connect using:
    python -m socketio.client "http://localhost:8000/presentations/socket.io"
Or use a browser-based client pointing to http://localhost:8000/presentations/socket.io
"""

import asyncio
from fastapi import FastAPI
from socketio import AsyncServer, ASGIApp
import uvicorn

# Create FastAPI app
app = FastAPI()

# Create Socket.IO server
sio = AsyncServer(
    cors_allowed_origins="*",
    logger=True,
    engineio_logger=True,
    async_mode='asgi',
    ping_timeout=60,
    ping_interval=25,
)

# Register a test event handler
@sio.event
async def connect(sid, environ):
    print(f"✅ Client connected: {sid}")
    await sio.emit('message', {'data': 'Connected successfully!'}, to=sid)

@sio.event
async def disconnect(sid):
    print(f"❌ Client disconnected: {sid}")

@sio.event
async def test_event(sid, data):
    print(f"📨 Received test_event from {sid}: {data}")
    await sio.emit('response', {'data': f'Echo: {data}'}, to=sid)

# Create Socket.IO ASGI app with socketio_path='' (critical for FastAPI mount)
sio_asgi_app = ASGIApp(sio, socketio_path='')

# Mount at /presentations/socket.io (simulating production)
app.mount("/presentations/socket.io", sio_asgi_app)

# Health check endpoint
@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/")
async def root():
    return {
        "message": "Socket.IO test server",
        "socket_path": "/presentations/socket.io",
        "instructions": "Connect your Socket.IO client to http://localhost:8000/presentations/socket.io"
    }

if __name__ == "__main__":
    print("=" * 80)
    print("🧪 Socket.IO Path Test Server")
    print("=" * 80)
    print("Socket.IO mounted at: /presentations/socket.io")
    print("Test with: python -m socketio.client 'http://localhost:8000/presentations/socket.io'")
    print("Or use browser client pointing to: http://localhost:8000/presentations/socket.io")
    print("=" * 80)
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
