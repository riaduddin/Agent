# socketio_manager.py
import asyncio
import json
import logging
import os
from typing import Dict, Optional, Set
from datetime import datetime, timezone
import redis.asyncio as redis
from bson import ObjectId
import socketio

logger = logging.getLogger(__name__)

def get_utc_timestamp_iso() -> str:
    """
    Get current UTC timestamp in ISO 8601 format with timezone info.
    Returns format like: '2024-01-15T10:30:45.123456+00:00'
    This allows frontend to properly convert to user's local timezone.
    """
    return datetime.now(timezone.utc).isoformat()

class SocketIOConnectionManager:
    """
    Multi-worker Socket.IO connection manager using Redis.
    
    Features:
    - ✅ Works across multiple Gunicorn workers via Redis Pub/Sub
    - ✅ One user can connect to MULTIPLE presentations simultaneously
    - ✅ Each presentation (p_id) is PRIVATE - only the owner can see it
    - ✅ Multiple connections from same user (different tabs/devices)
    - ✅ Automatic reconnection handling and cleanup
    """
    
    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        
        # Local session tracking on this worker
        # Format: {session_id: {"user_id": str, "p_id": str}}
        self.sessions: Dict[str, Dict[str, str]] = {}
        
        # Quick lookup: which presentations is each user watching?
        # Format: {user_id: Set[p_id]}
        self.user_presentations: Dict[str, Set[str]] = {}
        
        # Quick lookup: which sessions are watching each presentation?
        # Format: {p_id: Set[session_id]}
        self.presentation_sessions: Dict[str, Set[str]] = {}
        
        # Redis clients
        self.redis_client: Optional[redis.Redis] = None
        self.redis_pubsub: Optional[redis.client.PubSub] = None
        
        # Background tasks
        self._subscriber_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        
        # Worker identification
        self._worker_id = str(ObjectId())
        self._worker_pid = os.getpid()
        
        # Socket.IO server instance - will be set during initialization
        self.sio: Optional[socketio.AsyncServer] = None
    
    def set_sio_server(self, sio: socketio.AsyncServer):
        """Set the Socket.IO server instance and setup handlers"""
        global _sio_server
        
        if sio is None:
            logger.error("❌ Cannot set None as Socket.IO server")
            return False
            
        self.sio = sio
        _sio_server = sio  # Store globally for handlers
        logger.info(f"✅ Socket.IO server received: {type(self.sio)}")
        logger.info(f"📋 Has enter_room method: {hasattr(self.sio, 'enter_room')}")
        logger.info(f"🔍 Manager instance ID when setting sio: {id(self)}")
        logger.info(f"🔍 Global _sio_server set to: {_sio_server}")
        logger.info(f"🔍 Global _sio_server type: {type(_sio_server)}")
        logger.info(f"🔍 Global _sio_server is None: {_sio_server is None}")
        
        # Validate critical methods exist
        required_methods = ['enter_room', 'emit', 'on']
        for method in required_methods:
            if not hasattr(self.sio, method):
                logger.error(f"❌ Socket.IO server missing required method: {method}")
                return False
        
        # Setup handlers
        self.setup_handlers()
        
        # Verify handlers were registered
        if hasattr(self.sio, 'handlers'):
            total_handlers = sum(len(handlers) for handlers in self.sio.handlers.values())
            logger.info(f"✅ Total handlers registered: {total_handlers}")
            for namespace, handlers in self.sio.handlers.items():
                logger.info(f"   Namespace '{namespace}': {list(handlers.keys())}")
        else:
            logger.warning("⚠️ Socket.IO server has no handlers attribute")
        
        return True
    
    def setup_handlers(self):
        """Setup Socket.IO event handlers"""
        if self.sio is None:
            logger.error("❌ Cannot setup handlers: Socket.IO server not initialized")
            return
        
        logger.info("🔧 Setting up Socket.IO handlers...")
        
        # ✅ CRITICAL: Use self reference instead of capturing sio_ref
        manager_ref = self
        
        # Define the connect handler
        async def connect_handler(sid, environ):
            """Handle Socket.IO connection"""
            try:
                logger.info(f"📞 Connect handler called for sid={sid}")
                logger.info("=" * 50)
                logger.info("🚀 STARTING CONNECTION PROCESS")
                logger.info("=" * 50)
                
                # Check global _sio_server at handler start
                global _sio_server
                logger.info(f"🔍 Global _sio_server at handler start: {_sio_server}")
                logger.info(f"🔍 Global _sio_server type at handler start: {type(_sio_server)}")
                logger.info(f"🔍 Global _sio_server is None at handler start: {_sio_server is None}")
                
                # Get the current manager instance dynamically
                logger.info("📍 STEP 1: Getting manager instance...")
                # current_manager = await get_manager()
                # Use global _manager instead of importing
                global _manager
                if _manager is None:
                     logger.error("❌ Global _manager is None in connect_handler")
                     return False
                current_manager = _manager
                logger.info(f"✅ Manager obtained: {id(current_manager)}")
                
                # Use global SocketIO server reference
                logger.info("📍 STEP 2: Checking global SocketIO server...")
                if _sio_server is None:
                    logger.error("❌ FAILED AT STEP 2: Global SocketIO server not available")
                    logger.error("=" * 50)
                    logger.error("🚨 CONNECTION FAILED - NO SOCKETIO SERVER")
                    logger.error("=" * 50)
                    return False
                
                # Debug logging
                logger.info(f"✅ Global sio server available: {type(_sio_server)}")
                logger.info(f"✅ Global sio server is None: {_sio_server is None}")
                
                # Extract query parameters from URL
                logger.info("📍 STEP 3: Extracting query parameters...")
                query_string = environ.get('QUERY_STRING', '')
                params = {}
                for param in query_string.split('&'):
                    if '=' in param:
                        key, value = param.split('=', 1)
                        from urllib.parse import unquote
                        params[key] = unquote(value)
                
                p_id = params.get('p_id')
                token = params.get('token')
                
                logger.info(f"✅ Query params extracted: p_id={p_id}, has_token={bool(token)}")
                
                if not p_id or not token:
                    logger.error("❌ FAILED AT STEP 3: Missing p_id or token")
                    logger.error("=" * 50)
                    logger.error("🚨 CONNECTION FAILED - MISSING PARAMETERS")
                    logger.error("=" * 50)
                    return False
                
                # Authenticate user
                logger.info("📍 STEP 4: Authenticating user...")
                try:
                    import jwt
                    JWT_SECRET = os.getenv("JWT_SECRET", "your-secret-key-here")
                    JWT_ALGORITHM = "HS256"
                    
                    payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
                    user_id = payload.get("sub") or payload.get("_id")
                    
                    if not user_id:
                        logger.error("❌ FAILED AT STEP 4: Invalid token - no user_id")
                        logger.error("=" * 50)
                        logger.error("🚨 CONNECTION FAILED - INVALID TOKEN")
                        logger.error("=" * 50)
                        return False
                    
                    logger.info(f"✅ User authenticated: {user_id}")
                    
                except Exception as e:
                    logger.error(f"❌ FAILED AT STEP 4: Authentication failed: {e}")
                    logger.error("=" * 50)
                    logger.error("🚨 CONNECTION FAILED - AUTHENTICATION ERROR")
                    logger.error("=" * 50)
                    return False
                
                # Verify presentation ownership (optional - allow connection even if DB is down)
                logger.info("📍 STEP 5: Verifying presentation ownership...")
                try:
                    from core.database import get_mongo_client
                    client = get_mongo_client()
                    db = client["slide_creator_db"]
                    
                    presentation = db.presentations.find_one({"p_id": p_id})
                    
                    if not presentation:
                        logger.warning(f"⚠️ Presentation not found in DB: {p_id} - allowing connection anyway")
                    elif presentation.get("user_id") != user_id:
                        logger.warning(f"⚠️ User mismatch in DB - allowing connection anyway")
                    else:
                        logger.info(f"✅ Database verification passed")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Database unavailable: {e} - allowing connection anyway")
                    # Continue with connection even if database is not available
                
                logger.info("✅ Database verification completed (or skipped)")
                
                # Register connection
                logger.info("📍 STEP 6: Registering connection...")
                await current_manager._register_connection(user_id, sid, p_id)
                logger.info("✅ Connection registered")
                
                # ✅ Use global _sio_server reference
                logger.info("📍 STEP 7: Joining presentation room...")
                try:
                    logger.info(f"🚪 Attempting to join room: presentation:{p_id}")
                    logger.info(f"🔍 Global _sio_server: {_sio_server}")
                    logger.info(f"🔍 Global _sio_server type: {type(_sio_server)}")
                    logger.info(f"🔍 Global _sio_server is None: {_sio_server is None}")
                    
                    # Check if _sio_server has the enter_room method
                    if hasattr(_sio_server, 'enter_room'):
                        logger.info("✅ _sio_server has enter_room method")
                        logger.info(f"🔍 enter_room method: {getattr(_sio_server, 'enter_room')}")
                    else:
                        logger.error("❌ _sio_server does NOT have enter_room method")
                        logger.error("=" * 50)
                        logger.error("🚨 CONNECTION FAILED - NO ENTER_ROOM METHOD")
                        logger.error("=" * 50)
                        return False
                    
                    # Try to call enter_room
                    logger.info("🚀 Calling _sio_server.enter_room...")
                    _sio_server.enter_room(sid, f"presentation:{p_id}")
                    logger.info(f"✅ Successfully joined room: presentation:{p_id}")
                    
                except Exception as room_error:
                    logger.error("❌ FAILED AT STEP 7: Failed to join room")
                    logger.error("=" * 50)
                    logger.error("🚨 CONNECTION FAILED - ROOM JOIN ERROR")
                    logger.error(f"Error details: {room_error}")
                    logger.error(f"Error type: {type(room_error)}")
                    logger.error(f"Global _sio_server at error time: {_sio_server}")
                    logger.error(f"Global _sio_server type at error time: {type(_sio_server)}")
                    logger.error("=" * 50)
                    return False
                
                logger.info("📍 STEP 8: Sending welcome message...")
                try:
                    await _sio_server.emit('connected', {
                        "session_id": sid,
                        "p_id": p_id,
                        "user_id": user_id,
                        "worker_id": current_manager._worker_id,
                        "timestamp": get_utc_timestamp_iso()
                    }, room=sid)
                    logger.info("✅ Welcome message sent")
                except Exception as emit_error:
                    logger.warning(f"⚠️ Failed to send welcome: {emit_error}")
                
                logger.info("=" * 50)
                logger.info("🎉 CONNECTION SUCCESSFUL!")
                logger.info(f"✅ User: {user_id}")
                logger.info(f"✅ Session: {sid}")
                logger.info(f"✅ Presentation: {p_id}")
                logger.info("=" * 50)
                
                return True
                
            except Exception as e:
                logger.error("❌ FAILED AT UNKNOWN STEP: Unexpected error")
                logger.error("=" * 50)
                logger.error("🚨 CONNECTION FAILED - UNEXPECTED ERROR")
                logger.error(f"Error details: {e}")
                logger.error(f"Error type: {type(e)}")
                logger.error("=" * 50)
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                return False
        
        # Register connect handler
        try:
            manager_ref.sio.on('connect')(connect_handler)
            logger.info("✅ Connect handler registered")
        except Exception as e:
            logger.error(f"❌ Failed to register connect handler: {e}")
            return
        
        # Register disconnect handler
        @manager_ref.sio.on('disconnect')
        async def disconnect_handler(sid):
            """Handle disconnection"""
            # Get current manager instance
            # from socketio_manager import get_manager
            # current_manager = await get_manager()
            global _manager
            current_manager = _manager
            
            session_info = current_manager.sessions.get(sid)
            if session_info:
                user_id = session_info.get("user_id", "unknown")
                p_id = session_info.get("p_id", "unknown")
                logger.info(f"🔌 Disconnect: user={user_id}, sid={sid}, p_id={p_id}")
            
            await current_manager._unregister_connection(sid)
        
        logger.info("✅ Disconnect handler registered")
        
        # Register ping handler
        @manager_ref.sio.on('ping')
        async def ping_handler(sid, data):
            """Health check"""
            # Use global SocketIO server reference
            global _sio_server
            if _sio_server is None:
                logger.error("❌ Global SocketIO server not available for ping")
                return
            
            await _sio_server.emit('pong', {
                "timestamp": get_utc_timestamp_iso(),
                "client_timestamp": data.get("timestamp") if data else None
            }, room=sid)
        
        logger.info("✅ Ping handler registered")
        
        # Register subscribe_presentation handler
        @manager_ref.sio.on('subscribe_presentation')
        async def subscribe_handler(sid, data):
            """Subscribe to additional presentations"""
            # Use global SocketIO server reference
            global _sio_server
            
            try:
                # Get current manager instance
                # from socketio_manager import get_manager
                # current_manager = await get_manager()
                global _manager
                current_manager = _manager
                
                if _sio_server is None:
                    logger.error("❌ Global SocketIO server not available for subscribe")
                    return
                
                p_id = data.get("p_id")
                if not p_id:
                    await _sio_server.emit('error', {"message": "p_id required"}, room=sid)
                    return
                
                session_info = current_manager.sessions.get(sid)
                if not session_info:
                    await _sio_server.emit('error', {"message": "Session not found"}, room=sid)
                    return
                
                user_id = session_info["user_id"]
                
                # Verify ownership
                from core.database import get_mongo_client
                client = get_mongo_client()
                db = client["slide_creator_db"]
                
                presentation = db.presentations.find_one({"p_id": p_id})
                if not presentation or presentation.get("user_id") != user_id:
                    await _sio_server.emit('error', {
                        "message": "Access denied or presentation not found"
                    }, room=sid)
                    return
                
                # Add to tracking
                if user_id not in current_manager.user_presentations:
                    current_manager.user_presentations[user_id] = set()
                current_manager.user_presentations[user_id].add(p_id)
                
                if p_id not in current_manager.presentation_sessions:
                    current_manager.presentation_sessions[p_id] = set()
                current_manager.presentation_sessions[p_id].add(sid)
                
                # Join room
                _sio_server.enter_room(sid, f"presentation:{p_id}")
                
                await _sio_server.emit('subscribed', {"p_id": p_id}, room=sid)
                logger.info(f"➕ Subscribed: user={user_id}, p_id={p_id}")
                
            except Exception as e:
                logger.error(f"❌ Subscribe error: {e}")
                # Use global SocketIO server reference for error handling
                if _sio_server is not None:
                    await _sio_server.emit('error', {"message": str(e)}, room=sid)
        
        logger.info("✅ Subscribe handler registered")
        logger.info("🎉 All Socket.IO handlers setup complete")
    
    async def initialize(self):
        """Initialize Redis connections and background tasks"""
        try:
            print("\n" + "=" * 80)
            print("🔧 SOCKETIO MANAGER INITIALIZE CALLED")
            print("=" * 80)
            logger.info(f"🔧 Initializing Socket.IO manager...")
            logger.info(f"📋 Socket.IO server status: sio={'SET' if self.sio else 'NOT SET'}")
            
            print(f"   Current sio status: {'SET' if self.sio else 'NOT SET'}")
            print(f"   Worker ID: {self._worker_id[:8]}")
            print(f"   Worker PID: {self._worker_pid}")
            
            # ✅ Note: Socket.IO server will be set via set_sio_server() method AFTER initialize()
            # DO NOT setup handlers here - they will be setup in set_sio_server()
            
            if self.sio is None:
                print("   ⚠️ Socket.IO server not set yet - will be set later")
                logger.info("⚠️ Socket.IO server not set yet - will be set later")
            
            # Initialize Redis connection
            print(f"   Connecting to Redis: {self.redis_url}")
            self.redis_client = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            print("   Testing Redis connection...")
            await self.redis_client.ping()
            print("   ✅ Redis connection OK")
            logger.info(
                f"✅ Redis connected: worker={self._worker_id[:8]}, PID={self._worker_pid}"
            )
            
            # Start background tasks
            print("   Starting background tasks...")
            self._subscriber_task = asyncio.create_task(self._redis_subscriber())
            self._heartbeat_task = asyncio.create_task(self._heartbeat())
            print("   ✅ Background tasks started")
            
            logger.info("✅ Socket.IO manager initialization complete")
            logger.info("⏳ Waiting for Socket.IO server to be set via set_sio_server()...")
            
            print("=" * 80)
            print("✅ SOCKETIO MANAGER INITIALIZE COMPLETE")
            print("=" * 80 + "\n")
            
        except Exception as e:
            print(f"❌ Redis initialization failed: {e}")
            logger.error(f"❌ Redis initialization failed: {e}", exc_info=True)
            import traceback
            traceback.print_exc()
            raise
    
    async def _register_connection(self, user_id: str, session_id: str, p_id: str):
        """
        Register a new connection
        
        ✅ Supports multiple connections from same user
        ✅ Each connection can watch a different p_id
        """
        # Store session info
        self.sessions[session_id] = {
            "user_id": user_id,
            "p_id": p_id,
            "connected_at": get_utc_timestamp_iso()
        }
        
        # Track which presentations this user is watching
        if user_id not in self.user_presentations:
            self.user_presentations[user_id] = set()
        self.user_presentations[user_id].add(p_id)
        
        # Track which sessions are watching this presentation
        if p_id not in self.presentation_sessions:
            self.presentation_sessions[p_id] = set()
        self.presentation_sessions[p_id].add(session_id)
        
        # Register in Redis for cross-worker awareness
        await self._register_in_redis(user_id, session_id, p_id)
    
    async def _register_in_redis(self, user_id: str, session_id: str, p_id: str):
        """Register connection in Redis"""
        try:
            # Store session metadata (TTL: 1 hour)
            session_key = f"sio:session:{session_id}"
            await self.redis_client.setex(
                session_key,
                3600,
                json.dumps({
                    "user_id": user_id,
                    "p_id": p_id,
                    "worker_id": self._worker_id,
                    "connected_at": get_utc_timestamp_iso()
                })
            )
            
            # Add to presentation's active sessions (for monitoring)
            await self.redis_client.sadd(f"sio:presentation:{p_id}:sessions", session_id)
            await self.redis_client.expire(f"sio:presentation:{p_id}:sessions", 3600)
            
        except Exception as e:
            logger.error(f"❌ Redis registration failed: {e}")
    
    async def _unregister_connection(self, session_id: str):
        """Unregister and cleanup a connection"""
        session_info = self.sessions.pop(session_id, None)
        if not session_info:
            return
        
        user_id = session_info["user_id"]
        p_id = session_info["p_id"]
        
        # Remove from presentation sessions
        if p_id in self.presentation_sessions:
            self.presentation_sessions[p_id].discard(session_id)
            if not self.presentation_sessions[p_id]:
                del self.presentation_sessions[p_id]
        
        # Remove from user presentations (only if no other sessions watching this p_id)
        if user_id in self.user_presentations:
            # Check if user has other sessions watching this p_id
            other_sessions_watching = any(
                info["user_id"] == user_id and info["p_id"] == p_id
                for sid, info in self.sessions.items()
                if sid != session_id
            )
            
            if not other_sessions_watching:
                self.user_presentations[user_id].discard(p_id)
            
            # Clean up empty sets
            if not self.user_presentations[user_id]:
                del self.user_presentations[user_id]
        
        # Unregister from Redis
        await self._unregister_from_redis(session_id, p_id)
    
    async def _unregister_from_redis(self, session_id: str, p_id: str):
        """Remove session from Redis"""
        try:
            await self.redis_client.delete(f"sio:session:{session_id}")
            await self.redis_client.srem(f"sio:presentation:{p_id}:sessions", session_id)
        except Exception as e:
            logger.error(f"❌ Redis cleanup failed: {e}")
    
    async def broadcast_to_presentation(self, message: dict, p_id: str, user_id: str):
        """
        Broadcast message to ONLY the owner of this presentation across ALL workers
        
        Args:
            message: The message to send
            p_id: Presentation ID (ensures privacy - only owner receives)
            user_id: User ID who owns the presentation (double privacy check)
        """
        try:
            # Use SPECIFIC channel: presentation + user combination
            channel = f"sio:presentation:{p_id}:user:{user_id}"
            
            # Don't overwrite timestamp if it already exists in message (it's already ISO format)
            message_with_metadata = {
                **message,
                "p_id": p_id,
                "user_id": user_id,
                "worker_id": self._worker_id,
            }
            # Only add timestamp if not already present
            if "timestamp" not in message_with_metadata:
                message_with_metadata["timestamp"] = get_utc_timestamp_iso()
            
            # ✅ Check Redis client is available before publishing
            if self.redis_client is None:
                logger.error("❌ Redis client is None, cannot broadcast")
                # Fallback to local delivery
                await self._send_to_local_sessions(message_with_metadata, p_id, user_id)
                return
            
            # Publish to Redis - all workers listening will forward to matching sessions
            await self.redis_client.publish(
                channel,
                json.dumps(message_with_metadata, default=str)
            )
            
            logger.debug(f"📡 Published: {channel[:50]}... (worker: {self._worker_id[:8]})")
            
        except Exception as e:
            logger.error(f"❌ Broadcast failed: {e}")
            # Fallback: try sending locally
            try:
                await self._send_to_local_sessions(message, p_id, user_id)
            except Exception as fallback_error:
                logger.error(f"❌ Fallback broadcast also failed: {fallback_error}")

    async def _send_to_local_sessions(self, message: dict, p_id: str, user_id: str):
        """Send message to local sessions watching this p_id and owned by user_id"""
        if p_id not in self.presentation_sessions:
            logger.debug(f"🚫 No local sessions watching p_id={p_id}")
            return
        
        # ✅ Check if sio is available
        if self.sio is None:
            logger.error(f"❌ self.sio is None, cannot send to local sessions")
            return
        
        # Store agent output in agent_outputs_2 collection
        # try:
        #     from core.database import get_db
        #     db = get_db()
            
        #     agent_output = {
        #         "p_id": p_id,
        #         "user_id": user_id,
        #         "message": message,
        #         "timestamp": datetime.utcnow(),
        #         "worker_id": self._worker_id,
        #         "message_type": message.get("type", "unknown"),
        #         "event": message.get("event", "unknown")
        #     }
            
        #     db.agent_outputs_2.insert_one(agent_output)
        #     logger.debug(f"💾 Stored agent output: {message.get('type', 'unknown')}")
            
        # except Exception as e:
        #     logger.error(f"❌ Failed to store agent output: {e}")
        
        sent_count = 0
        for session_id in list(self.presentation_sessions[p_id]):
            session_info = self.sessions.get(session_id)
            if not session_info:
                continue
            
            # ✅ Privacy: Only send to sessions belonging to the owner
            if session_info["user_id"] != user_id:
                logger.debug(
                    f"🚫 Skipping session {session_id} - "
                    f"belongs to {session_info['user_id']}, not {user_id}"
                )
                continue
            
            try:
                # ✅ Double-check sio is still available
                if self.sio is None:
                    logger.error(f"❌ self.sio became None while sending!")
                    break
                
                await self.sio.emit('agent_output', message, room=session_id)
                sent_count += 1
                logger.debug(f"📤 Sent to session {session_id}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to send to {session_id}: {e}")
        
        if sent_count > 0:
            logger.debug(f"📤 Sent to {sent_count} local session(s) for p_id={p_id}")
    
    async def _redis_subscriber(self):
        """
        Background task listening to Redis Pub/Sub
        
        ✅ Subscribes to pattern: sio:presentation:*:user:*
        ✅ Forwards messages only to matching local sessions (privacy enforced)
        """
        try:
            self.redis_pubsub = self.redis_client.pubsub()
            
            # Subscribe to all presentation+user channels
            await self.redis_pubsub.psubscribe("sio:presentation:*:user:*")
            
            logger.info(f"👂 Redis subscriber active (worker: {self._worker_id[:8]})")
            
            async for message in self.redis_pubsub.listen():
                if message["type"] == "pmessage":
                    try:
                        channel = message["channel"]
                        data = json.loads(message["data"])
                        
                        # Parse channel: sio:presentation:{p_id}:user:{user_id}
                        parts = channel.split(":")
                        if len(parts) >= 5:
                            p_id = parts[2]
                            user_id = parts[4]
                            
                            # Forward to local sessions (with privacy check)
                            await self._send_to_local_sessions(data, p_id, user_id)
                        
                    except Exception as e:
                        logger.error(f"❌ Message processing error: {e}")
        
        except asyncio.CancelledError:
            logger.info("👂 Subscriber cancelled")
        except Exception as e:
            logger.error(f"❌ Subscriber error: {e}", exc_info=True)
    
    async def _heartbeat(self):
        """Send periodic heartbeat to all connections"""
        try:
            while True:
                await asyncio.sleep(30)
                
                if not self.sessions:
                    continue
                
                heartbeat = {
                    "type": "heartbeat",
                    "worker_id": self._worker_id,
                    "timestamp": get_utc_timestamp_iso()
                }
                
                for session_id in list(self.sessions.keys()):
                    try:
                        if self.sio:
                            await self.sio.emit('message', heartbeat, room=session_id)
                    except Exception as e:
                        logger.debug(f"Heartbeat failed for {session_id}: {e}")
        
        except asyncio.CancelledError:
            logger.info("💔 Heartbeat cancelled")
        except Exception as e:
            logger.error(f"❌ Heartbeat error: {e}")
    
    async def shutdown(self):
        """Clean shutdown"""
        logger.info(f"🔄 Shutting down (worker: {self._worker_id[:8]})")
        
        # Cancel tasks
        for task in [self._subscriber_task, self._heartbeat_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Close Redis
        if self.redis_pubsub:
            await self.redis_pubsub.unsubscribe()
            await self.redis_pubsub.close()
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info(f"✅ Shutdown complete (worker: {self._worker_id[:8]})")
    
    @property
    def active_connections(self):
        """Get count of active connections"""
        return len(self.sessions)
    
    def get_worker_info(self):
        """Get detailed worker information for debugging"""
        return {
            "worker_id": self._worker_id,
            "worker_pid": self._worker_pid,
            "total_connections": len(self.sessions),
            "unique_users": len(self.user_presentations),
            "unique_presentations": len(self.presentation_sessions),
            "details": {
                user_id: list(p_ids) 
                for user_id, p_ids in self.user_presentations.items()
            }
        }

# Global manager instance (one per worker process)
_manager: Optional[SocketIOConnectionManager] = None

# Global SocketIO server reference for handlers
_sio_server: Optional[socketio.AsyncServer] = None

def get_redis_url() -> str:
    """Construct and return the standardized Redis URL"""
    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = os.getenv("REDIS_PORT", "6379")
    redis_password = os.getenv("REDIS_PASSWORD")
    
    # Check if REDIS_URL is explicitly set, otherwise construct it
    redis_url = os.getenv("REDIS_URL")
    
    if not redis_url:
        if redis_password:
            # Use redis:// by default. If SSL is needed, it should be in REDIS_URL prepended with rediss://
            # The port 15384 in logs confirms standard Redis endpoint without mandatory SSL
            redis_url = f"redis://:{redis_password}@{redis_host}:{redis_port}/0"
        else:
            redis_url = f"redis://{redis_host}:{redis_port}/0"
    
    # Ensure Redis URL has proper format
    if not redis_url.startswith(('redis://', 'rediss://', 'unix://')):
        redis_url = f"redis://{redis_url}"
        
    return redis_url

async def get_manager() -> SocketIOConnectionManager:
    """Get or create the Socket.IO manager (singleton per worker)"""
    global _manager
    if _manager is None:
        redis_url = get_redis_url()
        
        # Robust masking for logs: show protocol and host, mask password
        masked_url = redis_url
        if "@" in redis_url:
            parts = redis_url.split("@")
            protocol_auth = parts[0]
            host_port = parts[1]
            if ":" in protocol_auth:
                protocol = protocol_auth.split(":")[0] + "://"
                masked_url = f"{protocol}***@{host_port}"
        
        logger.info(f"🔧 Connecting to Redis (Manager): {masked_url}")
        
        try:
            temp_manager = SocketIOConnectionManager(redis_url)
            await temp_manager.initialize()
            _manager = temp_manager
            logger.info(f"✅ Socket.IO Manager initialized singleton: {id(_manager)}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Socket.IO manager: {e}")
            raise
    return _manager

async def cleanup_manager():
    """Cleanup the manager on shutdown"""
    global _manager
    if _manager:
        await _manager.shutdown()
        _manager = None
