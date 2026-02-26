import os
import logging
from socketio import AsyncServer, ASGIApp, AsyncRedisManager
from .socketio_manager import get_manager, get_redis_url

logger = logging.getLogger(__name__)

def create_sio_server(cors_origins):
    """
    Creates and configures the Socket.IO server.
    """
    logger.info("🔧 Creating Socket.IO server...")
    
    # Get Redis URL for the manager
    redis_url = get_redis_url()
    client_manager = None
    if redis_url:
        logger.info(f"💾 Using Redis manager for multi-worker support")
        client_manager = AsyncRedisManager(redis_url)
    
    sio = AsyncServer(
        # Standardize Redis manager for cross-process communication
        client_manager=client_manager,
        
        # CORS configuration
        cors_allowed_origins=cors_origins,
        cors_credentials=True,
        
        # Logging configuration
        logger=True,
        engineio_logger=True,
        
        # ASGI mode for FastAPI integration
        async_mode='asgi',
        
        # Transport configuration for Cloud Run
        transports=['polling', 'websocket'],
        allow_upgrades=True,
        
        # Timeout configurations for Cloud Run
        ping_timeout=60,
        ping_interval=25,
        
        # Connection settings
        max_http_buffer_size=100000000,
        
        # Compression settings
        compression_threshold=None,
        
        # HTTP settings
        http_compression=False,  # Disable HTTP compression (equivalent to perMessageDeflate: false)
    )
    
    return sio

async def init_socketio(app, cors_origins, api_prefix):
    """
    Initializes Socket.IO and mounts it to the FastAPI app.
    """
    sio = create_sio_server(cors_origins)
    
    # Initialize manager and set server
    manager = await get_manager()
    manager.set_sio_server(sio)
    
    # Mount Socket.IO to FastAPI
    # CRITICAL: When mounting, FastAPI strips the mount path
    # So Socket.IO receives requests at / (root). We must set socketio_path='' to match this.
    socketio_mount_path = f"{api_prefix}/socket.io"
    sio_app = ASGIApp(sio, socketio_path='')
    app.mount(socketio_mount_path, sio_app)
    
    logger.info(f"✅ Socket.IO initialization and mounting complete at {socketio_mount_path}")
    return sio, manager
