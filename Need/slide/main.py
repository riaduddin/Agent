import os
from dotenv import load_dotenv
load_dotenv()
import asyncio
import logging
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
try:
    from opentelemetry import trace, context as otel_ctx
    HAS_OTEL = True
except ImportError:
    HAS_OTEL = False
    logger = logging.getLogger(__name__)
    logger.warning("⚠️ opentelemetry not found, tracing will be disabled")
from dotenv import load_dotenv

# Import version
from __version__ import __version__, RELEASE_NAME

# Import core modules
from core.database import get_db
from core.socketio_manager import cleanup_manager
from core.socketio_server import init_socketio

# Import routers
from routers.socketio import router as socketio_router
from routers.templates import router as template_router
from routers.presentations import router as presentation_router
from routers.files import router as file_router
from routers.conversion import router as conversion_router
from routers.google_auth import router as google_auth_router

# load_dotenv() # Moved to top

# Logging setup
logger = logging.getLogger(__name__)
logger.info(f"🚀 Starting Presentation Gen Service v{__version__} ({RELEASE_NAME})")

# Basic Configuration
api_prefix = os.getenv("API_PREFIX", "/api")
cors_origins_str = os.getenv("CORS_ORIGINS", "https://www.shothik.ai")
cors_origins = [origin.strip() for origin in cors_origins_str.split(",") if origin.strip()]

logger.info(f"⚙️ Configuration: API_PREFIX={api_prefix}, CORS_ORIGINS={cors_origins}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("=" * 40 + " 🚀 STARTING LIFESPAN " + "=" * 40)
    
    # 1. Initialize database connection
    try:
        from core.database import get_mongo_client, check_db_health
        db_client = get_mongo_client()
        db = db_client["slide_creator_db"]
        if check_db_health(db):
            logger.info("✅ Database connection healthy")
    except Exception as e:
        logger.error(f"❌ Database initialization error: {e}")
    
    # 2. Initialize Qdrant connection
    try:
        from tools.qdrant_utils import get_qdrant_manager
        qdrant = get_qdrant_manager()
        is_healthy, msg = qdrant.health_check()
        if is_healthy:
            logger.info(f"✅ Qdrant: {msg}")
    except Exception as e:
        logger.error(f"❌ Qdrant initialization error: {e}")
    
    # 3. Log all routes for debugging
    for route in app.routes:
        logger.info(f"📍 Registered Route: {route.path}")

    # 4. Initialize Socket.IO
    # Note: This also mounts SIO to the app
    # We wrap this in try-except to prevent lifespan crash if Redis is unavailable
    try:
        await init_socketio(app, cors_origins, api_prefix)
    except Exception as e:
        logger.error(f"❌ Socket.IO initialization failed: {e}")
        logger.warning("⚠️ Socket.IO functionality will be unavailable, but REST API will continue to start.")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down...")
    await cleanup_manager()
    try:
        from core.database import _MONGO_CLIENT
        if _MONGO_CLIENT:
            _MONGO_CLIENT.close()
            logger.info("✅ MongoDB connections closed")
    except: pass

app = FastAPI(
    title="Presentation Generation Service",
    version=__version__,
    lifespan=lifespan
)

# CORS Configuration
# Custom CORS middleware that excludes Socket.IO paths
# Socket.IO handles its own CORS to prevent duplicate headers
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

class ConditionalCORSMiddleware(BaseHTTPMiddleware):
    """
    Custom CORS middleware that excludes Socket.IO paths.
    Socket.IO handles its own CORS to prevent duplicate headers.
    """
    async def dispatch(self, request, call_next):
        # Check if this is a Socket.IO request
        # Socket.IO is mounted at {api_prefix}/socket.io
        if request.url.path.startswith(f"{api_prefix}/socket.io"):
            # Skip CORS for Socket.IO - it handles its own CORS
            return await call_next(request)
        
        # For all other requests, apply CORS
        origin = request.headers.get("origin")
        allow_origin = None
        if origin in cors_origins or "*" in cors_origins:
            allow_origin = origin if origin in cors_origins else (cors_origins[0] if cors_origins[0] != "*" else origin)
        
        # Handle preflight requests
        if request.method == "OPTIONS":
            if allow_origin:
                return Response(
                    headers={
                        "Access-Control-Allow-Origin": allow_origin,
                        "Access-Control-Allow-Methods": "*",
                        "Access-Control-Allow-Headers": "*",
                        "Access-Control-Allow-Credentials": "true",
                        "Access-Control-Max-Age": "600",
                    }
                )
            logger.warning(f"🚫 CORS: Origin {origin} not allowed. Allowed: {cors_origins}")
            return Response(status_code=403)
        
        # Handle actual requests
        try:
            response = await call_next(request)
        except Exception as e:
            logger.error(f"❌ Middleware caught exception: {e}", exc_info=True)
            # Create a 500 response if call_next fails
            response = Response(content=f"Internal Server Error: {str(e)}", status_code=500)
        
        # Add CORS headers to response ALWAYS if origin matches
        if allow_origin:
            response.headers["Access-Control-Allow-Origin"] = allow_origin
            response.headers["Access-Control-Allow-Credentials"] = "true"
            response.headers["Access-Control-Allow-Methods"] = "*"
            response.headers["Access-Control-Allow-Headers"] = "*"
        elif origin:
            logger.debug(f"ℹ️ CORS: Origin {origin} not in allowed list {cors_origins}")
        
        return response

# Add custom CORS middleware
app.add_middleware(ConditionalCORSMiddleware)

# Route Registration
api_router = APIRouter()
api_router.include_router(socketio_router)
api_router.include_router(template_router)
api_router.include_router(presentation_router)
api_router.include_router(file_router)
api_router.include_router(conversion_router)
api_router.include_router(google_auth_router) # Removed from here to handle explicitly below

# Health check endpoints
@app.get(f"{api_prefix}/debug-routes")
async def debug_routes():
    return {"registered_routes": [route.path for route in app.routes]}

@app.get(f"{api_prefix}/")
async def root():
    return {
        "status": "ok", 
        "service": "presentation-gen-service",
        "version": __version__
    }

@app.get(f"{api_prefix}/health")
async def health():
    """Comprehensive health check with version information"""
    return {
        "status": "healthy",
        "version": __version__,
        "release": RELEASE_NAME,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service": "presentation-gen-service"
    }

app.include_router(api_router, prefix=api_prefix)

# Explicitly register routers at multiple possible prefixes for maximum compatibility
app.include_router(google_auth_router, prefix="/api")
app.include_router(google_auth_router, prefix="/presentations")
app.include_router(google_auth_router, prefix="/presentations/api")
app.include_router(google_auth_router) # Root level registration (/google/...)

app.include_router(conversion_router, prefix="/api")
app.include_router(conversion_router, prefix="/presentations")
app.include_router(conversion_router, prefix="/presentations/api")
app.include_router(conversion_router) # Root level registration (/convert, /job/...)

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host=host, port=port)