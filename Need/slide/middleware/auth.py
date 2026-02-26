"""
JWT Authentication Middleware for FastAPI
Handles JWT token verification and user authentication
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, Dict, Any
import jwt
import os
from bson import ObjectId
from dotenv import load_dotenv
import logging

load_dotenv()
logger = logging.getLogger(__name__)

# JWT Configuration
JWT_SECRET = os.getenv("JWT_SECRET")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")  # Default to HS256

# Aliases for compatibility (used by WebSocket module)
SECRET_KEY = JWT_SECRET
ALGORITHM = JWT_ALGORITHM

# Security scheme for Swagger UI
security = HTTPBearer()


class AuthenticatedUser:
    """Class to hold authenticated user information"""
    def __init__(self, user_id: str, email: str, package: str, is_verified: bool, role: str, **kwargs):
        self.user_id = user_id
        self.email = email
        self.package = package
        self.is_verified = is_verified
        self.role = role
        self.extra_data = kwargs


def decode_jwt_token(token: str) -> Dict[str, Any]:
    """
    Decode and verify JWT token
    
    Args:
        token: JWT token string
        
    Returns:
        Decoded token payload
        
    Raises:
        HTTPException: If token is invalid or verification fails
    """
    if not JWT_SECRET:
        logger.error("JWT_SECRET is not configured in environment variables")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication configuration error"
        )
    
    try:
        # Decode the JWT token
        payload = jwt.decode(
            token,
            JWT_SECRET,
            algorithms=[JWT_ALGORITHM]
        )
        return payload
    
    except jwt.ExpiredSignatureError:
        logger.warning("JWT token has expired")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid JWT token: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    except Exception as e:
        logger.error(f"Error decoding JWT token: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
    ) -> AuthenticatedUser:
    """
    Dependency to get the current authenticated user from JWT token
    
    This function:
    1. Extracts the Bearer token from Authorization header
    2. Decodes and verifies the JWT token
    3. Validates that the user is verified
    4. Returns user information
    
    Args:
        credentials: HTTP Authorization credentials
        
    Returns:
        AuthenticatedUser: Authenticated user information
        
    Raises:
        HTTPException: If authentication fails or user is not verified
    """
    token = credentials.credentials
    
    # Decode the token
    payload = decode_jwt_token(token)
    
    # Extract user information from token
    # The token contains: _id, sub, email, package, is_verified, role, iat
    user_id = payload.get("_id") or payload.get("sub")
    email = payload.get("email")
    package = payload.get("package")
    is_verified = payload.get("is_verified", False)
    role = payload.get("role", "user")
    
    # Validate required fields
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token: missing user ID"
        )
    
    if not email:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token: missing email"
        )
    
    # Check if user is verified
    if not is_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User is not verified. Please verify your account."
        )
    
    # Auditor's Recommendation: Database check for user status (active/banned)
    try:
        from core.database import get_db, get_user_by_id
        db_handle = get_db()
        user_doc = get_user_by_id(db_handle, user_id)
        
        # If user document exists, verify they are active
        if user_doc:
            if user_doc.get("status") == "banned":
                logger.warning(f"Banned user attempted access: {user_id}")
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Your account has been suspended."
                )
            if not user_doc.get("is_active", True):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Your account is inactive."
                )
    except HTTPException: raise
    except Exception as e:
        # Fallback: if DB is down, we might still allow access if token is valid, 
        # but the auditor wants a "fast DB check". We'll log the error.
        logger.error(f"Error during user status check: {e}")
    
    # Log successful authentication
    logger.info(f"User authenticated: {user_id} ({email})")
    
    
    # Create and return authenticated user object
    return AuthenticatedUser(
        user_id=user_id,
        email=email,
        package=package,
        is_verified=is_verified,
        role=role,
        iat=payload.get("iat")
    )


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False))
    ) -> Optional[AuthenticatedUser]:
    """
    Optional authentication dependency
    Returns user if authenticated, None otherwise
    Useful for endpoints that work with or without authentication
    
    Args:
        credentials: Optional HTTP Authorization credentials
        
    Returns:
        AuthenticatedUser or None
    """
    if not credentials:
        return None
    
    try:
        return await get_current_user(credentials)
    except HTTPException:
        return None

