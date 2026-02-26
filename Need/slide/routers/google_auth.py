import asyncio
import re
import json
import os
import logging
import urllib.parse
from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import RedirectResponse
import requests
from middleware.auth import get_current_user, AuthenticatedUser

router = APIRouter(tags=["google-auth"])
logger = logging.getLogger(__name__)

# Constants removed from module level to ensure they are read after dotenv loads

# Scopes needed for Drive upload
# Using full drive scope to ensure conversion permissions
SCOPES = "https://www.googleapis.com/auth/drive"

@router.get("/google/ping")
async def google_ping():
    return {"status": "ok", "router": "google_auth"}

@router.get("/google/login")
async def google_login(return_url: str = None):
    """
    Redirects the user to Google's OAuth 2.0 consent screen.
    """
    client_id = os.getenv("GOOGLE_CLIENT_ID")
    redirect_uri = os.getenv("GOOGLE_REDIRECT_URI")
    
    if not client_id or not redirect_uri:
        raise HTTPException(status_code=500, detail=f"Google OAuth configuration missing")

    # Use state to pass the return URL back to ourselves
    state = return_url or ""
    
    auth_url = (
        f"https://accounts.google.com/o/oauth2/v2/auth?"
        f"client_id={client_id}&"
        f"redirect_uri={urllib.parse.quote(redirect_uri)}&"
        f"response_type=code&"
        f"scope={SCOPES}&"
        f"access_type=offline&"
        f"state={urllib.parse.quote(state)}&"
        f"prompt=consent"
    )
    return RedirectResponse(url=auth_url)

@router.get("/google/callback")
async def google_callback(code: str, state: str = None, error: str = None):
    """
    Handles the redirect from Google. Exchanges the code for an access token.
    """
    if error:
        logger.error(f"Google OAuth error: {error}")
        raise HTTPException(status_code=400, detail=f"Google OAuth error: {error}")

    if not code:
        raise HTTPException(status_code=400, detail="Missing authorization code")

    client_id = os.getenv("GOOGLE_CLIENT_ID")
    client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
    redirect_uri = os.getenv("GOOGLE_REDIRECT_URI")

    # Exchange code for tokens
    token_url = "https://oauth2.googleapis.com/token"
    data = {
        "code": code,
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": redirect_uri,
        "grant_type": "authorization_code",
    }

    try:
        response = requests.post(token_url, data=data)
        response.raise_for_status()
        tokens = response.json()
        access_token = tokens['access_token']
        
        # Determine redirect URL
        # 1. Check if state (return_url) was provided
        if state:
            redirect_base = state
        else:
            # 2. Fallback logic: check GOOGLE_AUTH_REDIRECT_BASE or CORS_ORIGINS
            # Prioritize the explicit environment variable for the redirect base
            env_redirect_base = os.getenv("GOOGLE_AUTH_REDIRECT_BASE")
            if env_redirect_base:
                redirect_base = env_redirect_base
            else:
                # Fallback to existing CORS logic if the specific env var isn't set
                cors_origins = os.getenv("CORS_ORIGINS", "").split(",")
                redirect_base = "http://localhost:3000" # Default
                for origin in cors_origins:
                    origin = origin.strip()
                    if "localhost:3000" in origin:
                        redirect_base = "http://localhost:3000"
                        break
                    elif "shothik.ai" in origin and "shothik" not in redirect_base:
                        redirect_base = origin
        
        # Ensure redirect_base doesn't end with slash before adding path
        redirect_base = redirect_base.rstrip("/")
        
        # We append a path that the frontend should handle to process the token
        final_redirect = f"{redirect_base}/export-success#access_token={access_token}"
        
        logger.info(f"✅ Google Auth successful. Redirecting to: {final_redirect}")
        return RedirectResponse(url=final_redirect)

    except Exception as e:
        logger.error(f"Failed to exchange Google code: {e}")
        raise HTTPException(status_code=500, detail="Failed to obtain Google tokens")
