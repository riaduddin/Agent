#!/usr/bin/env python3
"""
Test server to verify Socket.IO works without translate_request errors.
"""

import uvicorn
from main import app

if __name__ == "__main__":
    print("Starting Test Server")
    print("=" * 50)
    print("Server will be available at: http://127.0.0.1:8060")
    print("Socket.IO will be available at: http://127.0.0.1:8060/socket.io/")
    print("Press Ctrl+C to stop")
    print("=" * 50)
    
    try:
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=8060,
            log_level="info",
            access_log=True
        )
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"\nServer error: {e}")
