#!/usr/bin/env python3
"""
Client test for Socket.IO presentation generation.
"""

import socketio
import asyncio
import requests
import time

# Configuration
SERVER_URL = "http://127.0.0.1:8060"
TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJfaWQiOiI2ODgwODdlMzA1MTk0OTc2YWVhOTk0MDMiLCJzdWIiOiI2ODgwODdlMzA1MTk0OTc2YWVhOTk0MDMiLCJlbWFpbCI6InJycmlhZHVkZGluQGdtYWlsLmNvbSIsInBhY2thZ2UiOiJ1bmxpbWl0ZWQiLCJpc192ZXJpZmllZCI6dHJ1ZSwicm9sZSI6InVzZXIiLCJpYXQiOjE3NjEzNzczMzAsImV4cCI6MTc2MTQ2MzczMH0.dUeYJu4jNbaSfN8jRloiFiRTie1WkJ1prWtMe-_RQLg"
MESSAGE = "Create a presentation about AI in Healthcare"

# Socket.IO client
sio = socketio.AsyncClient()

@sio.event
async def connect():
    print("✅ Connected to Socket.IO")

@sio.event
async def disconnect():
    print("❌ Disconnected from Socket.IO")

@sio.event
async def agent_output(data):
    print(f"📨 Agent Output: {data}")

@sio.event
async def message(data):
    print(f"📨 Message: {data}")

async def create_presentation():
    response = requests.post(f"{SERVER_URL}/create-presentation", 
                           json={"message": MESSAGE, "file_urls": []},
                           headers={"Authorization": f"Bearer {TOKEN}"})
    return response.json()["p_id"] if response.status_code == 200 else None

async def start_presentation(p_id):
    response = requests.post(f"{SERVER_URL}/start-presentation/{p_id}",
                           headers={"Authorization": f"Bearer {TOKEN}"})
    return response.status_code == 200

async def main():
    print("🧪 Client Test Starting...")
    
    # Create presentation
    p_id = await create_presentation()
    print(f"📝 Created presentation: {p_id}")
    
    # Connect Socket.IO
    await sio.connect(f"{SERVER_URL}?p_id={p_id}&token={TOKEN}")
    print("🔌 Connected to Socket.IO")
    
    # Start presentation
    await start_presentation(p_id)
    print("🚀 Started presentation")
    
    # Wait for events
    await asyncio.sleep(30)
    
    # Disconnect
    await sio.disconnect()
    print("✅ Test completed")

if __name__ == "__main__":
    asyncio.run(main())
