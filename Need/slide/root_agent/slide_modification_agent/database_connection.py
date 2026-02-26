import logging
import os
from uuid import uuid4
from dotenv import load_dotenv
from pymongo import MongoClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
mongo_uri = os.getenv("MONGODB_URL")

# --- Init Vertex AI & ADK ---


# --- Mongo Setup ---
client = MongoClient(mongo_uri)
db = client["slide_creator_db"]
# agent_outputs = db["agent_outputs_2"]
# presentations = db["presentations"]
# slides = db["slides"]
# slide_status = db["slide_status"]
# slide_html=db["slide_html"]
# references = db["references"]
# system_logs= db["system_logs"]