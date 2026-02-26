import asyncio
import os
import json
import aio_pika
from dotenv import load_dotenv
from utils.feature_logger import FeatureLogger

# Load environment variables
load_dotenv()

async def test_log_publishing():
    rabbitmq_url = os.getenv("RABBITMQ_URL")
    if not rabbitmq_url:
        print("❌ RABBITMQ_URL not found in environment variables.")
        return

    print(f"Connecting to RabbitMQ at: {rabbitmq_url}")

    # Log payload according to schema
    log_payload = {
        "feature_endpoint_value": "agents-presentation-generation",
        "user_id": "688087e305194976aea99403",
        "user_email": "rrriaduddin@gmail.com",
        "usage_key": "ac36bd2e-549e-4104-82e2-232380cf72d7",
        "method": "POST",
        "query": {},
        "params": {},
        "payload": {"prompt": "Tell me a joke"},
        "response": {"result": "Why did the chicken cross the road?"},
        "code": 200,
        "status": "success"
    }

    try:
        connection = await aio_pika.connect_robust(rabbitmq_url)
        async with connection:
            channel = await connection.channel()
            queue_name = FeatureLogger.RMQ_QUEUE
            
            # Ensure queue exists (using robust logic from test_rabbit.py)
            try:
                await channel.declare_queue(queue_name, passive=True)
                print(f"✅ Found existing queue: {queue_name}")
            except aio_pika.exceptions.ChannelClosed:
                print(f"⚠️ Queue not found, creating new one: {queue_name}")
                channel = await connection.channel()
                await channel.declare_queue(queue_name, durable=True)
                print(f"✅ Created new queue: {queue_name}")
            except Exception as e:
                print(f"⚠️ Exception during declare: {e}")
                if channel.is_closed:
                    channel = await connection.channel()
                await channel.declare_queue(queue_name, durable=True)

            # Use the FeatureLogger helper
            success = await FeatureLogger.dispatch_to_queue(channel, log_payload)
            
            if success:
                print(f"✅ Successfully published test log to queue: {queue_name}")
                print(f"Payload: {json.dumps(log_payload, indent=2)}")
            else:
                print(f"❌ Failed to publish log using FeatureLogger.")
            
    except Exception as e:
        print(f"❌ Failed to test RabbitMQ logging: {e}")

if __name__ == "__main__":
    asyncio.run(test_log_publishing())
