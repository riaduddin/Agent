import os
import json
import httpx
import aio_pika
from typing import Any, Dict, Optional

class FeatureLogger:
    """
    FeatureLogger utility for capturing and dispatching feature usage telemetry.
    Supports both asynchronous (RabbitMQ) and synchronous (API) methods.
    """

    API_URL = os.getenv("FEATURE_USAGE_LOG_API_URL", "https://shothik-payment-engine.com/api/feature-usage-logs")
    API_KEY = os.getenv("SERVER_API_KEY")
    RMQ_QUEUE_LOGS = os.getenv("RMQ_QUEUE_LOGS", "feature_usage_queue")
    RMQ_QUEUE_CREDITS = os.getenv("RMQ_QUEUE_CREDITS", "credits_process_end_multimodel_queue")

    @staticmethod
    async def dispatch_to_queue(amqp_channel: aio_pika.Channel, log_data: Dict[str, Any], queue_name: str = RMQ_QUEUE_LOGS) -> bool:
        """
        Method A: Background Logging/Settlement via RabbitMQ.
        """
        try:
            message = aio_pika.Message(
                body=json.dumps(log_data).encode(),
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                content_type="application/json"
            )
            await amqp_channel.default_exchange.publish(
                message,
                routing_key=queue_name
            )
            return True
        except Exception as e:
            print(f"❌ FeatureLogger: Failed to publish to queue {queue_name}: {e}")
            return False

    @staticmethod
    async def settle_via_queue(amqp_channel: aio_pika.Channel, settlement_data: Dict[str, Any]) -> bool:
        """
        Method A: Credits Process Settlement via RabbitMQ.
        """
        return await FeatureLogger.dispatch_to_queue(amqp_channel, settlement_data, FeatureLogger.RMQ_QUEUE_CREDITS)

    @staticmethod
    async def dispatch_to_api(log_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Method B: Direct API Logging (Synchronous).
        """
        if not FeatureLogger.API_KEY:
            print("❌ FeatureLogger: SERVER_API_KEY not found in environment variables.")
            return None

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    FeatureLogger.API_URL,
                    headers={
                        "Content-Type": "application/json",
                        "x-server-api-key": FeatureLogger.API_KEY,
                    },
                    json=log_data,
                    timeout=10.0
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            print(f"❌ Failed to send log via API: {e}")
            return None
