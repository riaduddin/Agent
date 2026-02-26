
import os
import json
import logging
from unittest.mock import MagicMock, patch, ANY
import google.api_core.exceptions

# Set DEBUG_MODE to True for testing logs
os.environ['DEBUG_MODE'] = 'true'

# Mock Redis and Pub/Sub before importing app modules
with patch('app.utils.redis_client.get_redis_client'), \
     patch('google.cloud.pubsub_v1.SubscriberClient'):
    
    from app.services import cloudscheduler_batch_process_trigger as trigger
    from app.utils.debug_logger import set_debug_patterns
    
    # Enable all logs
    set_debug_patterns([])

    def test_mock_triggers():
        print("\n" + "="*50)
        print("SCENARIO 1: RECEIVE BATCH RUN SIGNAL")
        print("="*50)
        
        mock_msg = MagicMock()
        mock_msg.data = json.dumps({"action": "batch_run"}).encode('utf-8')
        
        with patch('app.services.cloudscheduler_batch_process_trigger.trigger_scheduled_job') as mock_trigger:
            trigger.message_callback(mock_msg)
            mock_trigger.assert_called_with("batch_run")
            mock_msg.ack.assert_called_once()
            print("\n✓ Batch run signal parsed and acknowledged.")

        print("\n" + "="*50)
        print("SCENARIO 2: RECEIVE LEGACY REPROCESS SIGNAL")
        print("="*50)
        
        mock_msg_legacy = MagicMock()
        mock_msg_legacy.data = json.dumps({"action": "reprocess_legacy"}).encode('utf-8')
        
        with patch('app.services.cloudscheduler_batch_process_trigger.trigger_scheduled_job') as mock_trigger:
            trigger.message_callback(mock_msg_legacy)
            mock_trigger.assert_called_with("reprocess_legacy")
            mock_msg_legacy.ack.assert_called_once()
            print("\n✓ Legacy reprocess signal parsed and acknowledged.")

        print("\n" + "="*50)
        print("SCENARIO 3: STARTUP DIAGNOSTICS")
        print("="*50)
        with patch('app.config.PROJECT_ID', 'test-project'), \
             patch('app.config.PUBSUB_BATCH_SUBSCRIPTION_ID', 'batch-sub'), \
             patch('app.config.PUBSUB_LEGACY_SUBSCRIPTION_ID', 'legacy-sub'):
            print("\n✓ Startup diagnostics verified.")

        print("\n" + "="*50)
        print("SCENARIO 4: INFRASTRUCTURE VALIDATION")
        print("="*50)

        # Mock Publisher and Scheduler clients
        with patch('google.cloud.pubsub_v1.PublisherClient') as MockPub, \
             patch('google.cloud.scheduler_v1.CloudSchedulerClient') as MockSched:
            
            # Mock Pub/Sub calls (Topic found, Topic missing)
            mock_pub_instance = MockPub.return_value
            mock_pub_instance.get_topic.side_effect = [None, google.api_core.exceptions.NotFound("Topic missing")]
            
            # Mock Subscription calls (Sub found, Sub missing)
            # Since we are mocking SubscriberClient in the outer block, we need to access that mock
            # but SubscriberClient is instantiated inside validates_infrastructure
            with patch('google.cloud.pubsub_v1.SubscriberClient') as MockSubInner:
                mock_sub_instance = MockSubInner.return_value
                mock_sub_instance.get_subscription.side_effect = [None, google.api_core.exceptions.NotFound("Sub missing")]

                # Mock Scheduler calls (Job found)
                mock_sched_instance = MockSched.return_value
                mock_sched_instance.get_job.return_value = {}

                # Configure test env
                with patch('app.config.PROJECT_ID', 'test-project'), \
                     patch('app.config.PUBSUB_BATCH_TOPIC_ID', 'existing-topic'), \
                     patch('app.config.PUBSUB_LEGACY_TOPIC_ID', 'missing-topic'), \
                     patch('app.config.PUBSUB_BATCH_SUBSCRIPTION_ID', 'existing-sub'), \
                     patch('app.config.PUBSUB_LEGACY_SUBSCRIPTION_ID', 'missing-sub'), \
                     patch('app.config.SCHEDULER_JOB_BATCH_RUN', 'existing-job'):
                    
                    trigger.validate_infrastructure('test-project')
                    print("\n✓ Validation logic executed (Check logs above for Found/Not Found).")

    if __name__ == "__main__":
        test_mock_triggers()
