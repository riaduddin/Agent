# # backend/run.py
# from app import create_app

# app = create_app()

# if __name__ == '__main__':
#     # Runs the development server on http://localhost:80
#     # Set debug=True for development, but False for production
#     app.run(host='0.0.0.0', port=5000, debug=True)


# backend/run.py
import os
from app import create_app
import atexit
import sys
import signal
import traceback
import logging
import contextlib
import faulthandler
from datetime import datetime
from threading import Thread
from app.services.cloudscheduler_batch_process_trigger import subscribe_to_scheduler_services
from app.utils.logging_utils import setup_cloud_logging

# Enable fault handler for segmentation faults
faulthandler.enable(file=sys.stderr, all_threads=True)


from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
port = int(os.getenv('PORT', 8080))

# Configure comprehensive logging using shared utility (JSON/Structured)
# This ensures Cloud Run sees correct severity levels.
logger = setup_cloud_logging(root_level=logging.INFO)

# Global variables to track exit information
exit_reason = "Unknown"
exit_traceback = None
exit_timestamp = None

def print_exit_banner(reason, tb=None):
    """Print a prominent exit banner"""
    banner = "=" * 60
    print(f"\n{banner}")
    print("*** APPLICATION IS EXITING ***")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Reason: {reason}")
    if tb:
        print(f"Traceback:\n{tb}")
    print(banner)

def exit_handler():
    """Called when the application is about to exit"""
    global exit_reason, exit_traceback, exit_timestamp
    
    exit_timestamp = datetime.now()
    
    # Log the exit
    logger.critical("=" * 50)
    logger.critical("APPLICATION EXIT HANDLER CALLED")
    logger.critical(f"Exit Reason: {exit_reason}")
    logger.critical(f"Exit Time: {exit_timestamp}")
    if exit_traceback:
        logger.critical(f"Traceback:\n{exit_traceback}")
    logger.critical("=" * 50)
    
    # Print to console
    print_exit_banner(exit_reason, exit_traceback)

# Register the exit handler
atexit.register(exit_handler)

def signal_handler(signum, frame):
    """Handle system signals"""
    global exit_reason, exit_traceback
    
    signal_names = {
        signal.SIGINT: "SIGINT (Ctrl+C - User Interruption)",
        signal.SIGTERM: "SIGTERM (Termination Signal)",
    }
    
    if hasattr(signal, 'SIGBREAK'):
        signal_names[signal.SIGBREAK] = "SIGBREAK (Ctrl+Break)"
    
    exit_reason = f"Signal received: {signal_names.get(signum, f'Unknown Signal {signum}')}"
    exit_traceback = ''.join(traceback.format_stack(frame))
    
    logger.critical(f"Signal {signum} received, initiating shutdown...")
    print(f"\n*** SIGNAL {signum} RECEIVED - SHUTTING DOWN ***")
    
    # Exit gracefully
    sys.exit(1)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
if hasattr(signal, 'SIGBREAK'):
    signal.signal(signal.SIGBREAK, signal_handler)

def custom_exception_hook(exc_type, exc_value, exc_traceback):
    """Custom exception handler for uncaught exceptions"""
    global exit_reason, exit_traceback
    
    if issubclass(exc_type, KeyboardInterrupt):
        exit_reason = "KeyboardInterrupt (User pressed Ctrl+C)"
        exit_traceback = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        logger.critical("Application interrupted by user (Ctrl+C)")
    else:
        exit_reason = f"Uncaught Exception: {exc_type.__name__}: {exc_value}"
        exit_traceback = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        
        logger.critical("UNCAUGHT EXCEPTION DETECTED!")
        logger.critical(exit_reason)
        logger.critical(f"Traceback:\n{exit_traceback}")
        
        print_exit_banner(exit_reason, exit_traceback)
    
    # Call the original exception hook
    sys.__excepthook__(exc_type, exc_value, exc_traceback)

# Set the custom exception hook
sys.excepthook = custom_exception_hook

@contextlib.contextmanager
def app_lifecycle():
    """Context manager to track application lifecycle"""
    global exit_reason, exit_traceback
    
    try:
        print("*** APPLICATION STARTING ***")
        logger.info("Application lifecycle started")
        exit_reason = "Application started successfully"
        yield
        
    except KeyboardInterrupt:
        exit_reason = "KeyboardInterrupt in main execution"
        exit_traceback = traceback.format_exc()
        logger.critical("Application interrupted by user in main execution")
        
    except SystemExit as e:
        exit_reason = f"SystemExit called (code: {e.code})"
        exit_traceback = traceback.format_exc()
        logger.critical(f"System exit called with code: {e.code}")
        
    except Exception as e:
        exit_reason = f"Unhandled Exception in main: {type(e).__name__}: {str(e)}"
        exit_traceback = traceback.format_exc()
        logger.critical(f"Unhandled exception in main: {str(e)}")
        logger.critical(f"Traceback:\n{exit_traceback}")
        
    finally:
        logger.info("Application lifecycle context manager exiting")

def validate_environment():
    """Validate the environment before starting"""
    logger.info("Validating environment...")
    
    # Check Python version
    logger.info(f"Python version: {sys.version}")
    
    # Check if we can import required modules
    try:
        import flask
        logger.info(f"Flask version: {flask.__version__}")
    except ImportError as e:
        logger.error(f"Flask import error: {e}")
        raise
    
    logger.info("Environment validation complete")

def main():
    global exit_reason, exit_traceback
    
    try:
        logger.info("Starting main function...")
        
        # Validate environment
        validate_environment()
        
        # Use lifecycle context manager
        with app_lifecycle():
            logger.info("Creating Flask application...")
            app = create_app()
            logger.info("Flask app created successfully")
            
            # Start the Unified Scheduler services (Batch & Legacy)
            scheduler_thread = Thread(target=subscribe_to_scheduler_services, daemon=True)
            scheduler_thread.start()
            logger.info("Unified Scheduler Services thread started.")

            # OLD: Start the Legacy Scheduler subscriber (Deprecated/Merged)
            # from app.services.scheduler_trigger_service import subscribe_to_scheduler_topic
            # scheduler_thread = Thread(target=subscribe_to_scheduler_topic, daemon=True)
            # scheduler_thread.start()
            # logger.info("Legacy Reprocess Scheduler thread started.")

            exit_reason = "Flask app created, starting server..."
            
            logger.info(f"Starting Flask development server on port {port}...")
            # Run the app on port defined by env or default
            app.run(host='0.0.0.0', port=port, debug=True, use_reloader=True)
            
            exit_reason = "Flask server stopped normally"
            
    except Exception as e:
        exit_reason = f"Exception in main function: {type(e).__name__}: {str(e)}"
        exit_traceback = traceback.format_exc()
        logger.critical(f"Exception in main function: {str(e)}")
        logger.critical(f"Traceback:\n{exit_traceback}")
        print_exit_banner(exit_reason, exit_traceback)
        raise
    
    finally:
        logger.info("Main function completed")

if __name__ == '__main__':
    main()
