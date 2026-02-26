# backend/app/__init__.py
import os
from flask import Flask
from flask_jwt_extended import JWTManager
from google.cloud import firestore
from google.oauth2 import service_account
import google.auth.exceptions
from flask_cors import CORS
from . import config # Import the new config module
from .utils.saml_config_generator import generate_saml_settings_json # Import the generator
from .utils.debug_logger import debug_log, debug_warn, debug_error
from .utils.logging_utils import setup_cloud_logging
import logging


# Initialize Firestore client using config values
db = None
credentials = None

def init_firestore():
    global db, credentials
    # Configure logging using shared utility
    # This ensures consistency if the app is initialized without run.py
    setup_cloud_logging(root_level=logging.INFO)

    try:
        if config.GOOGLE_APPLICATION_CREDENTIALS:
            try:
                credentials = service_account.Credentials.from_service_account_file(config.GOOGLE_APPLICATION_CREDENTIALS)
                # Pass the database ID from config
                db = firestore.Client(
                    project=config.PROJECT_ID,
                    credentials=credentials,
                    database=config.FIRESTORE_DATABASE_ID
                )
                debug_log(f"Firestore client initialized using service account: {config.GOOGLE_APPLICATION_CREDENTIALS} for database: {config.FIRESTORE_DATABASE_ID}")
            except FileNotFoundError:
                debug_warn(f"Service account key file not found at {config.GOOGLE_APPLICATION_CREDENTIALS}")
            except Exception as e:
                debug_warn(f"Failed to initialize Firestore client using service account: {e}")
        else:
            # Fallback to Application Default Credentials if GOOGLE_APPLICATION_CREDENTIALS is not set in config
            debug_warn("GOOGLE_APPLICATION_CREDENTIALS not set in config. Falling back to Application Default Credentials.")
            try:
                # Project ID and Database ID should be available from config here
                db = firestore.Client(
                    project=config.PROJECT_ID,
                    database=config.FIRESTORE_DATABASE_ID
                )
                debug_log(f"Firestore client initialized using Application Default Credentials for project {config.PROJECT_ID}, database: {config.FIRESTORE_DATABASE_ID}.")
            except google.auth.exceptions.DefaultCredentialsError as e:
                debug_warn(f"DefaultCredentialsError - Could not find default credentials. {e}")
            except Exception as e:
                debug_warn(f"Failed to initialize Firestore client using ADC: {e}")
    except Exception as e:
        debug_error(f"CRITICAL: Unexpected error during Firestore initialization: {e}")

# Run initialization safely
init_firestore()

if db is None:
     debug_warn("Firestore client failed to initialize. Routes requiring database access will fail.")

# Initialize Redis Client (ensure connection is attempted at startup)
from .utils.redis_client import get_redis_client
try:
    get_redis_client()
except Exception as e:
    # Log the error but allow the app to start, routes might fail later
    debug_warn(f"Failed to connect to Redis during app startup: {e}")

# Initialize JWT Manager
jwt = JWTManager()

def create_app():
    """Flask application factory."""
    app = Flask(__name__)

    # Configuration from config module
    app.config['SECRET_KEY'] = config.SECRET_KEY
    app.config['JWT_SECRET_KEY'] = config.JWT_SECRET_KEY
    app.config['JWT_ACCESS_TOKEN_EXPIRES'] = config.JWT_ACCESS_TOKEN_EXPIRES
    # Default location is headers, so explicitly setting is optional
    # app.config['JWT_TOKEN_LOCATION'] = ['headers']
    # Remove cookie settings
    # app.config['JWT_COOKIE_SECURE'] = False
    # app.config['JWT_COOKIE_SAMESITE'] = 'Lax'
    # app.config['JWT_COOKIE_PATH'] = '/'
    # CSRF Protection (optional but recommended)
    # app.config['JWT_COOKIE_CSRF_PROTECT'] = True # Requires sending CSRF token in header
    # app.config['JWT_ACCESS_CSRF_HEADER_NAME'] = "X-CSRF-TOKEN-ACCESS"
    # app.config['JWT_REFRESH_CSRF_HEADER_NAME'] = "X-CSRF-TOKEN-REFRESH" # Example header name
    # Validation for JWT_SECRET_KEY is now handled in config.py

    # Initialize extensions
    jwt.init_app(app)
    # CORS(app, ...) # Moved CORS initialization above jwt.init_app(app) - Reverting this based on common practice

    # Generate SAML settings.json at startup
    try:
        generate_saml_settings_json()
    except Exception as e:
        debug_error(f"Failed to generate SAML settings.json at startup: {e}")
        # Depending on criticality, you might want to raise here or log and continue
        # For now, we'll log and allow the app to proceed, but SAML will be broken.

    # Define the API prefix
    API_PREFIX = f"{config.ROUTE_PREFIX}/backend/api/v1"
    API_PREFIX_V2 = f"{config.ROUTE_PREFIX}/backend/api/v2"

    # --- Define root-level routes BEFORE blueprints ---
    # Removed temporary /test route

    @app.route(f"{config.ROUTE_PREFIX}/health") # Prefixed
    def health_check():
        return {"status": "ok"}
    # -------------------------------------------------

    # Register Blueprints

    # Import and register the auth blueprint
    from .routes.auth_routes import auth_bp
    app.register_blueprint(auth_bp, url_prefix=f'{API_PREFIX}/auth')

    # Import and register the document blueprint
    from .routes.document_routes import doc_bp
    # Register doc_bp with its full intended prefix and disable strict slashes
    # Relying on global CORS(app, ...) defined earlier
    app.register_blueprint(doc_bp, url_prefix=f'{API_PREFIX}/docs', strict_slashes=False)

    # Import and register the system blueprint
    from .routes.system_routes import system_bp
    # Register system_bp with its full intended prefix (prefix removed from blueprint definition)
    app.register_blueprint(system_bp, url_prefix=f'{API_PREFIX}/system')

    # Import and register the log blueprint
    from .routes.log_routes import log_bp
    app.register_blueprint(log_bp, url_prefix=f'{API_PREFIX}/logs')

    # Import and register the user blueprint
    from .routes.user_routes import user_bp
    app.register_blueprint(user_bp, url_prefix=f'{API_PREFIX}/users')

    from .routes.gcs_routes import gcs_bp
    app.register_blueprint(gcs_bp, url_prefix=f'{API_PREFIX}/gcs')

    # Import and register the processor rule blueprint
    from .routes.processor_rule_routes import processor_rules_bp
    app.register_blueprint(processor_rules_bp, url_prefix=f'{API_PREFIX}/processor-rules')

    # Import and register the batch processing blueprint
    from .routes.batch_processing_routes import batch_processing_bp
    app.register_blueprint(batch_processing_bp, url_prefix=f'{API_PREFIX}/batch')

    # Import and register the activity log blueprint
    from .routes.activity_log_routes import activity_log_bp
    app.register_blueprint(activity_log_bp, url_prefix=f'{API_PREFIX}/activity-logs')

    # --- V2 Blueprints for RBAC Implementation ---
    # API_PREFIX_V2 defined above

    # Import and register the batch processing blueprint
    from .routes.batch_routes import batch_bp
    app.register_blueprint(batch_bp, url_prefix=f'{API_PREFIX_V2}')

    # Import and register the secure document blueprint
    from .routes.secure_document_routes import secure_doc_bp
    app.register_blueprint(secure_doc_bp, url_prefix=f'{API_PREFIX_V2}')

    # Import and register the V2 system blueprint
    from .routes.system_routes_v2 import system_bp_v2
    app.register_blueprint(system_bp_v2, url_prefix=f'{API_PREFIX_V2}/system')

    # Enable CORS globally after all blueprints are registered
    # This ensures CORS headers are applied to all routes, including those in blueprints.
    CORS(
        app,
        origins="*",  # Allow all origins
        methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
        supports_credentials=False,  # Must be False when origins="*"
        allow_headers=["Authorization", "Content-Type"]
    )

    # --- Test routes moved above ---

    # --- Temporary Debug Route ---
    @app.route(f"{config.ROUTE_PREFIX}/debug/routes")
    def list_routes():
        import urllib.parse
        output = []
        for rule in app.url_map.iter_rules():
            options = {}
            for arg in rule.arguments:
                options[arg] = f"[{arg}]"
            methods = ','.join(rule.methods)
            url = urllib.parse.unquote(rule.endpoint)
            line = f"{rule.endpoint:50s} {methods:20s} {rule.rule}"
            output.append(line)
        
        # Sort routes for readability
        output.sort()
        # Return as plain text for easy viewing
        return "<pre>" + "\n".join(output) + "</pre>"
    
    from werkzeug.middleware.proxy_fix import ProxyFix
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)
 
    return app
