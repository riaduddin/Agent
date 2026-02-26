# backend/app/routes/auth_routes.py
from flask import Blueprint, request, jsonify, redirect
from app.models.user_model import create_user, find_user_by_email, verify_password, update_user_profile, change_user_password
from flask_jwt_extended import create_access_token, create_refresh_token, jwt_required, get_jwt_identity, get_jwt
from onelogin.saml2.auth import OneLogin_Saml2_Auth
import os
import logging
import json
import traceback
from functools import wraps
from app import config
from app.utils.utils import admin_required
from app.services.activity_log_service import log_authentication_activity
from app.models.activity_log_model import ActivityTypes
from app.utils.debug_logger import debug_log, debug_warn, debug_error

# FIXED: Setup logger
logger = logging.getLogger(__name__)

auth_bp = Blueprint('auth_bp', __name__)

def prepare_flask_request(req):
    url_data = req.url.split('?')
    return {
        'https': 'on' if req.environ.get('wsgi.url_scheme') == 'https' else 'off',
        'http_host': req.host,
        'server_port': req.environ.get('SERVER_PORT'),
        'script_name': req.path,
        'get_data': req.args.copy(),
        'post_data': req.form.copy()
    }

def load_saml_settings_from_json():
    """Load SAML settings from the settings.json file"""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        saml_settings_path = os.path.join(current_dir, '..', 'saml', 'settings.json')
        
        if not os.path.exists(saml_settings_path):
            saml_settings_path = os.path.join(os.getcwd(), 'saml', 'settings.json')
        
        debug_log(f"Looking for SAML settings at: {saml_settings_path}")
        
        if os.path.exists(saml_settings_path):
            with open(saml_settings_path, 'r') as f:
                settings = json.load(f)
                debug_log("SAML settings loaded from JSON file")
                return settings
        else:
            debug_error(f"SAML settings file not found at: {saml_settings_path}")
            return None
            
    except Exception as e:
        debug_error(f"Error loading SAML settings: {e}")
        return None

def get_saml_settings():
    return {
        'strict': False,  # Changed from True to False for more lenient validation
        'debug': True,
        "sp": {
            "entityId": "https://intellidocfinder-dfcsffs.dhs.ga.gov/backend/api/v1/auth/metadata/",
            "assertionConsumerService": {
            "url": "https://intellidocfinder-dfcsffs.dhs.ga.gov/backend/api/v1/auth/sso/acs",
            "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST"
            },
        },
        "idp": {
            "entityId": "http://www.okta.com/exk1lkg6mwaAZpE3P358",
            "singleSignOnService": {
            "url": "https://connect.gets.ga.gov/app/gets_dhsdfcs_1/exk1lkg6mwaAZpE3P358/sso/saml",
            "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect"
            },
            "x509cert": "MIIDmDCCAoCgAwIBAgIGAZfCBneWMA0GCSqGSIb3DQEBCwUAMIGMMQswCQYDVQQGEwJVUzETMBEGA1UECAwKQ2FsaWZvcm5pYTEWMBQGA1UEBwwNU2FuIEZyYW5jaXNjbzENMAsGA1UECgwET2t0YTEUMBIGA1UECwwLU1NPUHJvdmlkZXIxDTALBgNVBAMMBGdldHMxHDAaBgkqhkiG9w0BCQEWDWluZm9Ab2t0YS5jb20wHhcNMjUwNjMwMTgwNzEzWhcNMzUwNjMwMTgwODEyWjCBjDELMAkGA1UEBhMCVVMxEzARBgNVBAgMCkNhbGlmb3JuaWExFjAUBgNVBAcMDVNhbiBGcmFuY2lzY28xDTALBgNVBAoMBE9rdGExFDASBgNVBAsMC1NTT1Byb3ZpZGVyMQ0wCwYDVQQDDARnZXRzMRwwGgYJKoZIhvcNAQkBFg1pbmZvQG9rdGEuY29tMIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAkIBjRzquU8HgTagxUKSHCR1HRHD79YoWnosHulbX6s6/VQgSYMBijkF/5ym8AvS90ovaSE27iAYbJIdBUsbO4o2VU4htCR4mcPvWAx+PvTVUCGT7ykOJqaGWOreQvF63oZpQA6Po8INuEwc86RPk6gPlBrKRzpRzgglLLKoMnaLD7XO+UaBxze6eMX0MEBSQwkQhuoYaXD/VqEnq9C/qVyTkhLAtUyhdG0WRsqhW1LW0U8ZmFKOmb7P1ljkWHXb4HlMtPGkq5l4UFny6AymlKlzimtc3IVAf/3Is9vzfz3BwT+61qkcXaufkN0RqrblH7kOtyneInfk6k3GJlrJ+RQIDAQABMA0GCSqGSIb3DQEBCwUAA4IBAQArvWJu4fk33eB5taCpOatxj4K+BsQddu9VQlOSyVdcKcW4I5tHnwi4+hhQhn+mmQ+opyW0xygeJTXQ6Vls39ykXOsDbKVqykGdCDY+GCQQ+gnU4Glsw93rNd+IVdnH7szzr3n8kBisEqBXir1p4mwX2UgoGyWmru2+1buQCTi8iRMeID0DdbnVoVKZfRyFnqqqv1NwDg77FJDLkjJUmXeainu78xaXoGoUQtlUPu1x9cm/cvZuiS3Xr8zmbJRGyEoGA/8MfudFYoyErWe8sSwKKr6QXwyGtRch1rj8JAG7PkRA3p339J7FHnhof2645d/YXauQx505vZwaHdo0f8qN"
        },
        'security': {
            'nameIdEncrypted': False,
            'authnRequestsSigned': False,
            'logoutRequestsSigned': False,
            'logoutResponsesSigned': False,
            'signMetadata': False,
            'wantAssertionsSigned': False,
            'wantMessagesSigned': False,
            'wantNameId': True,
            'wantNameIdEncrypted': False,
            'wantAssertionsEncrypted': False,
            'allowRepeatAttributeName': False,
            'rejectUnsolicitedResponsesWithInResponseTo': False,  # Added this
            'signatureAlgorithm': 'http://www.w3.org/2001/04/xmldsig-more#rsa-sha256',
            'digestAlgorithm': 'http://www.w3.org/2001/04/xmlenc#sha256',
        }
    }


# CRASH-SAFE TEST ROUTES
@auth_bp.route('/test/ping', methods=["GET"])
def test_ping():
    """Simple test route"""
    return jsonify({"msg": "Auth routes working", "status": "ok"}), 200

@auth_bp.route('/test/saml-imports', methods=["GET"])
def test_saml_imports():
    """Test if SAML libraries can be imported"""
    try:
        from onelogin.saml2.auth import OneLogin_Saml2_Auth
        from onelogin.saml2.settings import OneLogin_Saml2_Settings
        
        return jsonify({
            "msg": "SAML imports successful",
            "saml_auth_class": str(OneLogin_Saml2_Auth),
            "saml_settings_class": str(OneLogin_Saml2_Settings),
            "status": "imports_ok"
        }), 200
    except Exception as e:
        return jsonify({
            "msg": "SAML import failed",
            "error": str(e),
            "error_type": type(e).__name__,
            "status": "import_error"
        }), 500

@auth_bp.route('/test/saml-settings', methods=["GET"])
def test_saml_settings():
    """Test if SAML settings can be loaded"""
    try:
        json_settings = load_saml_settings_from_json()
        hardcoded_settings = get_saml_settings()
        
        return jsonify({
            "msg": "SAML settings test",
            "json_settings_loaded": json_settings is not None,
            "hardcoded_settings_loaded": hardcoded_settings is not None,
            "json_settings_keys": list(json_settings.keys()) if json_settings else None,
            "hardcoded_settings_keys": list(hardcoded_settings.keys()) if hardcoded_settings else None,
            "status": "settings_test_complete"
        }), 200
    except Exception as e:
        return jsonify({
            "msg": "SAML settings test failed",
            "error": str(e),
            "error_type": type(e).__name__,
            "status": "settings_error"
        }), 500

# FIXED: SSO Login route
@auth_bp.route('/sso/login')
def sso_login():
    try:
        debug_log("SSO LOGIN CALLED")
        saml_settings = load_saml_settings_from_json()
        if not saml_settings:
            saml_settings = get_saml_settings()
        
        auth = OneLogin_Saml2_Auth(prepare_flask_request(request), old_settings=saml_settings)
        login_url = auth.login()
        debug_log(f"Redirecting to: {login_url}")
        return redirect(login_url)
    except Exception as e:
        debug_error(f"SSO Login failed: {e}")
        debug_error(f"Error traceback: {traceback.format_exc()}")
        return jsonify({"msg": "SSO login failed", "error": str(e)}), 500






# SAFE ACS ROUTE - Manual SAML Response Parsing (Bypass Library)
@auth_bp.route('/sso/acs', methods=["POST"])
def sso_acs():
    debug_log("=" * 80)
    debug_log("SSO ACS ENDPOINT CALLED - MANUAL PARSING MODE")
    debug_log("PURPOSE: Process SAML response from IdP and extract user email")
    debug_log("=" * 80)
    
    try:
        debug_log("STEP 1: INITIAL SAFETY CHECKS")
        debug_log("Checking if form data exists...")
        
        # Immediate safety checks
        if not request.form:
            debug_error("RESULT: No form data received from request")
            return jsonify({"msg": "No form data received"}), 400
        
        debug_log(f"RESULT: Form data found with keys: {list(request.form.keys())}")
        
        debug_log("Checking for SAMLResponse in form data...")
        if 'SAMLResponse' not in request.form:
            debug_error("RESULT: No SAMLResponse field found in form data")
            return jsonify({"msg": "No SAMLResponse found"}), 400
        
        debug_log("RESULT: SAMLResponse field found in form data")
        
        debug_log("STEP 2: EXTRACTING SAML RESPONSE")
        debug_log("Getting base64-encoded SAML response from form...")
        saml_response_b64 = request.form['SAMLResponse']
        debug_log(f"RESULT: SAML response extracted (length: {len(saml_response_b64)} characters)")
        debug_log(f"First 100 characters of base64 response: {saml_response_b64[:100]}...")
        
        debug_log("STEP 3: DECODING SAML RESPONSE")
        debug_log("Importing required libraries for decoding...")
        try:
            import base64
            import xml.etree.ElementTree as ET
            import urllib.parse
            debug_log("RESULT: Required libraries imported successfully")
            
            debug_log("Decoding base64 SAML response to XML...")
            saml_response_xml = base64.b64decode(saml_response_b64)
            debug_log(f"RESULT: Base64 decoded successfully (XML length: {len(saml_response_xml)} bytes)")
            
            debug_log("Parsing XML string into ElementTree...")
            root = ET.fromstring(saml_response_xml)
            debug_log(f"RESULT: XML parsed successfully (root tag: {root.tag})")
            
            debug_log("Converting XML bytes to readable string for debugging...")
            xml_str = saml_response_xml.decode('utf-8', errors='ignore')
            debug_log("RESULT: XML converted to string successfully")
            debug_log(f"RAW SAML RESPONSE (first 500 chars): {xml_str[:500]}...")
            
        except Exception as decode_error:
            debug_error(f"RESULT: Manual decode failed with error: {decode_error}")
            debug_error(f"ERROR TYPE: {type(decode_error).__name__}")
            return jsonify({"msg": "Failed to decode SAML response", "error": str(decode_error)}), 400
        
        debug_log("STEP 4: EMAIL EXTRACTION PROCESS")
        debug_log("PURPOSE: Extract user email from SAML response using multiple methods")
        email = None
        
        try:
            debug_log("Setting up XML namespaces for SAML parsing...")
            # Define XML namespaces commonly used in SAML
            namespaces = {
                'saml2': 'urn:oasis:names:tc:SAML:2.0:assertion',
                'saml2p': 'urn:oasis:names:tc:SAML:2.0:protocol',
                'saml': 'urn:oasis:names:tc:SAML:2.0:assertion',
                'samlp': 'urn:oasis:names:tc:SAML:2.0:protocol'
            }
            debug_log("RESULT: XML namespaces configured")
            debug_log(f"Namespaces: {list(namespaces.keys())}")
            
            debug_log("STEP 4A: METHOD 1 - SEARCHING FOR NAMEID")
            debug_log("Trying to find NameID element (primary user identifier)...")
            
            nameid_paths = [
                './/saml2:NameID',
                './/saml:NameID',
                './/NameID'
            ]
            debug_log(f"Will try these XPath patterns: {nameid_paths}")
            
            for i, path in enumerate(nameid_paths, 1):
                try:
                    debug_log(f"Attempt {i}: Searching with pattern '{path}'...")
                    nameid_element = root.find(path, namespaces)
                    if nameid_element is not None:
                        debug_log(f"FOUND: NameID element found with pattern '{path}'")
                        if nameid_element.text:
                            email = nameid_element.text
                            debug_log(f"SUCCESS: Email extracted from NameID: {email}")
                            debug_log(f"NameID Format: {nameid_element.get('Format', 'Not specified')}")
                            break
                        else:
                            debug_warn("WARNING: NameID element found but contains no text")
                    else:
                        debug_log(f"RESULT: No NameID found with pattern '{path}'")
                except Exception as e:
                    debug_error(f"ERROR: Exception while searching with pattern '{path}': {e}")
                    continue
            
            if email:
                debug_log(f"METHOD 1 SUCCESS: Email found via NameID: {email}")
            else:
                debug_log("METHOD 1 FAILED: No email found in NameID elements")
            
            debug_log("STEP 4B: METHOD 2 - SEARCHING IN ATTRIBUTES")
            debug_log("Searching for email in SAML attributes...")
            
            if not email:
                attribute_paths = [
                    './/saml2:Attribute',
                    './/saml:Attribute',
                    './/Attribute'
                ]
                debug_log(f"Will try these attribute XPath patterns: {attribute_paths}")
                
                for i, attr_path in enumerate(attribute_paths, 1):
                    try:
                        debug_log(f"Attempt {i}: Searching attributes with pattern '{attr_path}'...")
                        attributes = root.findall(attr_path, namespaces)
                        debug_log(f"FOUND: {len(attributes)} attribute(s) found with pattern '{attr_path}'")
                        
                        for j, attr in enumerate(attributes, 1):
                            attr_name = attr.get('Name', 'Unknown')
                            debug_log(f"Examining attribute {j}: '{attr_name}'")
                            
                            # Common email attribute names
                            email_keywords = ['email', 'mail', 'emailaddress']
                            attr_name_lower = attr_name.lower()
                            
                            debug_log(f"Checking if '{attr_name_lower}' contains email keywords: {email_keywords}")
                            
                            if any(email_attr in attr_name_lower for email_attr in email_keywords):
                                debug_log(f"MATCH: Attribute '{attr_name}' appears to be an email field")
                                
                                # Try different value element patterns
                                value_patterns = [
                                    ('.//saml2:AttributeValue', namespaces),
                                    ('.//saml:AttributeValue', namespaces),
                                    ('.//AttributeValue', {})
                                ]
                                
                                for k, (pattern, ns) in enumerate(value_patterns, 1):
                                    debug_log(f"Attempt {k}: Looking for value with pattern '{pattern}'...")
                                    value_element = attr.find(pattern, ns)
                                    
                                    if value_element is not None and value_element.text:
                                        email = value_element.text
                                        debug_log(f"SUCCESS: Email found in attribute '{attr_name}': {email}")
                                        break
                                    else:
                                        debug_log(f"No value found with pattern '{pattern}'")
                                
                                if email:
                                    break
                            else:
                                debug_log(f"SKIP: Attribute '{attr_name}' doesn't appear to be email-related")
                        
                        if email:
                            break
                            
                    except Exception as e:
                        debug_error(f"ERROR: Exception while searching attributes with pattern '{attr_path}': {e}")
                        continue
                
                if email:
                    debug_log(f"METHOD 2 SUCCESS: Email found via attributes: {email}")
                else:
                    debug_log("METHOD 2 FAILED: No email found in attributes")
            else:
                debug_log("SKIPPING METHOD 2: Email already found via NameID")
            
            debug_log("STEP 4C: METHOD 3 - REGEX SEARCH")
            debug_log("Searching for email patterns in entire XML content...")
            
            if not email:
                import re
                email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
                debug_log(f"Using regex pattern: {email_pattern}")
                
                email_matches = re.findall(email_pattern, xml_str)
                debug_log(f"FOUND: {len(email_matches)} email-like strings in XML")
                
                if email_matches:
                    email = email_matches[0]  # Take the first email found
                    debug_log(f"SUCCESS: Email found via regex: {email}")
                    if len(email_matches) > 1:
                        debug_log(f"Other emails found: {email_matches[1:]}")
                else:
                    debug_log("RESULT: No email patterns found in XML content")
                
                if email:
                    debug_log(f"METHOD 3 SUCCESS: Email found via regex: {email}")
                else:
                    debug_log("METHOD 3 FAILED: No email found via regex")
            else:
                debug_log("SKIPPING METHOD 3: Email already found")
            
            debug_log("STEP 4D: DEBUGGING - LISTING ALL ATTRIBUTES")
            debug_log("Extracting all available attributes for debugging...")
            
            if not email:
                debug_log("PURPOSE: Since no email found, listing all attributes to help debug")
                try:
                    attribute_paths = [
                        './/saml2:Attribute',
                        './/saml:Attribute',
                        './/Attribute'
                    ]
                    
                    all_attributes_found = False
                    
                    for attr_path in attribute_paths:
                        attributes = root.findall(attr_path, namespaces)
                        if attributes:
                            all_attributes_found = True
                            debug_log(f"Attributes found with pattern '{attr_path}':")
                            
                            for i, attr in enumerate(attributes, 1):
                                attr_name = attr.get('Name', 'Unknown')
                                
                                # Try to get value
                                value_element = attr.find('.//saml2:AttributeValue', namespaces) or \
                                              attr.find('.//saml:AttributeValue', namespaces) or \
                                              attr.find('.//AttributeValue')
                                
                                value = value_element.text if value_element is not None else 'No value'
                                debug_log(f"  {i}. {attr_name}: {value}")
                    
                    if not all_attributes_found:
                        debug_log("RESULT: No attributes found with any pattern")
                        
                except Exception as debug_error_exc:
                    debug_error(f"ERROR: Could not extract attributes for debugging: {debug_error_exc}")
            else:
                debug_log("SKIPPING DEBUG LISTING: Email already found")
            
        except Exception as parse_error:
            debug_error(f"CRITICAL ERROR during email extraction: {parse_error}")
            debug_error(f"ERROR TYPE: {type(parse_error).__name__}")
            debug_error(f"FULL TRACEBACK: {traceback.format_exc()}")
        
        debug_log("STEP 5: EMAIL VALIDATION AND FALLBACK")
        debug_log("Checking final email result...")
        
        # Use fallback email if extraction failed
        if not email:
            # Redirect user to frontend fallback page
            frontend_uri = os.getenv("FRONTEND_URL", "http://localhost:3000")
            debug_log(f"Frontend URL from environment: {frontend_uri}")
            
            redirect_uri = f"{frontend_url}/sso-callback-error?message=User not Found"
            debug_log("RESULT: Redirect URL constructed")
            debug_log(f"Full redirect URL: {redirect_uri}")
            return redirect(redirect_uri)

        else:
            debug_log("SUCCESS: Email validation passed")
        
        debug_log(f"FINAL RESULT: Email to use for JWT: {email}")
        
        debug_log("STEP 6: JWT TOKEN CREATION")
        debug_log("Creating JWT access token for user...")
        debug_log(f"Token will include: identity='{email}', role='user'")

        # get name from email
        email_name = email.split('@')[0]
        debug_log(f"Extracted name from email: {email_name}")


        
        debug_log("STEP 7: PREPARING REDIRECT RESPONSE")
        debug_log("Building redirect URL for frontend...")
        
        frontend_url = os.getenv("FRONTEND_URL", "http://localhost:3000")
        debug_log(f"Frontend URL from environment: {frontend_url}")


        # Check this user is already registered or not 
        user = find_user_by_email(email)
        if not user:
            debug_warn(f"WARNING: User '{email}' not found in database, creating new user profile")
            # Create new user profile
            # create_user(email=email, name=email_name, is_sso_user=True, role='user', password=None)
            # debug_log(f"RESULT: New user profile created for '{email}'")
            # Redirect user to frontend fallback page
            frontend_uri2 = os.getenv("FRONTEND_URL", "http://localhost:3000")
            debug_log(f"Frontend URL from environment: {frontend_uri2}")
            
            redirect_uri2 = f"{frontend_uri2}/sso-callback-error?message=User '{email}' Not found in the system"
            debug_log("RESULT: Redirect URL constructed")
            debug_log(f"Full redirect URL: {redirect_uri2}")
            return redirect(redirect_uri2)
        else:
            debug_log(f"RESULT: User '{email}' already exists in database")

            user_role = user.get('role')

            # Log User Role
            debug_log(f"User role found: {user_role}")


            access_token = create_access_token(identity=email, additional_claims={"role": user_role, "name": email_name})
            debug_log("RESULT: JWT token created successfully")
            debug_log(f"Token length: {len(access_token)} characters")
            logger.info(f"JWT token created successfully for: {email}")

            redirect_url = f"{frontend_url}/sso-callback?token={access_token}"
            debug_log("RESULT: Redirect URL constructed")
            debug_log(f"Full redirect URL: {redirect_url}")
            
            logger.info(f"Redirecting to: {redirect_url}")
            
            debug_log("OVERALL SUCCESS: SSO ACS processing completed successfully")
            debug_log(f"User '{email}' will be redirected to frontend with valid JWT token")
            debug_log("=" * 80)
            
            return redirect(redirect_url)
        
    except Exception as e:
        debug_error("CRITICAL SYSTEM ERROR occurred")
        debug_error(f"Error message: {str(e)}")
        debug_error(f"Error type: {type(e).__name__}")
        debug_error(f"Full traceback: {traceback.format_exc()}")
        
        logger.error(f"SSO ACS error: {str(e)}", exc_info=True)
        
        return jsonify({
            "msg": "SSO processing failed",
            "error": str(e),
            "error_type": type(e).__name__
        }), 500
    
    finally:
        debug_log("SSO ACS ENDPOINT FINISHED")
        debug_log("Cleanup completed")
        debug_log("=" * 80)




# @auth_bp.route('/register', methods=['POST'])
# def register():
    # """Registers a new user."""
    # print("--- REGISTER ROUTE START ---")
    # try:
    #     data = request.get_json()
    #     print(f"Received registration data: {data}")
    #     email = data.get('email')
    #     password = data.get('password')
    #     name = data.get('name')
    #     print(f"Extracted - Email: {email}, Password: {'******' if password else None}, Name: {name}")

    #     if not email or not password:
    #         print("Validation Error: Email or password missing.")
    #         return jsonify({"msg": "Email and password are required"}), 400

    #     if '@' not in email or '.' not in email:
    #          print(f"Validation Error: Invalid email format for {email}")
    #          return jsonify({"msg": "Invalid email format"}), 400

    #     if len(password) < 8:
    #         print("Validation Error: Password too short.")
    #         return jsonify({"msg": "Password must be at least 8 characters long"}), 400

    #     print(f"Attempting to create user for email: {email}")
    #     user_id, error = create_user(email, password, name, role='admin')

    #     if error:
    #         print(f"Error during user creation: {error}")
    #         status_code = 409 if "already exists" in error else 500
    #         print(f"Returning error response with status code: {status_code}")
    #         return jsonify({"msg": error}), status_code

    #     print(f"User created successfully with ID: {user_id}")
    #     print("--- REGISTER ROUTE SUCCESS ---")
    #     return jsonify({"msg": "User registered successfully", "user_id": user_id}), 201

    # except Exception as e:
    #     print(f"--- REGISTER ROUTE UNEXPECTED ERROR ---")
    #     print(f"Exception type: {type(e).__name__}")
    #     print(f"Exception args: {e.args}")
    #     logger.error(f"Unexpected error in registration route: {e}", exc_info=True)
    #     return jsonify({"msg": "An unexpected error occurred during registration."}), 500

@auth_bp.route('/login', methods=['POST'])
def login():
    """Logs in a user and returns JWT tokens."""
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({"msg": "Email and password are required"}), 400

    user = find_user_by_email(email)

    if user and verify_password(user.get('password'), password):
        identity = email.lower()
        user_role = user.get('role', 'user')
        additional_claims = {"role": user_role}
        
        access_token = create_access_token(identity=identity, additional_claims=additional_claims)
        refresh_token = create_refresh_token(identity=identity)

        user_info = {k: v for k, v in user.items() if k != 'password_hash'}
        user_info['role'] = user_role

        # Log successful login activity
        log_authentication_activity(
            user_email=email,
            activity_type=ActivityTypes.AUTH_LOGIN,
            success=True,
            additional_info={'role': user_role, 'login_method': 'password'},
            request_obj=request
        )

        return jsonify(
            access_token=access_token,
            user=user_info
        ), 200
    else:
        # Log failed login attempt
        log_authentication_activity(
            user_email=email,
            activity_type=ActivityTypes.AUTH_LOGIN_FAILED,
            success=False,
            additional_info={'reason': 'invalid_credentials', 'login_method': 'password'},
            request_obj=request
        )
        return jsonify({"msg": "Invalid credentials"}), 401

@auth_bp.route('/me', methods=['GET'])
@jwt_required()
def me():
    """Returns the current user's information."""
    try:
        current_user_email = get_jwt_identity()
        if not current_user_email:
            return jsonify({"msg": "Could not identify user from token"}), 401

        user = find_user_by_email(current_user_email)

        if user:
            user_info = {k: v for k, v in user.items() if k != 'password_hash'}
            if 'id' not in user_info and '_id' in user:
                 user_info['id'] = str(user['_id'])
            if 'role' not in user_info:
                 user_info['role'] = 'user'

            return jsonify(user_info), 200
        else:
            return jsonify({"msg": "User not found"}), 404

    except Exception as e:
        logger.error(f"Error fetching user profile: {e}", exc_info=True)
        return jsonify({"msg": "An internal error occurred while fetching user profile."}), 500


@auth_bp.route('/logout', methods=['POST'])
@jwt_required()
def logout():
    """Logs out a user and logs the activity."""
    try:
        current_user_email = get_jwt_identity()
        
        # Log logout activity
        log_authentication_activity(
            user_email=current_user_email,
            activity_type=ActivityTypes.AUTH_LOGOUT,
            success=True,
            additional_info={'logout_method': 'manual'},
            request_obj=request
        )
        
        return jsonify({"msg": "Successfully logged out"}), 200
        
    except Exception as e:
        logger.error(f"ERROR: Failed to logout user: {e}", exc_info=True)
        return jsonify({"msg": "An error occurred during logout."}), 500


@auth_bp.route('/profile', methods=['PATCH'])
@jwt_required()
def update_profile():
    """Updates the current user's profile information (e.g., name)."""
    try:
        current_user_email = get_jwt_identity()
        if not current_user_email:
            return jsonify({"msg": "Could not identify user from token"}), 401

        data = request.get_json()
        new_name = data.get('name')

        if not new_name:
            return jsonify({"msg": "Name is required for update"}), 400

        success, error_msg = update_user_profile(current_user_email, new_name)

        if success:
            updated_user = find_user_by_email(current_user_email)
            if updated_user:
                user_info = {k: v for k, v in updated_user.items() if k != 'password_hash'}
                return jsonify({"msg": "Profile updated successfully", "user": user_info}), 200
            else:
                 return jsonify({"msg": "Profile updated, but failed to retrieve updated data"}), 200
        else:
            status_code = 500 if "Internal" in error_msg else 404
            return jsonify({"msg": f"Profile update failed: {error_msg}"}), status_code

    except Exception as e:
        logger.error(f"Error updating profile: {e}", exc_info=True)
        return jsonify({"msg": "An internal error occurred during profile update."}), 500

@auth_bp.route('/change-password', methods=['PATCH'])
@jwt_required()
def change_password():
    """Changes the current user's password."""
    try:
        current_user_email = get_jwt_identity()
        if not current_user_email:
            return jsonify({"msg": "Could not identify user from token"}), 401

        data = request.get_json()
        current_password = data.get('currentPassword')
        new_password = data.get('newPassword')

        if not current_password or not new_password:
            return jsonify({"msg": "Current password and new password are required"}), 400

        if len(new_password) < 8:
             return jsonify({"msg": "New password must be at least 8 characters long"}), 400

        success, error_msg = change_user_password(current_user_email, current_password, new_password)

        if success:
            return jsonify({"msg": "Password updated successfully"}), 200
        else:
            if "Invalid current password" in error_msg:
                status_code = 401
            elif "User not found" in error_msg:
                status_code = 404
            else:
                status_code = 500
            return jsonify({"msg": f"Password change failed: {error_msg}"}), status_code

    except Exception as e:
        logger.error(f"Error changing password: {e}", exc_info=True)
        return jsonify({"msg": "An internal error occurred during password change."}), 500
