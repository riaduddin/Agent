from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity, get_jwt # Import get_jwt
from google.cloud.firestore import Client
from google.cloud import firestore # Import firestore module
from werkzeug.security import generate_password_hash
from flask_cors import CORS # Import CORS

from app import db # Import the initialized Firestore client
from app.services.activity_log_service import log_admin_activity, log_navigation_activity
from app.models.activity_log_model import ActivityTypes
from app.utils.debug_logger import debug_log, debug_warn, debug_error

# Reference to the 'users' collection in Firestore
users_ref = db.collection('users')

user_bp = Blueprint('user_bp', __name__, url_prefix='/api/users')
CORS(user_bp, origins="*")  # Allow all origins



@user_bp.route('/', methods=['GET'], strict_slashes=False)
@jwt_required()
def list_users():
    # Admin role check
    claims = get_jwt()
    if claims.get("role") != "admin":
        return jsonify({"msg": "Admins only"}), 403

    try:
        users = []
        for doc in users_ref.stream():
            user_data = doc.to_dict()
            user_data["id"] = doc.id  # Always include the document ID
            user_data.pop('password_hash', None)  # Remove sensitive fields
            user_data.pop('password', None)
            users.append(user_data)

        # Log admin user management access
        current_user_email = get_jwt_identity()
        log_navigation_activity(
            user_email=current_user_email,
            page_name='User Management',
            page_path='/user-management',
            request_obj=request
        )

        return jsonify(users), 200
    except Exception as e:
        current_app.logger.error(f"Error listing users: {e}")
        return jsonify({"message": "Error listing users", "error": str(e)}), 500



# Placeholder route to create a new user
@user_bp.route('/', methods=['POST'])
@jwt_required()
# @admin_required # TODO: Implement admin role check
def create_user():
    # Admin role check
    claims = get_jwt()
    if claims.get("role") != "admin":
        return jsonify({"msg": "Admins only"}), 403

    try:
        # Use the module-level users_ref
        data = request.get_json()
        name = data.get('name') # Get name instead of username
        email = data.get('email')
        password = data.get('password')
        role = data.get('role')

        if not name or not email or not password: # Check for name instead of username
            return jsonify({"message": "Missing name, email, or password"}), 400 # Update error message

        # Check if user already exists (by email or username)
        users_ref = db.collection('users')
        existing_user = users_ref.where('email', '==', email).limit(1).get()
        if existing_user:
            return jsonify({"message": "User with this email already exists"}), 409
            
        # Optional: Check if user with the same name already exists (if name must be unique)
        # existing_user = users_ref.where('username', '==', name).limit(1).get() # Check against 'username' field in Firestore
        # if existing_user:
        #     return jsonify({"message": "User with this name already exists"}), 409

        hashed_password = generate_password_hash(password)

        new_user_ref = users_ref.document()
        new_user_ref.set({
            'id': new_user_ref.id, # Store the document ID as a field
            'name': name, # Store the name in the 'name' field in Firestore
            'email': email,
            'password': hashed_password,
            'role': role,
            'created_at': firestore.SERVER_TIMESTAMP,
            'updated_at': firestore.SERVER_TIMESTAMP,
            # Add other fields as needed (e.g., roles)
        })

        # Log user creation activity
        current_user_email = get_jwt_identity()
        log_admin_activity(
            user_email=current_user_email,
            admin_action='user_create',
            target_info={
                'created_user_email': email,
                'created_user_name': name,
                'created_user_role': role,
                'user_id': new_user_ref.id
            },
            request_obj=request
        )

        return jsonify({"message": "User created successfully", "user_id": new_user_ref.id}), 201
    except Exception as e:
        current_app.logger.error(f"Error creating user: {e}")
        return jsonify({"message": "Error creating user", "error": str(e)}), 500

# Placeholder route to get a specific user by ID
@user_bp.route('/<string:user_id>', methods=['GET'])
@jwt_required()
# @admin_required # TODO: Implement admin role check
def get_user(user_id):
    try:
        # Use the module-level users_ref
        user_ref = users_ref.document(user_id)
        user_doc = user_ref.get()

        if not user_doc.exists:
            return jsonify({"message": "User not found"}), 404

        return jsonify(user_doc.to_dict()), 200
    except Exception as e:
        current_app.logger.error(f"Error getting user {user_id}: {e}")
        return jsonify({"message": f"Error getting user {user_id}", "error": str(e)}), 500

# Placeholder route to update a specific user by ID
@user_bp.route('/<string:user_id>', methods=['PUT'])
@jwt_required()
# @admin_required # TODO: Implement admin role check
def update_user(user_id):
    # Admin role check
    claims = get_jwt()
    if claims.get("role") != "admin":
        return jsonify({"msg": "Admins only"}), 403
    
    debug_log(f"Updating user with ID: {user_id}")  # Debugging line to check user_id

    try:
        # Use the module-level users_ref
        user_ref = users_ref.document(user_id)
        user_doc = user_ref.get()

        if not user_doc.exists:
            return jsonify({"message": "User not found"}), 404

        data = request.get_json()
        update_data = {}

        if 'username' in data:
            update_data['username'] = data['username']
        if 'name' in data:
            update_data['name'] = data['name']
        if 'email' in data:
            update_data['email'] = data['email']
        if 'password' in data and data['password']: # Only update password if provided
            update_data['password'] = generate_password_hash(data['password'])
        if 'role' in data:
            update_data['role'] = data['role']
        if 'accessible_categories' in data:
            update_data['accessible_categories'] = data['accessible_categories']

        if not update_data:
            return jsonify({"message": "No update data provided"}), 400

        update_data['updated_at'] = firestore.SERVER_TIMESTAMP # Update timestamp

        user_ref.update(update_data)

        # Log user update activity
        current_user_email = get_jwt_identity()
        user_data = user_doc.to_dict()
        log_admin_activity(
            user_email=current_user_email,
            admin_action='user_update',
            target_info={
                'updated_user_id': user_id,
                'updated_user_email': user_data.get('email'),
                'updated_fields': list(update_data.keys()),
                'update_data': {k: v for k, v in update_data.items() if k != 'password'}  # Exclude password from log
            },
            request_obj=request
        )

        return jsonify({"message": f"User {user_id} updated successfully"}), 200
    except Exception as e:
        current_app.logger.error(f"Error updating user {user_id}: {e}")
        return jsonify({"message": f"Error updating user {user_id}", "error": str(e)}), 500

# Placeholder route to delete a specific user by ID
@user_bp.route('/<string:user_id>', methods=['DELETE'])
@jwt_required()
# @admin_required # TODO: Implement admin role check
def delete_user(user_id):
    # Admin role check
    claims = get_jwt()
    if claims.get("role") != "admin":
        return jsonify({"msg": "Admins only"}), 403

    try:
        # Use the module-level users_ref
        user_ref = users_ref.document(user_id)
        user_doc = user_ref.get()

        if not user_doc.exists:
            return jsonify({"message": "User not found"}), 404

        # Get user data before deletion for logging
        user_data = user_doc.to_dict()
        
        user_ref.delete()

        # Log user deletion activity
        current_user_email = get_jwt_identity()
        log_admin_activity(
            user_email=current_user_email,
            admin_action='user_delete',
            target_info={
                'deleted_user_id': user_id,
                'deleted_user_email': user_data.get('email'),
                'deleted_user_name': user_data.get('name'),
                'deleted_user_role': user_data.get('role')
            },
            request_obj=request
        )

        return jsonify({"message": f"User {user_id} deleted successfully"}), 200
    except Exception as e:
        current_app.logger.error(f"Error deleting user {user_id}: {e}")
        return jsonify({"message": f"Error deleting user {user_id}", "error": str(e)}), 500
