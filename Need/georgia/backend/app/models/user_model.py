# backend/app/models/user_model.py
from app import db # Import the initialized Firestore client
from werkzeug.security import generate_password_hash, check_password_hash
import datetime
from app.utils.debug_logger import debug_log, debug_error

# Reference to the 'users' collection in Firestore
users_ref = db.collection('users')

def create_user(email, password, name=None, role='user',is_sso_user=False): # Add role parameter with default
    """Creates a new user document in Firestore."""
    debug_log("--- CREATE_USER START ---")
    try:
        debug_log(f"Checking if user exists for email: {email}")
        existing_user = find_user_by_email(email)
        # Check if user already exists
        if existing_user:
            debug_log(f"User found for email {email}. Registration aborted.")
            return None, "User with this email already exists."
        debug_log(f"User not found for email {email}. Proceeding with creation.")
        
        user_data = {
            'email': email.lower(),
            'name': name,
            'role': role,
            'is_sso_user': is_sso_user,
            'created_at': datetime.datetime.now(tz=datetime.timezone.utc)
        }
        if password:
            debug_log("Generating password hash...")
            hashed_password = generate_password_hash(password)
            user_data['password'] = hashed_password
            debug_log("Password hash generated.")
        else:
            debug_log("No password provided; assuming SSO registration.")
        # user_data = {
        #     'email': email.lower(),
        #     'password': hashed_password,
        #     'name': name,
        #     'role': role, # Include role in user data
        #     'created_at': datetime.datetime.now(tz=datetime.timezone.utc)
        #     # Add other fields as needed later
        # }
        debug_log(f"Prepared user data: { {k: v for k, v in user_data.items() if k != 'password_hash'} }") # Don't print hash
        # Use email as the document ID for easy lookup
        doc_id = email.lower()
        user_doc_ref = users_ref.document(doc_id)
        debug_log(f"Attempting to set user data in Firestore with document ID: {doc_id}")
        user_doc_ref.set(user_data)
        debug_log(f"Firestore set operation successful for document ID: {doc_id}")
        debug_log("--- CREATE_USER SUCCESS ---")
        return user_doc_ref.id, None
    except Exception as e:
        debug_error("--- CREATE_USER ERROR ---")
        debug_error(f"Exception type: {type(e).__name__}")
        debug_error(f"Exception args: {e.args}")
        debug_error(f"Exception args: {e}")
        debug_error(f"Error creating user: {e}")
        return None, "An error occurred during registration."

def find_user_by_email(email):
    """Finds a user document by email."""
    debug_log(f"--- FIND_USER_BY_EMAIL START for: {email} ---")
    try:
        # Query for the user document where the 'email' field matches the provided email
        debug_log(f"Attempting to query Firestore for user with email: {email}")
        users_ref = db.collection('users') # Get collection reference within the function
        query = users_ref.where('email', '==', email.lower()).limit(1)
        docs = query.stream()

        user_doc = next(docs, None) # Get the first document, or None if no match

        if user_doc and user_doc.exists:
            debug_log(f"Document found for email: {email}")
            debug_log("--- FIND_USER_BY_EMAIL SUCCESS (User Found) ---")
            user_data = user_doc.to_dict()
            user_data['id'] = user_doc.id # Add the document ID to the dictionary
            return user_data # Returns a dictionary with user data
        else:
            debug_log(f"Document not found for email: {email}")
            debug_log("--- FIND_USER_BY_EMAIL SUCCESS (User Not Found) ---")
            return None # Returns None if user is not found

    except Exception as e:
        debug_error("--- FIND_USER_BY_EMAIL ERROR ---")
        debug_error(f"Exception type: {type(e).__name__}")
        debug_error(f"Exception args: {e.args}")
        debug_error(f"Error finding user by email: {e}")
        return None

def verify_password(stored_hash, provided_password):
    """Verifies a provided password against the stored hash."""
    # Revert to standard signature
    if stored_hash:
        return check_password_hash(stored_hash, provided_password)
    return False # Return False if stored_hash is None


def update_user_profile(email, new_name):
    """Updates the user's name in their Firestore document."""
    try:
        user_doc_ref = users_ref.document(email.lower())
        user_doc = user_doc_ref.get()

        if not user_doc.exists:
            return False, "User not found."

        # Update the name field
        user_doc_ref.update({
            'name': new_name,
            'updated_at': datetime.datetime.now(tz=datetime.timezone.utc) # Optional: track updates
        })
        return True, None
    except Exception as e:
        debug_error(f"Error updating user profile for {email}: {e}")
        return False, "Internal server error during profile update."


def change_user_password(email, current_password, new_password):
    """Changes the user's password after verifying the current one."""
    try:
        user_doc_ref = users_ref.document(email.lower())
        user_doc = user_doc_ref.get()

        if not user_doc.exists:
            return False, "User not found."

        user_data = user_doc.to_dict()
        stored_hash = user_data.get('password')

        # Verify the current password
        if not stored_hash or not verify_password(stored_hash, current_password):
            return False, "Invalid current password."

        # Hash the new password
        new_password = generate_password_hash(new_password)

        # Update the password hash in Firestore
        user_doc_ref.update({
            'password': new_password,
            'updated_at': datetime.datetime.now(tz=datetime.timezone.utc) # Track update time
        })
        return True, None
    except Exception as e:
        debug_error(f"Error changing password for {email}: {e}")
        return False, "Internal server error during password change."
