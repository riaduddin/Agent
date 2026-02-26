# backend/app/routes/system_routes_v2.py
import logging
from flask import Blueprint, jsonify
from flask_jwt_extended import jwt_required
from app import db
from app.utils.utils import admin_or_superadmin_required
from app import config

logger = logging.getLogger(__name__)
system_bp_v2 = Blueprint('system_bp_v2', __name__)

@system_bp_v2.route('/categories', methods=['GET'])
@jwt_required()
def get_all_categories():
    """
    Retrieves a list of all unique document categories from Firestore.
    """
    try:
        all_categories = set()
        docs_stream = db.collection("document_metadata").stream()
        for doc in docs_stream:
            doc_data = doc.to_dict()
            categories = doc_data.get("categories")
            if categories and isinstance(categories, list):
                for category in categories:
                    all_categories.add(category)
        
        return jsonify(sorted(list(all_categories))), 200
    except Exception as e:
        logger.error(f"Failed to retrieve categories: {e}", exc_info=True)
        return jsonify({"msg": "Failed to retrieve document categories."}), 500

@system_bp_v2.route('/valid-categories', methods=['GET'])
@admin_or_superadmin_required
def get_valid_categories():
    """
    Retrieves the list of all valid document categories.
    Admin only endpoint.
    """
    try:
        return jsonify(config.VALID_CATEGORIES), 200
    except Exception as e:
        logger.error(f"Failed to retrieve valid categories: {e}", exc_info=True)
        return jsonify({"msg": "Failed to retrieve valid categories."}), 500
