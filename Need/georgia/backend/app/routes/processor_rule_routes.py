from flask import Blueprint, request, jsonify, current_app
from app.models.processor_routing_rule_model import ProcessorRoutingRuleModel
from app.utils.utils import admin_or_superadmin_required

processor_rules_bp = Blueprint('processor_rules_bp', __name__, url_prefix='/api/v1/processor-rules')

@processor_rules_bp.route('', methods=['POST'])
@admin_or_superadmin_required
def create_processor_rule():
    """
    Creates a new processor routing rule.
    Requires admin privileges.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing data"}), 400

    try:
        # Basic validation, more can be added
        # Removed 'classifierProcessorId' from required fields
        required_fields = ['ruleName', 'documentTypeLabel', 'targetParserProcessorId', 'isEnabled', 'priority']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # Ensure classifierProcessorId is at least an empty string if not provided, to avoid issues if model still expects it
        if 'classifierProcessorId' not in data:
            data['classifierProcessorId'] = "" # Or None, depending on how model handles it. Empty string is safer for now.

        rule_id = ProcessorRoutingRuleModel.create_rule(data)
        if rule_id:
            return jsonify({"message": "Processor rule created successfully", "id": rule_id}), 201
        else:
            return jsonify({"error": "Failed to create processor rule"}), 500
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        current_app.logger.error(f"Error creating processor rule: {e}")
        return jsonify({"error": "An unexpected error occurred"}), 500

@processor_rules_bp.route('', methods=['GET'])
@admin_or_superadmin_required # Or perhaps allow all authenticated users to view? For now, admin.
def get_all_processor_rules():
    """
    Retrieves all processor routing rules.
    Requires admin privileges.
    """
    try:
        enabled_only_str = request.args.get('enabled_only', 'true').lower()
        enabled_only = enabled_only_str == 'true'
        
        order_by_priority_str = request.args.get('order_by_priority', 'true').lower()
        order_by_priority = order_by_priority_str == 'true'

        rules = ProcessorRoutingRuleModel.get_all_rules(enabled_only=enabled_only, order_by_priority=order_by_priority)
        return jsonify(rules), 200
    except Exception as e:
        current_app.logger.error(f"Error getting all processor rules: {e}")
        return jsonify({"error": "An unexpected error occurred"}), 500

@processor_rules_bp.route('/<rule_id>', methods=['GET'])
@admin_or_superadmin_required # Or perhaps allow all authenticated users to view?
def get_processor_rule(rule_id):
    """
    Retrieves a specific processor routing rule by its ID.
    Requires admin privileges.
    """
    try:
        rule = ProcessorRoutingRuleModel.get_rule(rule_id)
        if rule:
            return jsonify(rule), 200
        else:
            return jsonify({"error": "Processor rule not found"}), 404
    except Exception as e:
        current_app.logger.error(f"Error getting processor rule {rule_id}: {e}")
        return jsonify({"error": "An unexpected error occurred"}), 500

@processor_rules_bp.route('/<rule_id>', methods=['PUT'])
@admin_or_superadmin_required
def update_processor_rule(rule_id):
    """
    Updates an existing processor routing rule.
    Requires admin privileges.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing data"}), 400

    try:
        success = ProcessorRoutingRuleModel.update_rule(rule_id, data)
        if success:
            return jsonify({"message": "Processor rule updated successfully"}), 200
        else:
            # Could be rule not found or other update error
            rule_exists = ProcessorRoutingRuleModel.get_rule(rule_id)
            if not rule_exists:
                return jsonify({"error": "Processor rule not found"}), 404
            return jsonify({"error": "Failed to update processor rule"}), 500
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        current_app.logger.error(f"Error updating processor rule {rule_id}: {e}")
        return jsonify({"error": "An unexpected error occurred"}), 500

@processor_rules_bp.route('/<rule_id>', methods=['DELETE'])
@admin_or_superadmin_required
def delete_processor_rule(rule_id):
    """
    Deletes a processor routing rule.
    Requires admin privileges.
    """
    try:
        success = ProcessorRoutingRuleModel.delete_rule(rule_id)
        if success:
            return jsonify({"message": "Processor rule deleted successfully"}), 200
        else:
            # Could be rule not found or other delete error
            rule_exists = ProcessorRoutingRuleModel.get_rule(rule_id)
            if not rule_exists:
                return jsonify({"error": "Processor rule not found"}), 404
            return jsonify({"error": "Failed to delete processor rule"}), 500
    except Exception as e:
        current_app.logger.error(f"Error deleting processor rule {rule_id}: {e}")
        return jsonify({"error": "An unexpected error occurred"}), 500
