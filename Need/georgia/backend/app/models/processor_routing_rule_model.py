from google.cloud import firestore
from google.cloud.firestore_v1.base_query import FieldFilter
import datetime

from app import db # Assuming db is the Firestore client initialized in app/__init__.py
from app.utils.debug_logger import debug_log, debug_error

PROCESSOR_RULES_COLLECTION = "processor_routing_rules"

class ProcessorRoutingRuleModel:
    """
    Model for managing Document AI processor routing rules in Firestore.
    """

    @staticmethod
    def create_rule(data: dict):
        """
        Creates a new processor routing rule.
        Args:
            data (dict): A dictionary containing rule data.
                         Expected keys: ruleName, documentTypeLabel, classifierProcessorId,
                                        targetParserProcessorId, isEnabled (bool), priority (int),
                                        description (optional).
        Returns:
            str: The ID of the newly created rule document, or None if error.
        """
        try:
            # Add timestamps
            timestamp = datetime.datetime.now(tz=datetime.timezone.utc)
            data['createdAt'] = timestamp
            data['updatedAt'] = timestamp

            # Validate required fields (basic validation)
            # 'classifierProcessorId' is now optional
            required_fields = ['ruleName', 'documentTypeLabel', 'targetParserProcessorId', 'isEnabled', 'priority']
            for field in required_fields:
                if field not in data:
                    raise ValueError(f"Missing required field: {field}")
            
            # Ensure classifierProcessorId exists, defaulting to empty string if not provided by API layer
            if 'classifierProcessorId' not in data:
                data['classifierProcessorId'] = ""

            if not isinstance(data['isEnabled'], bool):
                raise ValueError("isEnabled must be a boolean.")
            if not isinstance(data['priority'], int):
                raise ValueError("priority must be an integer.")

            doc_ref = db.collection(PROCESSOR_RULES_COLLECTION).add(data)
            return doc_ref[1].id # add() returns a tuple (timestamp, DocumentReference)
        except Exception as e:
            # Consider logging the error e.g., current_app.logger.error(...)
            debug_error(f"Error creating processor rule: {e}")
            return None

    @staticmethod
    def get_rule(rule_id: str):
        """
        Retrieves a specific processor routing rule by its ID.
        Args:
            rule_id (str): The ID of the rule document.
        Returns:
            dict: The rule data if found, else None.
        """
        try:
            doc_ref = db.collection(PROCESSOR_RULES_COLLECTION).document(rule_id)
            doc = doc_ref.get()
            if doc.exists:
                rule_data = doc.to_dict()
                rule_data['id'] = doc.id
                return rule_data
            return None
        except Exception as e:
            debug_error(f"Error getting processor rule {rule_id}: {e}")
            return None

    @staticmethod
    def get_all_rules(enabled_only: bool = True, order_by_priority: bool = True):
        """
        Retrieves all processor routing rules, optionally filtered and ordered.
        Args:
            enabled_only (bool): If True, only returns rules where isEnabled is True.
            order_by_priority (bool): If True, orders rules by the 'priority' field (ascending).
        Returns:
            list: A list of rule data dictionaries.
        """
        try:
            query = db.collection(PROCESSOR_RULES_COLLECTION)
            if enabled_only:
                query = query.where(filter=FieldFilter("isEnabled", "==", True))
            
            if order_by_priority:
                query = query.order_by("priority", direction=firestore.Query.ASCENDING)
            
            rules = []
            for doc in query.stream():
                rule_data = doc.to_dict()
                rule_data['id'] = doc.id
                rules.append(rule_data)
            return rules
        except Exception as e:
            debug_error(f"Error getting all processor rules: {e}")
            return []

    @staticmethod
    def update_rule(rule_id: str, data: dict):
        """
        Updates an existing processor routing rule.
        Args:
            rule_id (str): The ID of the rule document to update.
            data (dict): A dictionary containing fields to update.
        Returns:
            bool: True if update was successful, False otherwise.
        """
        try:
            doc_ref = db.collection(PROCESSOR_RULES_COLLECTION).document(rule_id)
            if not doc_ref.get().exists:
                return False # Rule not found

            # Add updatedAt timestamp
            data['updatedAt'] = datetime.datetime.now(tz=datetime.timezone.utc)
            
            # Basic validation for updatable fields
            if 'isEnabled' in data and not isinstance(data['isEnabled'], bool):
                raise ValueError("isEnabled must be a boolean.")
            if 'priority' in data and not isinstance(data['priority'], int):
                raise ValueError("priority must be an integer.")

            doc_ref.update(data)
            return True
        except Exception as e:
            debug_error(f"Error updating processor rule {rule_id}: {e}")
            return False

    @staticmethod
    def delete_rule(rule_id: str):
        """
        Deletes a processor routing rule.
        Args:
            rule_id (str): The ID of the rule document to delete.
        Returns:
            bool: True if deletion was successful, False otherwise.
        """
        try:
            doc_ref = db.collection(PROCESSOR_RULES_COLLECTION).document(rule_id)
            if not doc_ref.get().exists:
                return False # Rule not found
            
            doc_ref.delete()
            return True
        except Exception as e:
            debug_error(f"Error deleting processor rule {rule_id}: {e}")
            return False

if __name__ == '__main__':
    # This is for basic testing if you run the file directly.
    # You'd need to have Firestore emulator running or real credentials configured.
    # And `db` needs to be initialized.
    
    # Example:
    # from app import create_app
    # app = create_app()
    # with app.app_context():
    #     # Test data
    #     test_rule_data = {
    #         "ruleName": "Test Invoice Rule",
    #         "documentTypeLabel": "INVOICE_TEST",
    #         "classifierProcessorId": "projects/your-project/locations/us/processors/classifier-id",
    #         "targetParserProcessorId": "projects/your-project/locations/us/processors/invoice-parser-id",
    #         "isEnabled": True,
    #         "priority": 1,
    #         "description": "A test rule for invoices."
    #     }
    #     created_id = ProcessorRoutingRuleModel.create_rule(test_rule_data)
    #     if created_id:
    #         print(f"Created rule with ID: {created_id}")

    #         retrieved_rule = ProcessorRoutingRuleModel.get_rule(created_id)
    #         print(f"Retrieved rule: {retrieved_rule}")

    #         all_rules = ProcessorRoutingRuleModel.get_all_rules()
    #         print(f"All enabled rules (ordered): {all_rules}")
            
    #         update_success = ProcessorRoutingRuleModel.update_rule(created_id, {"isEnabled": False, "description": "Updated description."})
    #         print(f"Update success: {update_success}")
    #         if update_success:
    #             updated_rule = ProcessorRoutingRuleModel.get_rule(created_id)
    #             print(f"Updated rule: {updated_rule}")

    #         # delete_success = ProcessorRoutingRuleModel.delete_rule(created_id)
    #         # print(f"Delete success: {delete_success}")
    #     else:
    #         print("Failed to create rule.")
    pass
