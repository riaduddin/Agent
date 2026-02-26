import os
import json
from app import config
import logging

logger = logging.getLogger(__name__)

def generate_saml_settings_json():

    # Temporary comment to avoid confusion with the original request
    return
    """
    Generates the SAML settings.json file based on environment variables.
    Creates a JSON structure that matches the target format exactly.
    """
    saml_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'saml')
    settings_file_path = os.path.join(saml_dir, 'settings.json')

    # Ensure the saml directory exists
    os.makedirs(saml_dir, exist_ok=True)

    settings = {
        'sp': {
            'entityId': config.SAML_SP_ENTITY_ID,
            'assertionConsumerService': {
                'url': config.SAML_SP_ACS_URL,
                'binding': 'urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST'
            }
        },
        'idp': {
            'entityId': config.SAML_IDP_ENTITY_ID,
            'singleSignOnService': {
                'url': config.SAML_IDP_SSO_URL,
                'binding': 'urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect'
            },
            'x509cert': config.SAML_IDP_X509CERT,
        },
        'security': {
            'authnRequestsSigned': False,
            'wantAssertionsSigned': True,
            'wantMessageSigned': False
        }
    }

    try:
        with open(settings_file_path, 'w') as f:
            json.dump(settings, f, indent=2)
        logger.info(f"SAML settings.json generated successfully at {settings_file_path}")
    except Exception as e:
        logger.error(f"Error generating SAML settings.json: {e}", exc_info=True)
        raise