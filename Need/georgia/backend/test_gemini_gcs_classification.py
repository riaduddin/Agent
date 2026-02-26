# import logging
# import json
# import os
# from dotenv import load_dotenv

# # --- Load .env file ---
# backend_dir = os.path.dirname(os.path.abspath(__file__))
# dotenv_path = os.path.join(backend_dir, '.env')
# if os.path.exists(dotenv_path):
#     load_dotenv(dotenv_path=dotenv_path)
#     print(f"TEST_SCRIPT: Loaded .env file from: {dotenv_path}")
# else:
#     print(f"TEST_SCRIPT: .env file not found at {dotenv_path}. Relying on system environment variables.")

# # Attempting to use imports exactly as in the user's provided sample code
# try:
#     from google import genai # Using 'from google import genai'
#     from google.genai import types
#     google_genai_sdk_imported = True
#     print("TEST_SCRIPT: Successfully imported 'google.genai' and 'google.genai.types'")
# except ImportError as e:
#     print(f"TEST_SCRIPT: Failed to import from 'google.genai': {e}. This is the primary SDK path the user's sample uses.")
#     print("TEST_SCRIPT: Ensure the correct Google AI SDK providing 'google.genai' is installed and accessible.")
#     google_genai_sdk_imported = False
#     genai = None # Define to prevent NameError if import fails
#     types = None # Define to prevent NameError if import fails

# # --- Configuration (from environment variables) ---
# PROJECT_ID = os.getenv("PROJECT_ID") # Changed back to getenv
# VERTEX_LOCATION = os.getenv("VERTEX_LOCATION") # Changed back to getenv
# GEMINI_MODEL_NAME_FROM_ENV = os.getenv("GEMINI_MODEL_NAME") # Changed back to getenv

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# def test_gcs_classification_with_user_sample_structure():
#     if not google_genai_sdk_imported or not genai or not types:
#         logger.error("TEST_SCRIPT: 'google.genai' SDK not available or failed to import. Exiting.")
#         return

#     # --- Data from test script context ---
#     chunk_gcs_uri_to_test = "gs://georgia_bucket/chunks/1GvkZtcWJhHiywBp4i78/chunk_1_0aca97ed-a09a-450a-b1e9-5d5e2e245c78.pdf"
#     parser_labels_for_prompt = ["INVOICE", "DOCUMENT_OCR", "GENERAL_DOCUMENT"]
    
#     model_name_to_use = GEMINI_MODEL_NAME_FROM_ENV or "gemini-1.5-flash-latest"
#     if GEMINI_MODEL_NAME_FROM_ENV:
#         print(f"TEST_SCRIPT: Using GEMINI_MODEL_NAME from .env: {GEMINI_MODEL_NAME_FROM_ENV}")
#     else:
#         print(f"TEST_SCRIPT: GEMINI_MODEL_NAME not in .env, using fallback: {model_name_to_use}")

#     logger.info(f"TEST_SCRIPT: Using GCS URI: {chunk_gcs_uri_to_test}")
#     logger.info(f"TEST_SCRIPT: Using Parser Labels for prompt: {parser_labels_for_prompt}")
#     logger.info(f"TEST_SCRIPT: Using Model: {model_name_to_use}")
#     logger.info(f"TEST_SCRIPT: Using Project ID: {PROJECT_ID}, Location: {VERTEX_LOCATION}")

#     # --- Client Initialization (as per user sample) ---
#     client = None
#     try:
#         client = genai.Client(
#             vertexai=True,
#             project=PROJECT_ID,
#             location=VERTEX_LOCATION
#         )
#         logger.info("TEST_SCRIPT: Initialized genai.Client with vertexai=True.")
#     except Exception as e:
#         logger.error(f"TEST_SCRIPT: Failed to initialize genai.Client: {e}", exc_info=True)
#         return

#     if not client: # Should be caught by except block, but good practice
#         logger.error("TEST_SCRIPT: Client is None after initialization attempt.")
#         return

#     # --- Prepare Prompt, Parts, Contents, Config (as per user sample) ---
#     parser_labels_str = json.dumps(parser_labels_for_prompt)
#     system_instruction_text = f"""given parser is 
# {parser_labels_str}

# and find the appropriate parser of the given PDF. 

# # response format
# 1. Give only the string based on the available parser given
# 2. do not give any additional explanation or narration"""

#     try:
#         logger.info("TEST_SCRIPT: Constructing parts and config using 'google.genai.types'...")
        
#         # Using 'file_uri' as per user's sample and to fix previous TypeError
#         document_part = types.Part.from_uri(
#             file_uri=chunk_gcs_uri_to_test, # Changed to file_uri
#             mime_type="application/pdf",
#         )
#         empty_text_part = types.Part.from_text(text="") # Empty text part from user sample

#         system_instruction_for_config = [types.Part.from_text(text=system_instruction_text)]
        
#         contents = [
#             types.Content(
#                 role="user",
#                 parts=[document_part, empty_text_part] 
#             )
#         ]

#         generation_config_obj = types.GenerateContentConfig(
#             temperature=1.0,
#             top_p=1.0,
#             max_output_tokens=100, 
#             safety_settings=[
#                 types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
#                 types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
#                 types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
#                 types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF")
#             ],
#             system_instruction=system_instruction_for_config
#         )
#         logger.info("TEST_SCRIPT: Successfully constructed parts and generation_config.")

#         logger.info(f"TEST_SCRIPT: Calling client.models.generate_content (model: {model_name_to_use})...")
        
#         response = client.models.generate_content(
#             model=model_name_to_use, 
#             contents=contents,
#             generation_config=generation_config_obj 
#         )
        
#         logger.info("TEST_SCRIPT: Response received.")

#         if hasattr(response, 'text') and response.text:
#             logger.info(f"TEST_SCRIPT: Response Text: {response.text.strip()}")
#         elif hasattr(response, 'candidates') and response.candidates and \
#              response.candidates[0].content and response.candidates[0].content.parts and \
#              response.candidates[0].content.parts[0].text:
#             logger.info(f"TEST_SCRIPT: Response Text from Candidate: {response.candidates[0].content.parts[0].text.strip()}")
#         else:
#             logger.info(f"TEST_SCRIPT: No direct text in response. Full response object: {response}")

#         if hasattr(response, 'prompt_feedback'):
#             logger.info(f"TEST_SCRIPT: Prompt Feedback: {response.prompt_feedback}")

#     except AttributeError as ae:
#         logger.error(f"TEST_SCRIPT FAILED (AttributeError): {ae}. This indicates an issue with how 'types.Part', 'types.Content', 'types.GenerateContentConfig', or 'client.models.generate_content' is being used with the 'google.genai' SDK version in your environment.", exc_info=True)
#     except TypeError as te:
#         logger.error(f"TEST_SCRIPT FAILED (TypeError): {te}. This might be due to incorrect keyword arguments for 'client.models.generate_content' or config objects.", exc_info=True)
#     except Exception as e:
#         logger.error(f"TEST_SCRIPT FAILED (Other Exception): {e}", exc_info=True)

# if __name__ == "__main__":
#     if not all([PROJECT_ID, VERTEX_LOCATION, GEMINI_MODEL_NAME_FROM_ENV]):
#         print("ERROR: Missing one or more environment variables: PROJECT_ID, VERTEX_LOCATION, GEMINI_MODEL_NAME")
#         print("Please ensure your .env file is correctly set up in the backend directory.")
#     else:
#         print(f"TEST_SCRIPT: Running with Project: {PROJECT_ID}, Location: {VERTEX_LOCATION}")
#         test_gcs_classification_with_user_sample_structure()



from google import genai
from google.genai import types
import base64

def generate():
  client = genai.Client(
      vertexai=True,
      project="dev-project-458711",
      location="global",
  )

  msg1_document1 = types.Part.from_uri(
      file_uri="gs://georgia_bucket/6.pdf",
      mime_type="application/pdf",
  )
  si_text1 = """given parser is 
[\"INVOICE\", \"DOCUMENT\"]

and find the appropriate parser of the given PDF. 

# response format
1. Give only the string based on the available parser given
2. do not give any additional explanation or narration"""

  #model = "gemini-2.5-pro"
  model = "gemini-2.5-flash"
  contents = [
    types.Content(
      role="user",
      parts=[
        msg1_document1,
        types.Part.from_text(text=""".""")
      ]
    ),
  ]

  generate_content_config = types.GenerateContentConfig(
    temperature = 1,
    top_p = 1,
    seed = 0,
    max_output_tokens = 65535,
    safety_settings = [types.SafetySetting(
      category="HARM_CATEGORY_HATE_SPEECH",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_DANGEROUS_CONTENT",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_HARASSMENT",
      threshold="OFF"
    )],
    system_instruction=[types.Part.from_text(text=si_text1)],
  )

  for chunk in client.models.generate_content_stream(
    model = model,
    contents = contents,
    config = generate_content_config,
    ):
    print(chunk.text, end="")

generate()