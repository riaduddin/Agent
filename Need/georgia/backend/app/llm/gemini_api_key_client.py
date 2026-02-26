import os
import logging
from typing import List, Optional, Union, Any, Dict, Generator
from google import genai
from google.genai import types, errors

logger = logging.getLogger(__name__)

class GeminiConfigurationError(Exception):
    """Exception raised for Gemini API configuration errors (e.g., invalid API key)."""
    pass

class GeminiApiKeyClient:
    """
    A unified client for interacting with Google's Gemini models using an API Key.
    Uses the `google-genai` SDK (v1.0+).
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initializes the Gemini client.
        
        Args:
            api_key: Optional API key. If not provided, it looks for 'GEMINI_API_KEY' or 'GOOGLE_API_KEY'.
        """
        # SDKs often prefer GEMINI_API_KEY now
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        """Initializes the underlying genai.Client."""
        # Check if we should use Vertex AI
        use_vertex = os.environ.get("USE_VERTEX_AI", "true").lower() == "true"
        project = os.environ.get("PROJECT_ID", "gadhs-dd-uat")
        location = os.environ.get("VERTEX_LOCATION", "us-east1")

        try:
            if use_vertex and project:
                # Vertex AI uses ADC/Service Account, NOT API Key
                self.client = genai.Client(
                    vertexai=True,
                    project=project,
                    location=location
                )
                logger.info(f"GeminiApiKeyClient initialized with Vertex AI (Project: {project}, Location: {location})")
            elif self.api_key:
                # Google AI Studio uses API Key
                self.client = genai.Client(api_key=self.api_key)
                masked_key = f"{self.api_key[:4]}...{self.api_key[-4:]}" if len(self.api_key) > 8 else "****"
                logger.info(f"GeminiApiKeyClient initialized with API Key: {masked_key}")
            else:
                logger.warning("Neither GEMINI_API_KEY found nor USE_VERTEX_AI enabled. Client will not be functional.")
        except Exception as e:
            logger.error(f"Failed to initialize GeminiApiKeyClient: {e}")
            self.client = None

    def is_available(self) -> bool:
        """Checks if the client is successfully initialized."""
        return self.client is not None

    def generate_content(
        self,
        contents: Union[str, List[Union[str, Any]]],
        model_name: Optional[str] = None,
        temperature: float = 0.7,
        max_output_tokens: Optional[int] = None,
        system_instruction: Optional[str] = None,
        safety_settings: Optional[List[Dict[str, str]]] = None,
        response_mime_type: Optional[str] = None,
        stream: bool = False
    ) -> Union[str, Generator[str, None, None]]:
        """
        Generates content using the specified model.

        Args:
            contents: The input prompt or list of content parts.
            model_name: The model to use (default: gemini-2.5-flash).
            temperature: Strictness of the output (0.0 - 2.0).
            max_output_tokens: Max tokens to generate.
            system_instruction: Optional system instruction.
            safety_settings: Optional safety settings.
            response_mime_type: Optional MIME type (e.g., "application/json").
            stream: Whether to stream the response.

        Returns:
            The generated text or a generator if streaming is enabled.
        """
        if not self.client:
            msg = "Gemini client not initialized. Neither GEMINI_API_KEY nor GOOGLE_API_KEY found."
            logger.error(msg)
            raise RuntimeError(msg)

        try:
            # Map safety settings to SDK types if provided
            config_safety_settings = None
            if safety_settings:
                config_safety_settings = [
                    types.SafetySetting(
                        category=s.get("category"),
                        threshold=s.get("threshold")
                    ) for s in safety_settings
                ]

            config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                system_instruction=system_instruction,
                safety_settings=config_safety_settings,
                response_mime_type=response_mime_type
            )

            # Check if contents is a list of Pydantic Parts or dicts that need conversion?
            # The SDK handles list of strings or Part objects.
            
            # Fallback to default if model_name not provided
            target_model = model_name or "gemini-2.5-flash"

            if stream:
                response = self.client.models.generate_content_stream(
                    model=target_model,
                    contents=contents,
                    config=config
                )
                return self._stream_response_generator(response)
            else:
                response = self.client.models.generate_content(
                    model=target_model,
                    contents=contents,
                    config=config
                )
                return response

        except (errors.ClientError, errors.APIError) as e:
            # Check for authentication/configuration errors (401, 403, or invalid API key)
            err_msg = str(e).lower()
            if "api key" in err_msg or "unauthorized" in err_msg or "authenticated" in err_msg or "401" in err_msg or "403" in err_msg:
                friendly_msg = "AI Service temporarily unavailable (Configuration Error)."
                logger.error(f"Gemini Configuration Error: {e}")
                raise GeminiConfigurationError(friendly_msg) from e
            
            logger.error(f"Error in generate_content: {e}", exc_info=True)
            raise e
        except Exception as e:
            logger.error(f"Unexpected error in generate_content: {e}", exc_info=True)
            raise e

    def _stream_response_generator(self, response_iterator):
        """Helper to yield text from stream chunks."""
        try:
            for chunk in response_iterator:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            logger.error(f"Error during streaming: {e}")
            yield f" [Error during streaming: {e}]"

    def generate_chat_response(
        self,
        message: str,
        history: Optional[List[Dict[str, Any]]] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.7,
        max_output_tokens: Optional[int] = None,
        system_instruction: Optional[str] = None,
        safety_settings: Optional[List[Dict[str, str]]] = None,
        stream: bool = False
    ) -> Union[str, Generator[str, None, None]]:
        """
        Manages a chat session. Supports streaming and safety settings.
        """
        if not self.client:
            raise RuntimeError("Gemini client unavailable.")

        try:
            # Convert history format if necessary
            # The SDK expects history as list of Content objects or similar structure
            # app/services/vertex_ai_service.py uses [{'role': 'user', 'parts': [{'text': ...}]}]
            
            # types.Content(role="user", parts=[types.Part(text="...")])

            formatted_history = []
            if history:
                for turn in history:
                    # Robustly handle different history formats (dict vs object)
                    role = turn.get('role') if isinstance(turn, dict) else getattr(turn, 'role', 'user')
                    parts_raw = turn.get('parts') if isinstance(turn, dict) else getattr(turn, 'parts', [])
                    
                    parts = []
                    for p in parts_raw:
                        text = p.get('text') if isinstance(p, dict) else getattr(p, 'text', str(p))
                        parts.append(types.Part(text=text))
                    
                    formatted_history.append(types.Content(role=role, parts=parts))

            # Safety settings mapping
            config_safety_settings = None
            if safety_settings:
                config_safety_settings = [
                    types.SafetySetting(
                        category=s.get("category"),
                        threshold=s.get("threshold")
                    ) for s in safety_settings
                ]

            config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                system_instruction=system_instruction,
                safety_settings=config_safety_settings
            )

            # Fallback to default if model_name not provided
            target_model = model_name or "gemini-2.5-flash"

            chat = self.client.chats.create(
                model=target_model,
                history=formatted_history,
                config=config
            )
            
            if stream:
                response_iterator = chat.send_message_stream(message)
                return self._stream_response_generator(response_iterator)
            else:
                response = chat.send_message(message)
                return response.text

        except (errors.ClientError, errors.APIError) as e:
            # Check for authentication/configuration errors
            err_msg = str(e).lower()
            if "api key" in err_msg or "unauthorized" in err_msg or "authenticated" in err_msg or "401" in err_msg or "403" in err_msg:
                friendly_msg = "AI Service temporarily unavailable (Configuration Error)."
                logger.error(f"Gemini Configuration Error in chat: {e}")
                raise GeminiConfigurationError(friendly_msg) from e

            logger.error(f"Error in chat response: {e}", exc_info=True)
            raise e
        except Exception as e:
            logger.error(f"Unexpected error in chat response: {e}", exc_info=True)
            raise e

# Global instance
gemini_client = GeminiApiKeyClient()
