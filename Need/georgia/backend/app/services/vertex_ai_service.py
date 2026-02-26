# backend/app/services/vertex_ai_service.py
import os
import grpc
import google.cloud.aiplatform as aiplatform
from google.cloud import aiplatform_v1
from google.cloud.aiplatform.gapic.schema import predict
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
from google.api_core import exceptions as api_core_exceptions
from google import genai
from google.genai import types
import time
# import json # Removed as it was only used by the deleted function
from typing import Optional, Tuple
from app import config
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

from google.cloud.aiplatform import gapic
from google.api_core.exceptions import GoogleAPIError

from google.cloud.aiplatform_v1beta1 import IndexServiceClient
from google.cloud.aiplatform_v1 import IndexServiceClient as IndexServiceClientV1 # Alias to avoid conflict if used
from google.api_core.client_options import ClientOptions
from google.cloud.aiplatform_v1.types import RemoveDatapointsRequest
from google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint import Namespace
from google.api_core import exceptions
from app.llm.gemini_api_key_client import gemini_client, GeminiConfigurationError
from app.utils.redis_client import get_redis_client
from app.utils.debug_logger import debug_log, debug_warn, debug_error, debug_perf, debug_separator
import hashlib
import json
# Removed direct import of HarmCategory, HarmBlockThreshold, FinishReason from .types

# --- Configuration (from config module) ---
PROJECT_ID = config.PROJECT_ID
LOCATION = config.VERTEX_LOCATION
VECTOR_INDEX_NAME_CONFIG = config.VECTOR_INDEX_NAME
EMBEDDING_MODEL_NAME = config.EMBEDDING_MODEL_NAME
GEMINI_OCR_MODEL_NAME = config.GEMINI_OCR_MODEL_NAME
GEMINI_CHAT_MODEL_NAME = config.GEMINI_CHAT_MODEL_NAME
GEMINI_MODEL_NAME = config.GEMINI_MODEL_NAME # Backward compatibility
INDEX_ENDPOINT_ID_CONFIG = config.VECTOR_INDEX_ENDPOINT_ID
DEPLOYED_INDEX_ID_CONFIG = config.VECTOR_DEPLOYED_INDEX_ID
PRIVATE_ENDPOINT_IP_CONFIG = config.VECTOR_PRIVATE_ENDPOINT_IP

DEFAULT_ENDPOINT_DISPLAY_NAME = "georgia-doc-index-endpoint"
DEFAULT_INDEX_DISPLAY_NAME = "georgia-doc-index"
DEFAULT_DEPLOYED_INDEX_ID = "deployed_georgia_index_001"
EMBEDDING_DIMENSIONS = 768

logger = logging.getLogger(__name__)
aiplatform.init(project=PROJECT_ID, location=LOCATION)

index_endpoint = None
if INDEX_ENDPOINT_ID_CONFIG:
    try:
        index_endpoint_name = f"projects/{PROJECT_ID}/locations/{LOCATION}/indexEndpoints/{INDEX_ENDPOINT_ID_CONFIG}"
        index_endpoint = aiplatform.MatchingEngineIndexEndpoint(index_endpoint_name=index_endpoint_name)
        debug_log(f"Vector Search Index Endpoint object created for configured ID: {INDEX_ENDPOINT_ID_CONFIG}")
    except Exception as e:
        debug_warn(f" Failed to create IndexEndpoint object for configured ID {INDEX_ENDPOINT_ID_CONFIG}: {e}")
        index_endpoint = None
else:
    debug_warn(" VECTOR_INDEX_ENDPOINT_ID not found in config. Endpoint object not initialized.")

gemini_model = gemini_client # Reference new client

def check_vector_search_availability() -> dict:
    status = {"endpoint_connected": False, "index_deployed": False, "error": None, "setup_needed": False}
    if not INDEX_ENDPOINT_ID_CONFIG or not VECTOR_INDEX_NAME_CONFIG or not DEPLOYED_INDEX_ID_CONFIG:
        status["error"] = "One or more Vector Search environment variables (ENDPOINT_ID, INDEX_NAME, DEPLOYED_INDEX_ID) are missing."
        status["setup_needed"] = True
        return status
    global index_endpoint
    current_endpoint = index_endpoint
    if not current_endpoint:
         try:
            index_endpoint_name = f"projects/{PROJECT_ID}/locations/{LOCATION}/indexEndpoints/{INDEX_ENDPOINT_ID_CONFIG}"
            current_endpoint = aiplatform.MatchingEngineIndexEndpoint(index_endpoint_name=index_endpoint_name)
            index_endpoint = current_endpoint
            debug_log(f"Created IndexEndpoint object during check for: {INDEX_ENDPOINT_ID_CONFIG}")
         except Exception as e:
             status["error"] = f"Failed to initialize connection object for Endpoint ID '{INDEX_ENDPOINT_ID_CONFIG}': {e}"
             return status
    try:
        debug_log(f"Attempting to access deployed indexes for Endpoint ID: {INDEX_ENDPOINT_ID_CONFIG}")
        deployed_indexes_list = current_endpoint.deployed_indexes
        status["endpoint_connected"] = True
        debug_log(f"Successfully accessed deployed_indexes property for Endpoint ID: {INDEX_ENDPOINT_ID_CONFIG}")
        found = False
        for deployed in deployed_indexes_list:
            if deployed.id == DEPLOYED_INDEX_ID_CONFIG:
                status["index_deployed"] = True
                debug_log(f"Confirmed Deployed Index ID '{DEPLOYED_INDEX_ID_CONFIG}' is present on the endpoint.")
                found = True
                break
        if not found:
             status["error"] = f"Configured Deployed Index ID '{DEPLOYED_INDEX_ID_CONFIG}' not found on Endpoint '{INDEX_ENDPOINT_ID_CONFIG}'."
             debug_error(f" {status['error']}")
             status["setup_needed"] = True
    except api_core_exceptions.NotFound:
         status["error"] = f"Index Endpoint ID '{INDEX_ENDPOINT_ID_CONFIG}' not found in GCP project/location."
         debug_error(f" {status['error']}")
         index_endpoint = None
    except Exception as e:
        error_message = f"Failed during Vector Search availability check: {e}"
        status["error"] = error_message
        debug_error(f" {error_message}")
    return status

def setup_vector_search_resources():
    results = {
        "endpoint_status": "Unknown", "endpoint_id": INDEX_ENDPOINT_ID_CONFIG, "endpoint_name": None,
        "index_status": "Unknown", "index_id": None, "index_name": VECTOR_INDEX_NAME_CONFIG,
        "deployment_status": "Unknown", "deployed_index_id": DEPLOYED_INDEX_ID_CONFIG or DEFAULT_DEPLOYED_INDEX_ID,
        "error": None, "env_vars_to_set": {}
    }
    global index_endpoint
    endpoint_display_name = DEFAULT_ENDPOINT_DISPLAY_NAME
    endpoint_filter = f'display_name="{endpoint_display_name}"'
    if index_endpoint:
        try:
            endpoint_name_filter = f'name="{index_endpoint.resource_name}"'
            existing_endpoints = aiplatform.MatchingEngineIndexEndpoint.list(filter=endpoint_name_filter)
            if existing_endpoints:
                index_endpoint = existing_endpoints[0]
                results["endpoint_name"] = index_endpoint.resource_name; results["endpoint_id"] = index_endpoint.name; results["endpoint_status"] = "Exists (Verified)"
            else: index_endpoint = None; results["endpoint_id"] = None
        except Exception as e: index_endpoint = None; results["endpoint_id"] = None
    if not index_endpoint:
        try:
            endpoints = aiplatform.MatchingEngineIndexEndpoint.list(filter=endpoint_filter)
            if endpoints:
                index_endpoint = endpoints[0]; results["endpoint_name"] = index_endpoint.resource_name; results["endpoint_id"] = index_endpoint.name
                results["endpoint_status"] = "Exists (Found by Name)"; results["env_vars_to_set"]["VECTOR_INDEX_ENDPOINT_ID"] = index_endpoint.name
            else:
                index_endpoint = aiplatform.MatchingEngineIndexEndpoint.create(display_name=endpoint_display_name, project=PROJECT_ID, location=LOCATION, public_endpoint_enabled=True)
                results["endpoint_name"] = index_endpoint.resource_name; results["endpoint_id"] = index_endpoint.name; results["endpoint_status"] = "Created"
                results["env_vars_to_set"]["VECTOR_INDEX_ENDPOINT_ID"] = index_endpoint.name; index_endpoint.wait()
        except Exception as e: results["error"] = f"Error getting/creating Index Endpoint: {e}"; return results
    index_display_name = VECTOR_INDEX_NAME_CONFIG or DEFAULT_INDEX_DISPLAY_NAME; index_filter = f'display_name="{index_display_name}"'
    try:
        indexes = aiplatform.MatchingEngineIndex.list(filter=index_filter)
        if indexes:
            vector_index = indexes[0]; results["index_name"] = vector_index.display_name; results["index_id"] = vector_index.name; results["index_status"] = "Exists"
            if not VECTOR_INDEX_NAME_CONFIG: results["env_vars_to_set"]["VECTOR_INDEX_NAME"] = vector_index.name
        else:
            vector_index = aiplatform.MatchingEngineIndex.create_tree_ah_index(display_name=index_display_name, project=PROJECT_ID, location=LOCATION, dimensions=EMBEDDING_DIMENSIONS, approximate_neighbors_count=15, distance_measure_type="DOT_PRODUCT_DISTANCE", leaf_node_embedding_count=500, leaf_nodes_to_search_percent=7)
            results["index_name"] = vector_index.display_name; results["index_id"] = vector_index.name; results["index_status"] = "Created"; results["env_vars_to_set"]["VECTOR_INDEX_NAME"] = vector_index.name
    except Exception as e: results["error"] = f"Error getting/creating Index: {e}"; return results
    deployed_index_id = DEPLOYED_INDEX_ID_CONFIG or DEFAULT_DEPLOYED_INDEX_ID
    try:
        already_deployed = False; deployed_indexes_list = index_endpoint.deployed_indexes
        for deployed in deployed_indexes_list:
            if deployed.id == deployed_index_id:
                 if deployed.index == vector_index.resource_name: results["deployment_status"] = "Already Deployed"; results["env_vars_to_set"]["VECTOR_DEPLOYED_INDEX_ID"] = deployed_index_id; already_deployed = True; break
                 else: results["deployment_status"] = "Error"; results["error"] = f"Deployed Index ID '{deployed_index_id}' exists but points to wrong index '{deployed.index}'."; return results
        if not already_deployed:
            index_endpoint.deploy_index(index=vector_index, deployed_index_id=deployed_index_id); results["deployment_status"] = "Deployment Initiated"; results["env_vars_to_set"]["VECTOR_DEPLOYED_INDEX_ID"] = deployed_index_id
    except Exception as e: results["error"] = f"Error deploying Index: {e}"; results["deployment_status"] = "Error"; return results
    if not results["error"]: results["message"] = "Vector Search setup check/initiation complete. Deployment may take time. Update .env if new IDs were generated."
    return results

from vertexai.language_models import TextEmbeddingModel
class EmbeddingError(Exception): pass

# Global variable to store the initialized model
embedding_model_client = None

@retry(stop=stop_after_attempt(config.EMBEDDING_MAX_RETRY), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
def get_text_embedding(text: str) -> list:
    global embedding_model_client
    if not text or not text.strip():
        logger.warning("Empty text provided for embedding, returning zero vector")
        return [0.0] * EMBEDDING_DIMENSIONS

    # Safeguard: Truncate text if it's excessively long for the embedding model
    # text-embedding-004 has a limit of ~3072 tokens (~12k-15k characters)
    max_chars = 12000
    if len(text) > max_chars:
        logger.info(f"Truncating text for embedding from {len(text)} to {max_chars} chars")
        text = text[:max_chars]

    logger.info(f"Attempting to generate embedding for text (length: {len(text)} chars)")
    try:
        if embedding_model_client is None:
             logger.info(f"Initializing TextEmbeddingModel: {EMBEDDING_MODEL_NAME}")
             embedding_model_client = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL_NAME)

        embeddings = embedding_model_client.get_embeddings([text])
        if embeddings and embeddings[0] and embeddings[0].values: return embeddings[0].values
        else: raise EmbeddingError("Embedding generation failed: No embedding values received from the model.")
    except (api_core_exceptions.GoogleAPICallError, api_core_exceptions.RetryError, TimeoutError) as e: logger.warning(f"Retrying embedding generation due to potentially transient error: {e}"); raise
    except EmbeddingError: raise
    except Exception as e: logger.error(f"Unexpected error during embedding generation: {e}", exc_info=True); raise EmbeddingError(f"Unexpected error during embedding generation: {e}") from e
    except Exception as e: logger.error(f"Unexpected error during embedding generation: {e}", exc_info=True); raise EmbeddingError(f"Unexpected error during embedding generation: {e}") from e

@retry(stop=stop_after_attempt(config.EMBEDDING_MAX_RETRY), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
def get_text_embeddings_batch(texts: list[str]) -> list[list]:
    global embedding_model_client
    if not texts: return []
    logger.info(f"Attempting to generate batch embeddings for {len(texts)} items")
    try:
        if embedding_model_client is None:
             logger.info(f"Initializing TextEmbeddingModel: {EMBEDDING_MODEL_NAME}")
             embedding_model_client = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL_NAME)

        # Vertex AI SDK supports batching up to 5 or more depending on model
        embeddings = embedding_model_client.get_embeddings(texts)
        return [e.values for e in embeddings]
    except (api_core_exceptions.GoogleAPICallError, api_core_exceptions.RetryError, TimeoutError) as e: logger.warning(f"Retrying batch embedding generation due to potentially transient error: {e}"); raise
    except Exception as e: logger.error(f"Unexpected error during batch embedding generation: {e}", exc_info=True); raise EmbeddingError(f"Unexpected error during batch embedding generation: {e}") from e
def find_vector_neighbors(query_embedding: list, num_neighbors: int = 40, allowed_doc_ids: Optional[list] = None, metadata_filters: Optional[dict] = None) -> list | None:
    if not INDEX_ENDPOINT_ID_CONFIG or not DEPLOYED_INDEX_ID_CONFIG: logger.error("Vector Search config missing."); return None
    try:
        neighbors = []

        # Prepare Filters/Restricts
        grpc_restricts = []
        sdk_filters = []

        # 1. Apply Doc ID filters if present
        if allowed_doc_ids:
            # gRPC
            grpc_restricts.append(aiplatform_v1.IndexDatapoint.Restriction(namespace="doc_id", allow_list=allowed_doc_ids))
            # SDK
            sdk_filters.append(Namespace(name="doc_id", allow_tokens=allowed_doc_ids))

        # 2. Apply Dynamic Metadata Filters (with normalization!)
        # Filters must be normalized the same way as during indexing
        if metadata_filters:
            from app.services.entity_normalizer import EntityNormalizer

            for key, val in metadata_filters.items():
                if val:
                    val_str = str(val)

                    # Normalize values to match how they're stored in the index
                    if key == "entity_name":
                        # Names are stored lowercase
                        val_str = EntityNormalizer.normalize_name(val_str) or val_str.lower()
                    elif key == "entity_id":
                        # IDs are stored uppercase with no special chars
                        val_str = EntityNormalizer.normalize_id(val_str) or val_str.upper()
                    elif key == "date":
                        # Dates are stored in ISO format
                        val_str = EntityNormalizer.normalize_date(val_str) or val_str
                    elif key == "doc_type":
                        # Doc types are stored uppercase
                        val_str = val_str.upper()

                    logger.info(f"Vector search filter: {key}={val_str}")

                    # gRPC
                    grpc_restricts.append(aiplatform_v1.IndexDatapoint.Restriction(namespace=key, allow_list=[val_str]))
                    # SDK
                    sdk_filters.append(Namespace(name=key, allow_tokens=[val_str]))

        if PRIVATE_ENDPOINT_IP_CONFIG:
            api_target = f"{PRIVATE_ENDPOINT_IP_CONFIG}:10000"; logger.info(f"Using MatchServiceClient with gRPC target: {api_target}")
            channel = grpc.insecure_channel(target=api_target); transport = aiplatform_v1.services.match_service.transports.MatchServiceGrpcTransport(channel=channel)
            match_client = aiplatform_v1.MatchServiceClient(transport=transport)
            index_endpoint_path = f"projects/{PROJECT_ID}/locations/{LOCATION}/indexEndpoints/{INDEX_ENDPOINT_ID_CONFIG}"

            # Construct Query with Restricts
            query_obj = aiplatform_v1.FindNeighborsRequest.Query(
                datapoint=aiplatform_v1.IndexDatapoint(feature_vector=query_embedding, restricts=grpc_restricts),
                neighbor_count=num_neighbors
            )
            request = aiplatform_v1.FindNeighborsRequest(index_endpoint=index_endpoint_path, deployed_index_id=DEPLOYED_INDEX_ID_CONFIG, queries=[query_obj], return_full_datapoint=False)
            response = match_client.find_neighbors(request=request)
            if response.nearest_neighbors and response.nearest_neighbors[0].neighbors:
                for neighbor in response.nearest_neighbors[0].neighbors:
                    if neighbor.datapoint and neighbor.datapoint.datapoint_id: neighbors.append({"id": neighbor.datapoint.datapoint_id, "distance": neighbor.distance})
        else:
            index_endpoint_name = f"projects/{PROJECT_ID}/locations/{LOCATION}/indexEndpoints/{INDEX_ENDPOINT_ID_CONFIG}"
            local_index_endpoint = aiplatform.MatchingEngineIndexEndpoint(index_endpoint_name=index_endpoint_name)
            sdk_response = local_index_endpoint.find_neighbors(
                deployed_index_id=DEPLOYED_INDEX_ID_CONFIG,
                queries=[query_embedding],
                num_neighbors=num_neighbors,
                filter=sdk_filters # Use the SDK filter list
            )
            if sdk_response and sdk_response[0]:
                for neighbor in sdk_response[0]: neighbors.append({"id": neighbor.id, "distance": neighbor.distance})
        return neighbors
    except Exception as e: logger.error(f"Exception during find_neighbors: {e}", exc_info=True); return None

def generate_direct_gemini_response(query: str, stream: bool = False):
    if not gemini_client.is_available(): 
        logger.error("Gemini model not initialized.")
        return "Sorry, AI service unavailable." if not stream else (lambda: (yield "Sorry, AI service unavailable."))()
    
    prompt = f"User Query: {query}\n\nAnswer:"
    try:
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        
        if stream:
            return gemini_client.generate_content(prompt, temperature=0.7, safety_settings=safety_settings, stream=True)
        else:
            return gemini_client.generate_content(prompt, temperature=0.7, safety_settings=safety_settings, stream=False)

    except Exception as e: 
        logger.error(f"Failed to generate direct Gemini response: {e}", exc_info=True)
        return "Sorry, error generating AI response." if not stream else (lambda: (yield "Sorry, error generating AI response."))()


def get_most_relevant_chunk_id(chunk_id_with_data, query):
    """
    Uses Gemini to find the most relevant chunk ID for a given query.

    Args:
        chunk_id_with_data: List of tuples (chunk_id, text).
        query: The query to evaluate.

    Returns:
        The most relevant chunk ID, or None.
    """
    if not chunk_id_with_data or not query:
        return None

    if not gemini_client.is_available():
        logger.error("Gemini client not initialized for relevance check")
        return None

    system_instruction = (
        "You are a helpful assistant. From the list of (chunk_id, text) pairs, select the chunk ID "
        "whose text best answers the user query. If no chunk provides any relevant information, return None. "
        "Only return the chunk ID, and nothing else."
    )

    formatted_input = "\n".join(
        [f"[{chunk_id}] {text}" for chunk_id, text in chunk_id_with_data]
    )

    prompt = f"{system_instruction}\n\nQuery: {query}\n\nChunks:\n{formatted_input}"

    try:
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        
        response_text = gemini_client.generate_content(
            prompt, 
            model_name=GEMINI_OCR_MODEL_NAME,
            temperature=0.3, 
            safety_settings=safety_settings
        )

        if not response_text:
            return None

        chunk_id_response = response_text.strip()

        # Handle empty or "None" return
        if chunk_id_response.lower() in ["none", "null", ""]:
            return None

        return chunk_id_response
    except Exception as e:
        logger.error(f"Error in get_most_relevant_chunk_id: {e}")
        return None

def generate_chat_response(query: str, context_chunks: list[str], chat_history: list = None, stream: bool = False, user_email: str = None):
    """
    Generates a chat response using Gemini API Key client.
    """
    if not gemini_client.is_available():
        logger.error("ERROR: Gemini client not initialized.")
        if stream:
            def error_stream():
                yield "Sorry, the AI chat service is currently unavailable."
            return error_stream()
        else:
            return "Sorry, the AI chat service is currently unavailable."

    # Prepare context text and chunk-to-page mapping
    context_text = ""
    chunk_page_mapping = {}  # Maps chunk_id to page range for user-friendly justification
    
    for i, chunk in enumerate(context_chunks):
        if isinstance(chunk, dict):
            c_text = chunk.get('text', '')
            c_id = chunk.get('id', 'Unknown')
            c_parent_id = chunk.get('original_doc_id', 'N/A')
            c_dist = chunk.get('distance', 'N/A')
            
            # Extract page information
            start_page = chunk.get('start_page')
            end_page = chunk.get('end_page')
            
            # Create user-friendly page display
            if start_page is not None and end_page is not None:
                if start_page == end_page:
                    page_info = f"Page {start_page}"
                else:
                    page_info = f"Pages {start_page}-{end_page}"
            else:
                page_info = "Page information unavailable"
            
            # Store mapping for later reference in justification
            chunk_page_mapping[c_id] = page_info
            
            # Helper to format metadata
            metadata_text = ""
            if chunk.get('extracted_entities'):
                ents = chunk.get('extracted_entities', {})
                metadata_parts = []
                
                # Check for grouped structure (v2.1)
                if ents.get('identifiers'): metadata_parts.append(f"Identifiers: {', '.join(map(str, ents['identifiers']))}")
                if ents.get('amounts'): metadata_parts.append(f"Amounts: {', '.join(map(str, ents['amounts']))}")
                if ents.get('dates'): metadata_parts.append(f"Dates: {', '.join(map(str, ents['dates']))}")
                if ents.get('names'): metadata_parts.append(f"Names: {', '.join(map(str, ents['names']))}")

                # Check for flat structure (v2.4)
                # Filter out internal fields and already handled lists
                flat_meta = []
                for k, v in ents.items():
                    if k.startswith('_') or k in ['identifiers', 'amounts', 'dates', 'names']:
                        continue
                    if v is not None:
                        label = k.replace('_', ' ').capitalize()
                        flat_meta.append(f"{label}: {v}")
                
                if flat_meta:
                    metadata_parts.append("\n".join(flat_meta))

                if metadata_parts:
                    metadata_text = "\nExtracted Metadata:\n" + "\n".join(metadata_parts)
            
            # Use the NEW flattened entities array if available
            if chunk.get('entities'):
                entities_list = chunk.get('entities', [])
                if entities_list:
                    metadata_text += f"\nRelevant Document Entities: {', '.join(map(str, entities_list))}"

            context_text += f"--- Chunk {i+1} ---\nRank: {i+1}\nID: {c_id}\n{page_info}\nSource Document ID: {c_parent_id}\nConfidence Score: {c_dist}\nContent:\n{c_text}{metadata_text}\n\n"
        else:
            context_text += f"--- Chunk {i+1} ---\nRank: {i+1}\n{chunk}\n\n"
    
    # Create a readable mapping for the AI to reference
    page_reference_guide = "\n".join([f"{chunk_id}: {page_info}" for chunk_id, page_info in chunk_page_mapping.items()])

    system_instruction = f"""
You are an advanced language model integrated with a retrieval system that has fetched context from relevant documents based on the user's query. Your task is to generate an answer based *only* on the retrieved context, which may include multiple document excerpts.

- **First, provide a clear, concise, and direct text answer to the user's question.** Do not start with the analysis block.
- If the answer is found in the provided context, respond clearly using only the information within the context.
- If the answer cannot be determined from the provided context, and the user is requesting a previous response (e.g., asking to "answer the previous question again"), attempt to locate the relevant part of the conversation history that can answer the current query. If such context is available, provide the answer based on that.
- If the user's query refers to a previously mentioned entity (e.g., using "he", "she", or other pronouns to refer to someone discussed earlier), use the previous context to identify the entity and answer accordingly without asking the user for clarification.
- If the provided context includes relevant details related to the query (e.g., specific information about an individual or topic), use it to provide a direct and accurate response.

The system uses the following document excerpts to answer the user's question. Each excerpt is marked with its Rank (Position), ID, Page Information, and Confidence Score (Distance).

---
{context_text}
---

CHUNK ID TO PAGE MAPPING (for your reference):
{page_reference_guide}

User Question: {query}

Formatting Instructions:
- Use Markdown headings (e.g., `## Section Title` or `### Subsection Title`) to structure longer responses.
- Use bold text (`**text**`) for emphasis or to highlight key terms.
- Use bulleted lists (`* item`) or numbered lists (`1. item`) when appropriate for clarity.
- Ensure readability by using paragraphs and line breaks correctly. For a simple line break within the same logical block of text, end the line with two spaces before the newline character. Use a blank line to separate distinct paragraphs.
- If presenting data that fits a table structure, use Markdown table syntax.

Important Notes:
- Use the context to answer as precisely as possible.
- Avoid making assumptions or fabricating answers that aren't supported by the context.
- If the user asks about an entity previously mentioned (using pronouns like "her", "him", or "they"), correctly associate the query with that entity and provide the relevant answer.
- If the context is insufficient or the user request is unclear, suggest possible refinements to the query or explain why an answer cannot be provided based on the context.

note:
    - Do not mension "MESSAGE: (No message displayed)" in the response.
    - Do not mension "like "MESSAGE" line in context with no following text)"
    - Do not mension "Based on the provided document"

BEST_MATCH_IDENTIFICATION & CITATION:
You must analyze the provided chunks to identify all relevant chunks that contributed to your answer.
Select the top 1 to 3 most relevant chunks. Do not select more than 3.
Evaluate chunks based on:
1. How well the content answers the user's question
2. How complete the information is
3. The relevance score provided
4. The position in the search results (Rank 1 is usually most relevant)

At the very end of your response, you MUST output a JSON block with your analysis in this EXACT format:

[[BEST_MATCH_ANALYSIS:
{{
  "selected_chunk_ids": ["<chunk_id_1>", "<chunk_id_2>", "<chunk_id_3>"],
  "rank": [<integer_rank_1>, <integer_rank_2>, <integer_rank_3>],
  "justification": "<user-friendly explanation in simple language>",
  "confidence_level": "<High|Medium|Low>"
}}
]]

IMPORTANT GUIDELINES FOR JUSTIFICATION:
- Write in simple, everyday language that anyone can understand
- DO NOT use technical terms like "chunk", "semantic relevance", "confidence score", "vector search", etc.
- Instead of saying "chunk 1" or "chunk 2", refer to the specific PAGE NUMBERS from the documents
- Use the CHUNK ID TO PAGE MAPPING above to find the page numbers for each chunk ID
- Explain WHY these pages were helpful in plain language
- Example of GOOD justification: "The answer was found on Page 5, which contains the complete salary information for the employee. Pages 6-7 provide additional details about benefits and deductions."
- Example of BAD justification: "Chunk 1 was selected due to high semantic relevance and optimal confidence score. The datapoint shows strong vector similarity."

If no chunk is relevant, set "selected_chunk_ids" to [].
Do NOT output the [[CITATION: ...]] tag separately. The system will use the JSON block.
Ensure you close the JSON block with "]]".
"""

    redis_client = get_redis_client()
    # Cache key calculation
    cache_data = {
        "query": query,
        "context": context_text,
        "history": chat_history,
        "user": user_email
    }
    cache_key = f"chat_response:{hashlib.sha256(json.dumps(cache_data, sort_keys=True).encode('utf-8')).hexdigest()}"

    try:
        cached_response = redis_client.get(cache_key)
        if cached_response:
            logger.info(f"CACHE HIT for chat response with key: {cache_key}")
            if stream:
                def stream_cached_response():
                    yield cached_response
                return stream_cached_response()
            else:
                return cached_response
    except Exception as e:
        logger.error(f"Redis cache get failed for chat response: {e}")

    try:
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]

        if stream:
            response_generator = gemini_client.generate_chat_response(
                message=query,
                history=chat_history,
                model_name=GEMINI_CHAT_MODEL_NAME,
                system_instruction=system_instruction,
                temperature=0.3,
                safety_settings=safety_settings,
                stream=True
            )
            
            def stream_and_cache():
                full_text = []
                for chunk in response_generator:
                    full_text.append(chunk)
                    yield chunk
                
                # Cache the full response
                try:
                    redis_client.set(cache_key, "".join(full_text), ex=86400)
                except Exception as e:
                    logger.error(f"Redis cache set failed: {e}")
            
            return stream_and_cache()
        else:
            response_text = gemini_client.generate_chat_response(
                message=query,
                history=chat_history,
                model_name=GEMINI_CHAT_MODEL_NAME,
                system_instruction=system_instruction,
                temperature=0.3,
                safety_settings=safety_settings,
                stream=False
            )
            
            try:
                redis_client.set(cache_key, response_text, ex=86400)
            except Exception as e:
                logger.error(f"Redis cache set failed: {e}")
            
            return response_text

    except GeminiConfigurationError as e:
        logger.error(f"Gemini Configuration Error: {e}")
        error_msg = str(e) or "AI Service temporarily unavailable (Configuration Error)."
        if stream:
             def config_error_stream():
                 yield error_msg
             return config_error_stream()
        else:
             return error_msg
    except Exception as e:
        logger.error(f"ERROR: Failed to generate chat response: {e}", exc_info=True)
        if stream:
             def error_stream():
                 yield "Sorry, an error occurred while generating the AI response."
             return error_stream()
        else:
             return "Sorry, an error occurred while generating the AI response."

@retry(stop=stop_after_attempt(1), wait=wait_exponential(multiplier=1, min=2, max=5), reraise=False)
def generate_query_variations(query: str) -> list[str]:
    """Generates search query variations using Gemini Flash with Redis caching."""
    import re
    t_start = time.time()
    logger.info(f"Generating search variations for: '{query}'")

    # Normalize query for better cache hit rate
    normalized_query = query.lower().strip()
    normalized_query = re.sub(r'\s+', ' ', normalized_query)

    # Redis Cache Check
    cache_key = f"query_variations:{hashlib.md5(normalized_query.encode()).hexdigest()}"
    cache_ttl = getattr(config, 'QUERY_VARIATIONS_CACHE_TTL_SECONDS', 86400)
    redis_client = None
    try:
        redis_client = get_redis_client()
        if redis_client:
             cached_variations = redis_client.get(cache_key)
             if cached_variations:
                 logger.info(f"CACHE HIT: Found cached variations for query '{query}'")
                 return json.loads(cached_variations)
    except Exception as e:
        logger.error(f"Redis cache check failed: {e}")

    if not gemini_client.is_available():
        logger.error("Gemini client not initialized for query variations")
        return []

    try:
        instruction = f"You are an AI search assistant. Generate {config.VECTOR_SEARCH_VARIATION_COUNT} different search queries based on the user's input to find relevant documents. Output ONLY a JSON array of strings. Do not explain."

        response = gemini_client.generate_content(
            [instruction, f"User Query: {query}"],
            model_name=GEMINI_OCR_MODEL_NAME,
            temperature=0.7,
            response_mime_type="application/json"
        )

        if not response or not hasattr(response, 'text') or not response.text:
            return []

        variations = json.loads(response.text)
        if isinstance(variations, list):
            try:
                if redis_client:
                    redis_client.setex(cache_key, cache_ttl, json.dumps(variations))
            except Exception as e:
                logger.error(f"Failed to cache query variations: {e}")

            logger.info(f"Generated variations in {time.time() - t_start:.2f}s: {variations}")
            return variations
        else:
            logger.warning(f"Unexpected variation format: {variations}")
            return []

    except Exception as e:
        logger.error(f"Error generating query variations: {e}")
        return []


def generate_plain_chat_response(query: str):
    if not gemini_client.is_available():
        return "Sorry, the AI chat service is currently unavailable."

    prompt = f"User Question: {query}"
    
    try:
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        
        return gemini_client.generate_content(
            prompt,
            model_name=GEMINI_CHAT_MODEL_NAME,
            temperature=0.3,
            safety_settings=safety_settings
        )
    except Exception as e:
        logger.error(f"Gemini response error: {e}", exc_info=True)
        return "Sorry, an error occurred while generating the AI response."




# call the function generate_plain_chat_response

# response = generate_plain_chat_response("Analyze the following filename and extract its primary category. The filename is: 'test_bucket_bucket_data/1099/test-user - Copy - Copy.pdf'. Based on the name, classify it into one of the following categories: [1099, CHECKS, CHILD_WELFARE_REPORTS, LEAVE_DOCUMENTS, MONTH_END_REPORTS, PAYROLL_REPORTS_N_DOCUMENTS, PENDING_FILES, PERSONNEL_FILES, TRAVEL_REPORTS, OTHER]. Return only the category name as a single string. If you cannot determine a category from the filename, return 'None'.")

# print(f"Generated response: {response}")


# Generate a normal response without chat history and no system prompt and user prompt



# def generate_chat_response(query: str, context_chunks: list[str], chat_history: list = None, stream: bool = False):
#     if not gemini_model: logger.error("Gemini model not initialized."); return "Sorry, AI service unavailable." if not stream else (lambda: (yield "Sorry, AI service unavailable."))()
#     contents = list(chat_history) if chat_history else []
#     context_text = "\n\n".join(context_chunks)
#     system_instruction_text = f"You are an advanced language model... User Question: {query}" # Truncated for brevity
#     contents.append({'role': 'user', 'parts': [{'text': system_instruction_text}]})
#     try:
#         generation_config = genai.GenerationConfig(temperature=0.3)
#         safety_settings = [{"category": c, "threshold": "BLOCK_MEDIUM_AND_ABOVE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
#         response = gemini_model.generate_content(contents=contents, generation_config=generation_config, safety_settings=safety_settings, stream=stream)
#         return response if stream else response.text
#     except Exception as e: logger.error(f"Failed to generate chat response: {e}", exc_info=True); return "Sorry, error generating AI response." if not stream else (lambda: (yield "Sorry, error generating AI response."))()

def extract_text_with_gemini(pdf_bytes: bytes, mime_type: str = "application/pdf") -> str | None:
    if not gemini_client.is_available(): 
        logger.error("Gemini client not initialized for text extraction.")
        return None
    try:
        prompt = "Extract all text content from this PDF document."
        
        # Use direct client for multimodal if wrapper doesn't support specific Part construction well enough for simple calls
        # Actually our wrapper generate_content supports prompt as string or list.
        # But for bytes we should ideally use types.Part
        
        request_parts = [
            types.Part.from_text(text=prompt),
            types.Part.from_bytes(data=pdf_bytes, mime_type=mime_type)
        ]
        
        response = gemini_client.client.models.generate_content(
            model=GEMINI_OCR_MODEL_NAME,
            contents=request_parts
        )
        
        return response.text if response.text else None
    except Exception as e: 
        debug_error(f" Failed to extract text using Gemini: {e}", exc_info=True)
        return None

@retry(stop=stop_after_attempt(config.VECTOR_UPSERT_MAX_RETRY), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
def add_embedding_to_index(datapoint_id: str, embedding: list, filename: str, start_page: int, end_page: int, gcs_uri: str, text_preview: str, doc_id: str, entity_metadata: Optional[dict] = None) -> None:
    """
    Upsert a datapoint (chunk embedding) to the Vector Search index.

    Args:
        datapoint_id: Unique ID for the datapoint (usually chunk_id)
        embedding: The embedding vector
        filename: Original filename
        start_page: Start page number
        end_page: End page number
        gcs_uri: GCS URI of the chunk PDF
        text_preview: OCR text preview
        doc_id: Parent document ID (for RBAC filtering)
        entity_metadata: Dictionary of extracted entities for this chunk (per-chunk extraction v2.0)
    """
    logger.info(f"Attempting to upsert datapoint {datapoint_id} to Vector Search")

    local_index_endpoint = None
    if INDEX_ENDPOINT_ID_CONFIG:
        try:
            index_endpoint_name = f"projects/{PROJECT_ID}/locations/{LOCATION}/indexEndpoints/{INDEX_ENDPOINT_ID_CONFIG}"
            local_index_endpoint = aiplatform.MatchingEngineIndexEndpoint(index_endpoint_name=index_endpoint_name)
        except Exception as e:
            logger.error(f"Failed to re-initialize IndexEndpoint object for upsert: {e}", exc_info=True)
            raise EmbeddingError("Failed to init endpoint for upsert")
    else:
        logger.error("VECTOR_INDEX_ENDPOINT_ID not found in config during upsert.")
        raise EmbeddingError("Missing endpoint ID for upsert")

    if not local_index_endpoint:
        logger.error("IndexEndpoint object is None after re-initialization attempt during upsert.")
        raise EmbeddingError("Endpoint object is None for upsert")
    if not VECTOR_INDEX_NAME_CONFIG:
        debug_error("VECTOR_INDEX_NAME is not configured.")
        raise EmbeddingError("Missing index name for upsert")

    try:
        # Use dict to collect values per namespace (PREVENTS DUPLICATE NAMESPACE ERROR)
        namespace_values: dict = {}

        # Base restricts (always present)
        namespace_values["filename"] = {filename or ""}
        namespace_values["start_page"] = {str(start_page or 0)}
        namespace_values["end_page"] = {str(end_page or 0)}
        namespace_values["gcs_uri"] = {gcs_uri or ""}
        namespace_values["doc_id"] = {doc_id or ""}  # Critical for RBAC

        # NOTE: text_preview removed from restricts to reduce index size
        # It's stored in Firestore for retrieval

        # Inject Dynamic Entity Metadata (per-chunk extraction v2.0)
        if entity_metadata:
            # Import normalizer for consistent value formatting
            from app.services.entity_normalizer import EntityNormalizer, map_field_to_namespace

            entity_count = 0
            for key, value in entity_metadata.items():
                # Skip internal fields (start with _)
                if key.startswith("_"):
                    continue

                if value is not None and str(value).strip():
                    # Normalize the value for consistent searching
                    normalized_value = EntityNormalizer.normalize_entity_value(value, key)

                    if normalized_value:
                        # Map field name to standardized namespace
                        namespace = map_field_to_namespace(key)

                        # Add to existing namespace set or create new (DEDUPLICATE!)
                        if namespace not in namespace_values:
                            namespace_values[namespace] = set()
                        namespace_values[namespace].add(normalized_value)
                        entity_count += 1
                        logger.debug(f"Injected entity restrict for {datapoint_id}: {namespace}={normalized_value[:50]}...")

            # Also add doc_type as a restriction if present
            doc_type = entity_metadata.get("_doc_type")
            if doc_type:
                if "doc_type" not in namespace_values:
                    namespace_values["doc_type"] = set()
                namespace_values["doc_type"].add(doc_type.upper())
                entity_count += 1

            logger.info(f"Injected {entity_count} entity restrictions for datapoint {datapoint_id}")

        # Convert dict to list format (each namespace appears EXACTLY ONCE)
        restricts_list = []
        for namespace, values in namespace_values.items():
            restricts_list.append({
                "namespace": namespace,
                "allow_list": list(values)
            })

        datapoints_payload = [{"datapoint_id": datapoint_id, "feature_vector": embedding, "restricts": restricts_list}]
        current_index_id = config.VECTOR_INDEX_NAME

        if not current_index_id:
            debug_error("VECTOR_INDEX_NAME is missing from config inside add_embedding_to_index.")
            raise EmbeddingError("Missing index name in config for upsert")

        # Check if current_index_id is a full resource name (starts with "projects/") or just an ID
        if current_index_id.startswith("projects/"):
            index = aiplatform.MatchingEngineIndex(index_name=current_index_id)
        else:
            index = aiplatform.MatchingEngineIndex(index_name=current_index_id, project=PROJECT_ID, location=LOCATION)

        index.upsert_datapoints(datapoints=datapoints_payload)
        logger.info(f"Successfully completed index.upsert_datapoints call for ID: {datapoint_id} to Index: {current_index_id} with {len(restricts_list)} restrictions")

    except (api_core_exceptions.GoogleAPICallError, api_core_exceptions.RetryError, TimeoutError) as e:
        logger.warning(f"Retrying vector upsert for datapoint {datapoint_id} due to potentially transient error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error upserting datapoint {datapoint_id} to Vector Search.", exc_info=True)
        raise EmbeddingError(f"Unexpected error upserting datapoint {datapoint_id} to Vector Search: {e}") from e

def remove_vector_datapoints(datapoint_ids: list[str]) -> Tuple[bool, Optional[str]]:
    if not datapoint_ids: return True, None
    if not VECTOR_INDEX_NAME_CONFIG: return False, "VECTOR_INDEX_NAME is not configured."
    try:
        # Check if VECTOR_INDEX_NAME_CONFIG is a full resource name (starts with "projects/") or just an ID
        if VECTOR_INDEX_NAME_CONFIG.startswith("projects/"):
            # It's a full resource name, use it without project/location parameters
            index = aiplatform.MatchingEngineIndex(index_name=VECTOR_INDEX_NAME_CONFIG)
        else:
            # It's just an ID, use it with project/location parameters
            index = aiplatform.MatchingEngineIndex(index_name=VECTOR_INDEX_NAME_CONFIG, project=PROJECT_ID, location=LOCATION)
        index.remove_datapoints(datapoint_ids=datapoint_ids)
        return True, None
    except Exception as e: return False, f"Failed to remove datapoints: {e}"

def classify_document_type_with_gemini(first_page_text: str, available_parsers: list[dict]) -> Optional[str]:
    if not gemini_model: logger.error("Gemini model not initialized for classification."); return None
    if not first_page_text: logger.warning("No text for Gemini classification."); return "NONE"
    if not available_parsers: logger.info("No parsers for Gemini classification."); return "NONE"
    parser_options_str = "\n".join([f"- \"{p['documentTypeLabel']}\"" for p in available_parsers])
    prompt = f"Analyze text: {first_page_text[:3000]}\nChoose best type from:\n{parser_options_str}\nReturn ONLY type label or \"NONE\"."
    try:
        generation_config = genai.GenerationConfig(temperature=0.1, max_output_tokens=50)
        safety_settings = [ # Using string literals as fallback
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        response = gemini_model.generate_content(prompt, generation_config=generation_config, safety_settings=safety_settings)
        if not response.candidates: logger.error(f"Gemini (text) no candidates. Feedback: {response.prompt_feedback if response.prompt_feedback else 'N/A'}"); return "NONE"
        candidate = response.candidates[0]
        finish_reason_value = candidate.finish_reason
        finish_reason_name = getattr(finish_reason_value, 'name', str(finish_reason_value)) # Robust access to name
        if finish_reason_name != 'STOP': logger.warning(f"Gemini (text) non-STOP: {finish_reason_name}. Full: {response}"); return "NONE"
        if candidate.content and candidate.content.parts and candidate.content.parts[0].text:
            chosen_label = candidate.content.parts[0].text.strip()
            valid_labels = [p['documentTypeLabel'] for p in available_parsers] + ["NONE"]
            return chosen_label if chosen_label in valid_labels else "NONE"
        logger.warning(f"Gemini (text) no usable text. Candidate: {candidate}"); return "NONE"
    except Exception as e: logger.error(f"ERROR Gemini (text) classification: {e}", exc_info=True); return None
# Removed classify_document_type_with_gemini_from_gcs_uri function

def extract_query_metadata(query: str) -> dict:
    """
    Analyzes the user query to extract specific entity filters (metadata) for Vector Search.
    Returns a dictionary of filters (e.g., {'entity_id': '123', 'doc_type': 'INVOICE'}).
    """
    if not gemini_model:
        logger.error("Gemini model not initialized for query extraction.")
        return {}

    prompt = f"""
    Analyze this search query. The user is looking for documents.
    Extract values for the following metadata fields IF they are explicitly present in the query.

    Fields to Extract:
    - "entity_name": Person Name OR Organization/Vendor/Business Name
    - "entity_id": Any specific ID (e.g., Invoice No, Tax ID, License Plate, Passport No)
    - "vehicle_tag": Vehicle VIN or Tag/Plate Number
    - "date": Any specific date (convert to YYYY-MM-DD if possible, or YYYY)
    - "doc_type": Document Type (e.g., INVOICE, CHECK, REPORT, LICENSE, FORM)

    If a field is NOT present, do NOT include it in the JSON.
    Return ONLY a valid JSON object.

    Query: "{query}"
    """

    try:
        t_start = time.time()
        response = gemini_model.generate_content(
            prompt, 
            temperature=0.1,
            response_mime_type="application/json"
        )

        if response and hasattr(response, 'text') and response.text:
            text = response.text.strip()
            # Clean fences if present
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"): text = text[4:]

            filters = json.loads(text)
            # Remove empty/null values and normalize
            cleaned_filters = {}
            for k, v in filters.items():
                if v and str(v).lower() not in ['none', 'null', 'n/a']:
                    if k in ["doc_type", "entity_id", "vehicle_tag"]:
                        cleaned_filters[k] = str(v).upper()
                    else:
                        cleaned_filters[k] = str(v)

            logger.info(f"Extracted query filters in {time.time() - t_start:.2f}s: {cleaned_filters}")
            return cleaned_filters

    except Exception as e:
        logger.error(f"Failed to extract query metadata: {e}")
        return {}

    return {}
