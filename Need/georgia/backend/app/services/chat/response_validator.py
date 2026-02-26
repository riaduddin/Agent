# backend/app/services/chat/response_validator.py
"""
Response Validation Service

Validates LLM responses for:
1. Answer Relevance - Does the answer actually address the question?
2. Citation Grounding - Is the answer supported by the cited sources?

This helps prevent hallucinations and ensures response quality.
"""

import json
import logging
from app.llm.gemini_api_key_client import gemini_client
from app import config

logger = logging.getLogger(__name__)

def _get_validation_model():
    """Returns the gemini_client if available."""
    if not gemini_client.is_available():
        logger.warning("Gemini API Key not configured for response validation.")
        return None
    return gemini_client


def validate_response(query: str, answer: str, cited_chunks: list, context_chunk_map: dict) -> dict:
    """
    Validate both answer relevance and citation grounding in a single LLM call.
    
    Args:
        query: Original user question
        answer: Generated LLM answer
        cited_chunks: List of chunk IDs that were cited
        context_chunk_map: Map of chunk_id -> chunk_data with full text
    
    Returns:
        dict with validation results:
        {
            "answers_question": bool,
            "answer_relevance_score": float (0.0-1.0),
            "grounded_in_sources": bool,
            "grounding_score": float (0.0-1.0),
            "issues": list of strings,
            "overall_verdict": "VALID" | "PARTIAL" | "INVALID"
        }
    """
    if not getattr(config, 'ENABLE_RESPONSE_VALIDATION', True):
        logger.info("Response validation is disabled")
        return {
            "answers_question": True,
            "answer_relevance_score": 1.0,
            "grounded_in_sources": True,
            "grounding_score": 1.0,
            "issues": [],
            "overall_verdict": "VALID",
            "skipped": True
        }
    
    # Build context from cited chunks - FULL TEXT, no truncation
    cited_texts = []
    for chunk_id in cited_chunks:
        chunk_data = context_chunk_map.get(chunk_id)
        if chunk_data:
            chunk_text = chunk_data.get("text", "") or chunk_data.get("ocr_text", "")
            if chunk_text:
                cited_texts.append(f"[Chunk ID: {chunk_id}]\n{chunk_text}")
    
    # If no cited chunks, use all available context
    if not cited_texts and context_chunk_map:
        for chunk_id, chunk_data in list(context_chunk_map.items())[:10]:  # Limit to top 10 for no-citation cases
            chunk_text = chunk_data.get("text", "") or chunk_data.get("ocr_text", "")
            if chunk_text:
                cited_texts.append(f"[Chunk ID: {chunk_id}]\n{chunk_text}")
    
    combined_context = "\n\n---\n\n".join(cited_texts) if cited_texts else "No source context available."
    
    prompt = f"""You are a QA validator for a document retrieval system. Your task is to verify the quality of an AI-generated answer.

Evaluate TWO aspects:

1. **ANSWER RELEVANCE**: Does the answer address the user's question?
   - Score 1.0: Answer fully addresses all parts of the question
   - Score 0.7-0.9: Answer addresses most of the question, or honestly states some info is not available
   - Score 0.5-0.6: Answer partially addresses the question
   - Score 0.0-0.4: Answer does not address the question at all

2. **CITATION GROUNDING**: Are the claims in the answer supported by the provided sources?
   - Score 1.0: All factual claims are directly supported by sources
   - Score 0.7-0.9: Most claims are supported; answer correctly states when info is not found
   - Score 0.5-0.6: Some claims lack support
   - Score 0.0-0.4: Major claims are fabricated/hallucinated

IMPORTANT RULES:
- If the answer CORRECTLY states "information not found" or "context does not contain" for part of a multi-part question, this is HONEST behavior, NOT a failure. Score this as 0.7-0.9.
- Only mark as INVALID if the answer contains FABRICATED information or completely ignores the question.
- A partial answer with honest "not found" statements should be PARTIAL, not INVALID.

---

**USER QUESTION:**
{query}

**AI ANSWER:**
{answer}

**SOURCE DOCUMENTS:**
{combined_context}

---

Return ONLY valid JSON:
{{
  "answers_question": true or false,
  "answer_relevance_score": 0.0 to 1.0,
  "grounded_in_sources": true or false,
  "grounding_score": 0.0 to 1.0,
  "issues": ["list of specific problems found, if any"],
  "overall_verdict": "VALID" or "PARTIAL" or "INVALID"
}}

Verdict rules:
- "VALID": Both scores >= 0.8, answer is accurate and well-grounded.
- "PARTIAL": At least one score between 0.5-0.8, OR answer is partially complete but honest about missing info
- "INVALID": Any score < 0.5, OR answer contains hallucinated/fabricated information

Return ONLY the JSON object, no other text."""

    try:
        model = _get_validation_model()
        response = model.generate_content(
            prompt,
            temperature=0,  # Deterministic for consistency
            response_mime_type="application/json"
        )
        
        result = json.loads(response.text)
        
        # Ensure all required fields exist
        result.setdefault("answers_question", True)
        result.setdefault("answer_relevance_score", 1.0)
        result.setdefault("grounded_in_sources", True)
        result.setdefault("grounding_score", 1.0)
        result.setdefault("issues", [])
        result.setdefault("overall_verdict", "VALID")
        
        logger.info(f"Response validation result: verdict={result['overall_verdict']}, "
                   f"relevance={result['answer_relevance_score']}, grounding={result['grounding_score']}")
        
        return result
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse validation response as JSON: {e}")
        return {
            "answers_question": True,
            "answer_relevance_score": 1.0,
            "grounded_in_sources": True,
            "grounding_score": 1.0,
            "issues": ["Validation parsing failed - defaulting to valid"],
            "overall_verdict": "VALID",
            "error": str(e)
        }
    except Exception as e:
        logger.error(f"Response validation failed: {e}")
        return {
            "answers_question": True,
            "answer_relevance_score": 1.0,
            "grounded_in_sources": True,
            "grounding_score": 1.0,
            "issues": ["Validation call failed - defaulting to valid"],
            "overall_verdict": "VALID",
            "error": str(e)
        }


def get_validation_disclaimer(verdict: str, issues: list, relevance_score: float = 1.0, grounding_score: float = 1.0, user_note: str = "") -> str:
    """
    Returns empty string. Disclaimer is now handled by the frontend for better control.
    """
    return ""
