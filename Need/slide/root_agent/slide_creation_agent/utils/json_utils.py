import re
import json

def extract_valid_json(text: str) -> str:
    """
    Extract a valid JSON object from a string, handling markdown-style wrapping.
    Includes a heuristic attempt to repair truncated JSON at the end, specifically
    when a string value or overall structure is cut off.
    """
    text = text.strip()
    # Remove markdown-style wrapping
    text = re.sub(r"^```json|^```|```$", "", text, flags=re.MULTILINE).strip()

    # Find the outermost JSON object. This regex is greedy and will capture from the first
    # '{' to the last '}' in the provided `text`. If the text itself is truncated,
    # it won't magically find non-existent closing braces.
    match = re.search(r'{.*}', text, re.DOTALL)
    
    if match:
        extracted_json = match.group(0)
        
        # --- Heuristic for common LLM truncation issues ---
        
        # 1. Attempt to close an unclosed string at the very end
        # Check for an odd number of quotes AND if the last char is not a quote
        # (meaning the quote *was* there but the string was cut *after* it,
        # or the value itself was incomplete without a final quote).
        if extracted_json.count('"') % 2 != 0 and (not extracted_json.endswith('"')):
            # This covers cases like "key": "value part" (missing final quote)
            # or "key": "value " (ending with space, but missing quote)
            extracted_json += '"'
        
        # 2. Ensure structural completeness by adding missing closing braces and brackets
        open_braces = extracted_json.count('{')
        close_braces = extracted_json.count('}')
        open_brackets = extracted_json.count('[')
        close_brackets = extracted_json.count(']')

        # Add missing closing braces and brackets
        extracted_json += '}' * max(0, open_braces - close_braces)
        extracted_json += ']' * max(0, open_brackets - close_brackets)
        
        return extracted_json
    return None

def fix_multiline_body_content(text: str) -> str:
    """
    Fix common multiline bullet issues under "body_content".
    This pattern catches a "body_content": "line", followed by multiple list-like bullets without keys.
    """
    pattern = re.compile(
        r'"body_content"\s*:\s*"([^"]*?)",\s*\n("•[^"]*?",\s*\n)+',
        re.MULTILINE
    )

    def replacer(match):
        # Extract initial line and rest of bullets
        initial = match.group(1)
        bullets = re.findall(r'"(•.*?)"', match.group(0))
        combined = "\\n".join([initial] + bullets)
        return f'"body_content": "{combined}",'

    return pattern.sub(replacer, text)

def sanitize_and_parse_json(s: str) -> dict:
    """
    Sanitize and parse an LLM-generated JSON string, handling markdown, newlines, and malformed bullets.
    """
    s = s.strip()

    # Remove markdown-style wrapping
    s = re.sub(r"^```json|^```|```$", "", s.strip(), flags=re.MULTILINE).strip()

    # Fix common multiline bullet issues
    s = fix_multiline_body_content(s)

    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        # In production, you might want to log this properly
        # print("❌ Failed to parse JSON.")
        # print("Input snippet:", repr(s[:300])) 
        raise e
