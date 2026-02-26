"""
Slide Modification Agent V2
Content-only modification system with structure preservation

Architecture:
1. Request Parser - Parses natural language into structured modification requests
2. Content Modifier - LLM agent that modifies content while preserving structure
3. Single Slide Modifier - Orchestrates modification of one slide with validation
4. Multi-Slide Orchestrator - Handles single or multiple slide modifications

Safety Features:
- Structure validation before saving
- Raw HTML storage (no prettifying)
- Content-only modifications
- Research integration for data updates
"""

from .multi_slide_orchestrator import multi_slide_modification_orchestrator
from .single_slide_modifier import single_slide_modifier
from .request_parser import modification_request_parser
from .validation_utils import validate_structure, extract_content_structure

__all__ = [
    "multi_slide_modification_orchestrator",
    "single_slide_modifier",
    "modification_request_parser",
    "validate_structure",
    "extract_content_structure"
]


