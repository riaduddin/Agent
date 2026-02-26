#!/usr/bin/env python3
"""
Deployment Readiness Check Script
Validates all entity extraction components before deployment.
"""

import sys
import traceback

print('=== DEPLOYMENT READINESS CHECK ===')
print()

issues = []
warnings = []

print('1. CHECKING IMPORTS...')
print('-' * 50)

try:
    from app.services.extraction_schemas import (
        EXTRACTION_SCHEMAS,
        get_schema_for_doc_type,
        get_vector_restriction_fields,
        CATEGORY_TO_SCHEMA_MAP
    )
    print('✅ extraction_schemas.py - All imports OK')
except Exception as e:
    issues.append(f'extraction_schemas.py import failed: {e}')
    print(f'❌ extraction_schemas.py - IMPORT FAILED: {e}')

try:
    from app.services.entity_normalizer import (
        EntityNormalizer,
        build_vector_restrictions,
        map_field_to_namespace
    )
    print('✅ entity_normalizer.py - All imports OK')
except Exception as e:
    issues.append(f'entity_normalizer.py import failed: {e}')
    print(f'❌ entity_normalizer.py - IMPORT FAILED: {e}')

try:
    from app.services.type_specific_extraction_service import (
        TypeSpecificExtractionService,
        extract_chunk_entities,
        get_chunk_restrictions
    )
    print('✅ type_specific_extraction_service.py - All imports OK')
except Exception as e:
    issues.append(f'type_specific_extraction_service.py import failed: {e}')
    print(f'❌ type_specific_extraction_service.py - IMPORT FAILED: {e}')

print()
print('2. CHECKING DEPENDENCIES...')
print('-' * 50)

try:
    import pdf2image
    from importlib.metadata import version as get_version
    pdf2image_version = get_version('pdf2image')
    print(f'✅ pdf2image - Version {pdf2image_version}')
except ImportError as e:
    issues.append(f'pdf2image not installed: {e}')
    print(f'❌ pdf2image - NOT INSTALLED: {e}')

try:
    from PIL import Image
    import PIL
    print(f'✅ Pillow - Version {PIL.__version__}')
except ImportError as e:
    issues.append(f'Pillow not installed: {e}')
    print(f'❌ Pillow - NOT INSTALLED: {e}')

try:
    import google.generativeai as genai
    print('✅ google-generativeai - OK')
except ImportError as e:
    issues.append(f'google-generativeai not installed: {e}')
    print(f'❌ google-generativeai - NOT INSTALLED: {e}')

print()
print('3. CHECKING SCHEMA COVERAGE...')
print('-' * 50)

categories = [
    '1099', 'CHECKS', 'CHILD_WELFARE_REPORTS', 'LEAVE_DOCUMENTS',
    'MONTH_END_REPORTS', 'OTHER', 'PAYROLL_REPORTS_N_DOCUMENTS',
    'PENDING_FILES', 'PERSONNEL_FILES', 'TRAVEL_REPORTS'
]

for cat in categories:
    schema = get_schema_for_doc_type(cat)
    if schema and 'fields' in schema:
        fields_count = len(schema['fields'])
        print(f'✅ {cat}: {fields_count} fields')
    else:
        issues.append(f'Missing schema for {cat}')
        print(f'❌ {cat}: NO SCHEMA!')

print()
print('4. CHECKING NORMALIZER FUNCTIONS...')
print('-' * 50)

# Test name normalization
test_name = EntityNormalizer.normalize_name('John Doe Jr.')
if test_name == 'john doe':
    print('✅ normalize_name - Works correctly')
else:
    warnings.append(f'normalize_name returned unexpected: {test_name}')
    print(f'⚠️  normalize_name - Unexpected result: {test_name}')

# Test ID normalization  
test_id = EntityNormalizer.normalize_id('CHK-12345')
if test_id == 'CHK12345':
    print('✅ normalize_id - Works correctly')
else:
    warnings.append(f'normalize_id returned unexpected: {test_id}')
    print(f'⚠️  normalize_id - Unexpected result: {test_id}')

# Test date normalization
test_date = EntityNormalizer.normalize_date('12/25/2024')
if test_date == '2024-12-25':
    print('✅ normalize_date - Works correctly')
else:
    warnings.append(f'normalize_date returned unexpected: {test_date}')
    print(f'⚠️  normalize_date - Unexpected result: {test_date}')

print()
print('5. CHECKING SERVICE METHODS...')
print('-' * 50)

methods = ['extract_entities_multimodal', 'get_vector_restrictions', 
           '_build_vision_prompt', '_convert_pdf_to_images', 'extract_entities']
for method in methods:
    if hasattr(TypeSpecificExtractionService, method):
        print(f'✅ {method} - Method exists')
    else:
        issues.append(f'{method} method missing!')
        print(f'❌ {method} - METHOD MISSING!')

print()
print('6. CHECKING GEMINI CONFIGURATION...')
print('-' * 50)

try:
    from app import config
    model_name = getattr(config, 'GEMINI_MODEL_NAME', None)
    if model_name:
        print(f'✅ GEMINI_MODEL_NAME = {model_name}')
    else:
        warnings.append('GEMINI_MODEL_NAME not set in config')
        print('⚠️  GEMINI_MODEL_NAME - Not found in config')
except Exception as e:
    warnings.append(f'Could not check config: {e}')
    print(f'⚠️  Config check failed: {e}')

# Check HAS_PDF2IMAGE flag
try:
    from app.services.type_specific_extraction_service import HAS_PDF2IMAGE
    if HAS_PDF2IMAGE:
        print('✅ HAS_PDF2IMAGE = True (PDF conversion available)')
    else:
        warnings.append('HAS_PDF2IMAGE is False - multimodal may fall back to text')
        print('⚠️  HAS_PDF2IMAGE = False (will use text fallback)')
except Exception as e:
    print(f'⚠️  Could not check HAS_PDF2IMAGE: {e}')

print()
print('7. CHECKING VECTOR RESTRICTION BUILDING...')
print('-' * 50)

# Test building vector restrictions
try:
    test_entities = {
        "check_number": "12345",
        "payee_name": "John Smith",
        "date": "2024-12-25",
        "_doc_type": "CHECKS"
    }
    restrictions = TypeSpecificExtractionService.get_vector_restrictions(
        entities=test_entities,
        doc_type="CHECKS",
        doc_id="test_doc_123"
    )
    print(f'✅ Vector restrictions built: {len(restrictions)} restrictions')
    for r in restrictions[:3]:
        print(f'   → {r}')
except Exception as e:
    issues.append(f'Vector restriction building failed: {e}')
    print(f'❌ Vector restriction building failed: {e}')

# Summary
print()
print('=' * 60)
print('📊 DEPLOYMENT READINESS SUMMARY')
print('=' * 60)

if not issues:
    print('🟢 STATUS: READY FOR DEPLOYMENT')
    print()
    print('All critical checks passed!')
else:
    print('🔴 STATUS: NOT READY - CRITICAL ISSUES FOUND')
    print()
    print('Critical Issues:')
    for i, issue in enumerate(issues, 1):
        print(f'  {i}. {issue}')

if warnings:
    print()
    print('⚠️  Warnings (non-blocking):')
    for i, warning in enumerate(warnings, 1):
        print(f'  {i}. {warning}')

print()
print('=' * 60)

# Exit with error code if issues found
sys.exit(1 if issues else 0)
