# Tests & Diagnostic Scripts

This folder contains all test scripts and diagnostic tools for the Presentation Generation Service.

---

## 🧪 Test Scripts

### **Core Functionality Tests**

#### `test_lightweight_approach.py`
- **Purpose**: Test the lightweight planning approach
- **Tests**: Planning agent → Parallel slide generation
- **Usage**: `python tests/unit/test_lightweight_approach.py`

#### `test_simple.py`
- **Purpose**: Basic functionality tests
- **Tests**: Simple agent execution
- **Usage**: `python tests/unit/test_simple.py`

#### `testing.py`
- **Purpose**: General testing utilities
- **Tests**: Various components
- **Usage**: `python tests/testing.py`

---

### **Feature-Specific Tests**

#### `test_auth.py`
- **Purpose**: Test JWT authentication
- **Tests**: Token validation, user extraction
- **Usage**: `python tests/unit/test_auth.py`

#### `test_clone_presentation.py`
- **Purpose**: Test presentation cloning feature
- **Tests**: Clone endpoint, data duplication
- **Usage**: `python tests/unit/test_clone_presentation.py`

#### `test_image_search.py`
- **Purpose**: Test image search tool
- **Tests**: Image API connectivity, search results
- **Usage**: `python tests/unit/test_image_search.py`

#### `test_template_selection_flow.py`
- **Purpose**: Test template selection feature
- **Tests**: Template detection, application
- **Usage**: `python tests/unit/test_template_selection_flow.py`

---

### **Quality & Rendering Tests**

#### `test_slide_quality_verification.py`
- **Purpose**: Test slide quality verifier agent
- **Tests**: Quality analysis, HTML enhancement
- **Usage**: `python tests/unit/test_slide_quality_verification.py`

#### `test_text_overflow_fix.py`
- **Purpose**: Test text overflow detection and fixes
- **Tests**: Responsive CSS, overflow handling
- **Usage**: `python tests/unit/test_text_overflow_fix.py`

#### `test_slide_issues_fix.py`
- **Purpose**: Comprehensive slide issue testing
- **Tests**: Text overflow, missing slides, rendering
- **Usage**: `python tests/unit/test_slide_issues_fix.py`

#### `test_slide_validation.py`
- **Purpose**: Test slide validation logic
- **Tests**: HTML validation, structure checks
- **Usage**: `python tests/unit/test_slide_validation.py`

---

## 🔍 Diagnostic Scripts

### **Qdrant Diagnostics**

#### `diagnose_qdrant_issue.py`
- **Purpose**: Diagnose Qdrant connectivity and configuration
- **Checks**:
  - Qdrant connection
  - Collection existence
  - Embedding generation
  - Retrieval functionality
  - Python-based filtering workaround
- **Usage**: `python tests/diagnose_qdrant_issue.py`
- **When to Use**: Qdrant connection errors, vector issues

#### `fix_qdrant_dimensions.py`
- **Purpose**: Fix Qdrant collection dimension mismatches
- **Actions**:
  - Check current dimensions
  - Delete and recreate collection (768D)
  - Test new configuration
- **Usage**: `python tests/fix_qdrant_dimensions.py`
- **Warning**: ⚠️ Deletes existing vectors!
- **When to Use**: Dimension mismatch errors (384 vs 768)

---

## 🚀 Running Tests

### **Run All Tests**
```bash
# Run individual tests
python tests/unit/test_lightweight_approach.py
python tests/unit/test_slide_quality_verification.py
python tests/unit/test_image_search.py
```

### **Run Diagnostics**
```bash
# Diagnose Qdrant
python tests/unit/diagnose_qdrant_issue.py

# Fix dimensions (careful!)
python tests/unit/fix_qdrant_dimensions.py
```

### **Prerequisites**

Before running tests:
1. ✅ Virtual environment activated
2. ✅ `.env` file configured
3. ✅ MongoDB running
4. ✅ Qdrant running (for Qdrant tests)
5. ✅ Dependencies installed: `pip install -r requirements.txt`

---

## 📋 Test Categories

### **By Type**

| Category | Tests |
|----------|-------|
| **Core** | `test_lightweight_approach.py`, `test_simple.py`, `testing.py` |
| **Features** | `test_auth.py`, `test_clone_presentation.py`, `test_image_search.py`, `test_template_selection_flow.py` |
| **Quality** | `test_slide_quality_verification.py`, `test_text_overflow_fix.py`, `test_slide_issues_fix.py`, `test_slide_validation.py` |
| **Diagnostics** | `diagnose_qdrant_issue.py`, `fix_qdrant_dimensions.py` |

### **By Priority**

#### **High Priority** (Run First)
1. `diagnose_qdrant_issue.py` - Verify infrastructure
2. `test_lightweight_approach.py` - Test core functionality
3. `test_slide_quality_verification.py` - Test quality system

#### **Medium Priority**
4. `test_auth.py` - Verify authentication
5. `test_image_search.py` - Test external API
6. `test_slide_issues_fix.py` - Comprehensive checks

#### **Low Priority** (Feature-specific)
7. `test_clone_presentation.py`
8. `test_template_selection_flow.py`
9. `test_slide_validation.py`

---

## 🐛 Troubleshooting Tests

### **Common Test Errors**

#### "ModuleNotFoundError: No module named 'dotenv'"
```bash
# Activate virtual environment first
source venv/Scripts/activate  # Windows Git Bash
source venv/bin/activate      # Linux/Mac
```

#### "Qdrant connection refused"
```bash
# Start Qdrant Docker
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

#### "DATABASE_URL not found"
```bash
# Create .env file
cp memory_bank/guides/ENVIRONMENT_SETUP.txt .env
# Edit .env with your values
```

#### "GEMINI_API_KEY not found"
```bash
# Add to .env
echo "GEMINI_API_KEY=your_key" >> .env
```

---

## 📝 Adding New Tests

When creating new tests:

1. **Name Convention**: `test_<feature_name>.py`
2. **Location**: Place in `tests/` folder
3. **Documentation**: Add entry to this README
4. **Dependencies**: Keep minimal, use existing utilities
5. **Cleanup**: Clean up resources after tests

### **Test Template**

```python
"""
Test for <feature name>

Purpose: What this test verifies
Usage: python tests/test_<feature>.py
"""
import asyncio
from dotenv import load_dotenv

load_dotenv()

async def test_feature():
    """Test the feature."""
    print("🧪 Testing <feature>...")
    
    # Test logic here
    
    print("✅ Test passed!")

if __name__ == "__main__":
    asyncio.run(test_feature())
```

---

## 📊 Test Coverage

Current test coverage:

- ✅ **Lightweight Planning** - Full coverage
- ✅ **Quality Verification** - Full coverage
- ✅ **Qdrant Operations** - Full coverage
- ✅ **Authentication** - Basic coverage
- ✅ **Image Search** - Basic coverage
- ⚠️ **SSE Streaming** - Limited coverage
- ⚠️ **File Upload** - No tests
- ⚠️ **MongoDB Operations** - No tests

---

## 🎯 Testing Best Practices

1. **Always activate venv** before running tests
2. **Run diagnostics first** to verify infrastructure
3. **Clean up after tests** (delete test data)
4. **Use .env for config** (don't hardcode values)
5. **Log test progress** (helpful for debugging)
6. **Handle errors gracefully** (tests shouldn't crash)

---

## 📖 Related Documentation

- **Setup**: `memory_bank/guides/QUICK_START.md`
- **Troubleshooting**: `memory_bank/troubleshooting/COMMON_ISSUES.md`
- **Agent Details**: `memory_bank/references/AGENT_CATALOG.md`
- **API Reference**: `memory_bank/references/API_REFERENCE.md`

---

**Total Tests**: 13 files  
**Last Updated**: October 21, 2025  
**Maintained By**: Development Team

