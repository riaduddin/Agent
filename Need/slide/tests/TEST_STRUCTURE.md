# Test Structure Organization

## 📁 Directory Structure

```
tests/
├── unit/                    # Unit tests (Python)
│   ├── test_*.py           # All test files
│   ├── diagnose_*.py       # Diagnostic scripts
│   └── fix_*.py            # Fix scripts
├── integration/            # Integration tests (empty for now)
├── e2e/                    # End-to-end tests (empty for now)
├── client/                 # Client-side tests
│   ├── *.html             # HTML test files
│   └── *.js               # JavaScript test files
├── scripts/                # Test scripts
│   ├── run_client_tests.sh
│   └── test_socketio_curl.sh
└── README.md              # Test documentation
```

## 🧪 Test Categories

### Unit Tests (`tests/unit/`)
- **Core functionality**: `test_lightweight_approach.py`, `test_simple.py`
- **Authentication**: `test_auth.py`, `test_jwt_*.py`
- **Features**: `test_clone_presentation.py`, `test_image_search.py`
- **Quality**: `test_slide_quality_verification.py`, `test_text_overflow_fix.py`
- **SocketIO**: `test_socketio_*.py`
- **WebSocket**: `test_websocket_*.py`
- **Diagnostics**: `diagnose_qdrant_issue.py`, `fix_qdrant_dimensions.py`

### Client Tests (`tests/client/`)
- **HTML**: `client_test.html`, `socketio_test.html`, etc.
- **JavaScript**: `client_test.js`

### Scripts (`tests/scripts/`)
- **Shell scripts**: `run_client_tests.sh`, `test_socketio_curl.sh`

## 🚀 Running Tests

### Unit Tests
```bash
# Run specific test
python tests/unit/test_lightweight_approach.py

# Run all unit tests
find tests/unit -name "test_*.py" -exec python {} \;
```

### Client Tests
```bash
# Run client test script
bash tests/scripts/run_client_tests.sh

# Or open HTML files directly
open tests/client/client_test.html
```

### Diagnostics
```bash
# Diagnose Qdrant issues
python tests/unit/diagnose_qdrant_issue.py

# Fix Qdrant dimensions
python tests/unit/fix_qdrant_dimensions.py
```

## 📝 Migration Summary

**Moved from root directory:**
- 70+ Python test files → `tests/unit/`
- 6 HTML test files → `tests/client/`
- 1 JavaScript test file → `tests/client/`
- 2 shell scripts → `tests/scripts/`

**Updated references in:**
- `README.md`
- `tests/README.md`
- `tests/scripts/run_client_tests.sh`
- Various documentation files

## ✅ Benefits

1. **Organized structure** - Clear separation of test types
2. **Easy navigation** - All tests in one place
3. **Scalable** - Ready for integration and e2e tests
4. **Maintainable** - Clear naming conventions
5. **Documented** - Comprehensive README files
