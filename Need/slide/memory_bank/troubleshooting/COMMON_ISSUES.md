# Common Issues & Solutions

Troubleshooting guide for frequently encountered problems.

---

## 🔐 Authentication Issues

### **Issue: "JWT_SECRET is not configured"**

**Error**:
```
HTTPException: 500 - Authentication configuration error
```

**Cause**: `JWT_SECRET` not set in `.env`

**Solution**:
```bash
# Add to .env
JWT_SECRET=your_jwt_secret_from_nodejs_service
```

**Verify**:
```bash
grep JWT_SECRET .env
```

---

### **Issue: "Invalid authentication token"**

**Error**:
```
HTTPException: 401 - Invalid authentication token
```

**Possible Causes**:
1. Wrong `JWT_SECRET` (doesn't match Node.js service)
2. Malformed JWT token
3. Token signature invalid

**Solutions**:
1. **Verify JWT_SECRET matches**:
   ```bash
   # Check Node.js service .env
   # Copy exact JWT_SECRET to Python service .env
   ```

2. **Test JWT token**:
   ```python
   import jwt
   import os
   
   token = "YOUR_TOKEN"
   secret = os.getenv("JWT_SECRET")
   
   try:
       decoded = jwt.decode(token, secret, algorithms=["HS256"])
       print("Valid token:", decoded)
   except jwt.InvalidTokenError as e:
       print("Invalid token:", e)
   ```

3. **Check token format**:
   ```
   Format: Bearer <token>
   Example: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
   ```

---

### **Issue: "User is not verified"**

**Error**:
```
HTTPException: 403 - User is not verified. Please verify your account.
```

**Cause**: `is_verified: false` in JWT payload

**Solution**: User must verify their account in the Node.js auth service

---

## 🗄️ Database Issues

### **Issue: "DATABASE_URL not found"**

**Error**:
```
KeyError: 'DATABASE_URL'
```

**Solution**:
```bash
# Add to .env
DATABASE_URL=mongodb://localhost:27017/presentation_db
```

---

### **Issue: MongoDB connection timeout**

**Error**:
```
pymongo.errors.ServerSelectionTimeoutError: localhost:27017: [Errno 111] Connection refused
```

**Solutions**:
1. **Check MongoDB is running**:
   ```bash
   # Linux/Mac
   sudo systemctl status mongod
   
   # Windows
   net start MongoDB
   
   # Docker
   docker ps | grep mongo
   ```

2. **Start MongoDB**:
   ```bash
   # Local installation
   sudo systemctl start mongod
   
   # Docker
   docker run -d -p 27017:27017 --name mongodb mongo:latest
   ```

3. **Verify connection**:
   ```bash
   mongosh
   # Should connect successfully
   ```

---

## 🔍 Qdrant Issues

### **Issue: "Connection refused" (Qdrant)**

**Error**:
```
ConnectionRefusedError: [Errno 111] Connection refused
```

**Solution**:
```bash
# Start Qdrant Docker container
docker run -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage:z \
  qdrant/qdrant
```

**Verify**:
```bash
curl http://localhost:6333/
# Should return: {"title":"qdrant - vector search engine",...}
```

---

### **Issue: "Collection not found"**

**Error**:
```
QdrantException: Collection 'browser_research' not found
```

**Solution**:
```bash
# Create collection
python tests/diagnose_qdrant_issue.py
```

Or manually:
```python
from qdrant_utils import QdrantManager

manager = QdrantManager()
# Collection created automatically on first use
```

---

### **Issue: "Wrong vector dimension"**

**Error**:
```
QdrantException: Expected 768 dimensions, got 384
```

**Cause**: Collection created with wrong dimensions

**Solution**:
```bash
# Fix dimensions (will recreate collection)
python fix_qdrant_dimensions.py
```

**Warning**: This deletes existing data!

---

### **Issue: "OutputTooSmall" (Qdrant Internal Error)**

**Error**:
```
Unexpected Response: 500 (Internal Server Error)
Service internal error: OutputTooSmall
```

**Cause**: Known Qdrant bug with filtered searches

**Status**: ✅ **Workaround in place** (Python-based filtering)

**Info**: This error should not appear anymore. If it does:
```bash
# Check qdrant_utils.py has workaround
grep "Python-based filtering" qdrant_utils.py
```

---

### **Issue: "Vectors: None" in Qdrant**

**Status**: ⚠️ **Not an error**

**Info**: Normal when:
1. Fresh installation
2. No presentations generated yet
3. Collection just created

**Verify**:
```bash
# Check collection info
curl http://localhost:6333/collections/browser_research
# "vectors_count": 0 is normal initially
```

**Vectors added**: During presentation generation (research phase)

---

## 🤖 Agent Issues

### **Issue: "Function create_slides_agent is not found"**

**Error**:
```
KeyError: 'create_slides_agent' in tools_dict
```

**Cause**: Tool definition syntax error

**Solution**: Already fixed in `qdrant_retrieval_tool.py`

**Verify**:
```python
# Check tool definition
grep "FunctionTool" qdrant_retrieval_tool.py
# Should see: name=..., description=..., func=...
```

---

### **Issue: "Context variable not found: `variable`"**

**Error**:
```
KeyError: 'Context variable not found: `slide_title`'
```

**Cause**: LLM interpreting `{{variable}}` syntax in instructions

**Status**: ✅ **Fixed** - Template syntax removed from agent instructions

**If reoccurs**: Check agent instruction doesn't use `{{...}}` syntax

---

### **Issue: "name 'margin' is not defined"**

**Error**:
```
NameError: name 'margin' is not defined
```

**Cause**: CSS code blocks in agent instructions interpreted as Python

**Status**: ✅ **Fixed** - CSS examples removed from instructions

**If reoccurs**: Remove CSS code blocks from agent instructions

---

### **Issue: "Failed to generate valid slide outline"**

**Error**:
```
ERROR: ❌ Failed to generate valid slide outline
```

**Possible Causes**:
1. Planning agent returned non-JSON response
2. JSON wrapped in markdown
3. Multiple responses without JSON

**Status**: ✅ **Fixed** - 4 fallback JSON parsing methods

**Debug**:
```python
# Check logs for actual response
grep "plan_text" logs/app.log
grep "all_responses" logs/app.log
```

---

## 🎨 Slide Rendering Issues

### **Issue: Text overflow / text cut off**

**Symptoms**:
- Text not fully visible
- Content extends beyond slide boundaries
- Font too large for container

**Status**: ✅ **Fixed** - Quality verifier detects and fixes

**Manual Fix** (if needed):
```css
/* Add to HTML style */
font-size: clamp(1.2rem, 2.5vw, 2rem);
overflow: hidden;
word-wrap: break-word;
max-width: 100%;
max-height: 600px;
```

---

### **Issue: Missing slides (N-1 instead of N)**

**Symptoms**:
- Generated 5/6 slides
- Last slide missing

**Possible Causes**:
1. Parallel generation race condition
2. Agent timeout
3. Slide counting logic error

**Status**: 🔍 **Monitoring** - Detection in place

**Debug**:
```python
# Check logs
grep "processed_generators" logs/app.log
grep "Generated slide" logs/app.log
```

---

## 🌐 Connection Issues

### **Issue: ConnectionResetError**

**Error**:
```
ConnectionResetError: [WinError 64] The specified network name is no longer available
```

**Possible Causes**:
1. Too many parallel API calls
2. Network instability
3. API rate limits

**Status**: ⚠️ **Intermittent** - Agent retries usually succeed

**Mitigation**:
```python
# Reduce parallel agents (if needed)
# In agent config
max_parallel_agents = 5  # Default: 10
```

---

### **Issue: aiohttp ClientConnectionError**

**Error**:
```
aiohttp.client_exceptions.ClientConnectionError: Connection lost
```

**Status**: ⚠️ **Minor** - Doesn't break functionality

**Info**: Cosmetic warnings from async session cleanup

**Ignore**: Low priority, doesn't affect results

---

### **Issue: "Unclosed client session" warnings**

**Warning**:
```
ERROR:asyncio:Unclosed client session
```

**Status**: ⚠️ **Known** - Low priority

**Cause**: Async session cleanup in tools

**Impact**: None (functionality works normally)

---

## 🚀 Performance Issues

### **Issue: Slow presentation generation**

**Symptoms**: Takes >3 minutes for 6 slides

**Possible Causes**:
1. Web scraping timeouts
2. Gemini API latency
3. Many parallel calls overwhelming system

**Solutions**:
1. **Check network**:
   ```bash
   ping googleapis.com
   ```

2. **Check Gemini API quota**:
   - Go to Google Cloud Console
   - Check API quotas and usage

3. **Reduce parallel workers**:
   ```python
   # In browser agent config
   max_workers = 5  # Reduce from 10
   ```

---

### **Issue: High memory usage**

**Symptoms**: Python process using >2GB RAM

**Possible Causes**:
1. Large file uploads in memory
2. Many concurrent requests
3. Qdrant vectors in memory

**Solutions**:
1. **Limit file size**:
   ```python
   MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
   ```

2. **Process files streaming**:
   ```python
   # Don't load entire file in memory
   # Use streaming upload to GCS
   ```

3. **Restart service periodically** (production)

---

## 🔧 Configuration Issues

### **Issue: "GEMINI_API_KEY not found"**

**Error**:
```
ValueError: Neither GEMINI_API_KEY nor GOOGLE_APPLICATION_CREDENTIALS is set
```

**Solution**:
```bash
# Add to .env
GEMINI_API_KEY=your_api_key

# OR
GOOGLE_APPLICATION_CREDENTIALS=service-account.json
```

---

### **Issue: Import errors after update**

**Error**:
```
ModuleNotFoundError: No module named 'qdrant_client'
```

**Solution**:
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

---

## 🧪 Testing Issues

### **Issue: Test script fails with "no module dotenv"**

**Error**:
```
ModuleNotFoundError: No module named 'dotenv'
```

**Solution**:
```bash
# Activate virtual environment first!
source venv/Scripts/activate  # Windows Git Bash
source venv/bin/activate      # Linux/Mac

# Then run test
python test_lightweight_approach.py
```

---

## 📊 Diagnostic Commands

### **Check All Services**

```bash
# MongoDB
mongosh --eval "db.runCommand({ ping: 1 })"

# Qdrant
curl http://localhost:6333/

# FastAPI
curl http://localhost:8000/health

# JWT validation
python -c "import jwt; print('PyJWT installed')"
```

### **Check Environment**

```bash
# Show .env (without secrets)
grep -v "KEY\|SECRET" .env

# Check Python version
python --version  # Should be 3.9+

# Check dependencies
pip list | grep -E "fastapi|qdrant|pymongo|jwt"
```

### **Check Logs**

```bash
# FastAPI logs (if running)
tail -f logs/app.log

# Qdrant logs (Docker)
docker logs $(docker ps | grep qdrant | awk '{print $1}')

# MongoDB logs
tail -f /var/log/mongodb/mongod.log
```

---

## 🆘 Still Need Help?

1. **Enable DEBUG logging**:
   ```python
   # In main.py
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **Run diagnostic scripts**:
   ```bash
   python tests/diagnose_qdrant_issue.py
   python tests/test_slide_issues_fix.py
   ```
   
   **See all tests**: [`tests/README.md`](../../tests/README.md)

3. **Check memory bank**:
   - `memory_bank/architecture/CURRENT_STATE.md`
   - `memory_bank/references/AGENT_CATALOG.md`

4. **Review recent changes**:
   - `memory_bank/architecture/RECENT_CHANGES.md`

---

**Related Documents**:
- `QUICK_START.md` - Setup guide
- `API_REFERENCE.md` - API errors
- `AGENT_CATALOG.md` - Agent details

