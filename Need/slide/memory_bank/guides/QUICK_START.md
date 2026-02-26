# Quick Start Guide

Get the Presentation Generation Service running in 10 minutes.

---

## 📋 Prerequisites

- Python 3.9+
- MongoDB (local or cloud)
- Qdrant (Docker recommended)
- Google Cloud account (for Gemini API)
- Node.js auth service (for JWT tokens)

---

## 🚀 Step-by-Step Setup

### **1. Clone and Navigate**

```bash
cd presentation-gen-service
```

### **2. Create Virtual Environment**

```bash
# Windows (Git Bash)
python -m venv venv
source venv/Scripts/activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### **3. Install Dependencies**

```bash
pip install -r requirements.txt
```

### **4. Start Qdrant (Docker)**

```bash
docker run -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage:z \
  qdrant/qdrant
```

**Verify Qdrant**:
```bash
curl http://localhost:6333/
# Should return: {"title":"qdrant - vector search engine",...}
```

### **5. Configure Environment**

Create `.env` file in project root:

```bash
# Copy template
cp memory_bank/guides/ENVIRONMENT_SETUP.txt .env

# Edit .env with your values
nano .env  # or use your preferred editor
```

**Minimum required**:
```env
DATABASE_URL=mongodb://localhost:27017/presentation_db
GEMINI_API_KEY=your_gemini_api_key
JWT_SECRET=your_jwt_secret_from_nodejs_service
```

### **6. Get Google API Credentials**

#### **Option A: API Key (Development)**
1. Go to https://makersuite.google.com/app/apikey
2. Create API key
3. Add to `.env`: `GEMINI_API_KEY=your_key`

#### **Option B: Service Account (Production)**
1. Go to Google Cloud Console
2. Create service account
3. Download JSON key file
4. Save as `service-account.json`
5. Add to `.env`: `GOOGLE_APPLICATION_CREDENTIALS=service-account.json`

### **7. Initialize Qdrant Collection**

```bash
python diagnose_qdrant_issue.py
```

This will:
- Test Qdrant connection
- Create `browser_research` collection (768D)
- Test embedding generation
- Verify retrieval

### **8. Start the Server**

```bash
# Development
uvicorn main:app --reload --port 8000

# Production
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### **9. Verify Server**

```bash
# Health check
curl http://localhost:8000/health

# Expected response:
# {"status": "healthy", "database": true}
```

### **10. Test with SSE**

#### **Get JWT Token**
First, get a JWT token from your Node.js auth service (login endpoint).

#### **Create Presentation**
```bash
curl -X POST http://localhost:8000/sse/presentations \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Create a 5-slide presentation about AI in healthcare"
  }'
```

This will stream SSE events showing progress.

---

## 🧪 Run Tests

```bash
# Test lightweight planning approach
python test_lightweight_approach.py

# Test Qdrant
python diagnose_qdrant_issue.py

# Test image search
python test_image_search.py

# Test quality verification
python test_slide_quality_verification.py
```

---

## 🔍 Verify Everything Works

### **1. Check Logs**
```bash
tail -f logs/app.log  # if logging to file
# or check terminal output
```

### **2. Check MongoDB**
```bash
mongosh
> use presentation_db
> db.presentations.find().limit(1)
> db.agent_logs.find().limit(1)
```

### **3. Check Qdrant**
```bash
# Collection info
curl http://localhost:6333/collections/browser_research

# Expected: {"status":"ok","result":{"vectors_count":...}}
```

### **4. Test Authentication**
```bash
curl http://localhost:8000/slides/?p_id=test_id \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"

# Should NOT return 401 Unauthorized
```

---

## 🎯 Your First Presentation

### **Using SSE Endpoint (Recommended)**

```javascript
// Frontend code
const eventSource = new EventSource(
  'http://localhost:8000/sse/presentations',
  {
    headers: {
      'Authorization': `Bearer ${jwtToken}`
    }
  }
);

eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(data);
  
  if (data.author === 'source') {
    // Research source
    console.log('Source:', data.title, data.url);
  } else if (data.text.includes('<!DOCTYPE html>')) {
    // Generated slide HTML
    console.log('Slide generated');
  }
};
```

### **Using Python**

```python
import requests

response = requests.post(
    'http://localhost:8000/sse/presentations',
    headers={'Authorization': f'Bearer {jwt_token}'},
    json={
        'message': 'Create a presentation about quantum computing'
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        print(line.decode('utf-8'))
```

---

## 🛠️ Common Issues

### **Error: "DATABASE_URL not found"**
- **Fix**: Add `DATABASE_URL` to `.env`

### **Error: "Qdrant connection refused"**
- **Fix**: Start Qdrant Docker container
- **Verify**: `curl http://localhost:6333/`

### **Error: "Invalid authentication token"**
- **Fix**: Ensure `JWT_SECRET` in `.env` matches your Node.js service
- **Verify**: Check JWT token is valid

### **Error: "Collection not found"**
- **Fix**: Run `python diagnose_qdrant_issue.py` to create collection

### **Warning: "Vectors: 0"**
- **Status**: Normal on first run
- **Info**: Vectors are added during presentation generation

### **Error: "GEMINI_API_KEY not found"**
- **Fix**: Add `GEMINI_API_KEY` or `GOOGLE_APPLICATION_CREDENTIALS` to `.env`

---

## 📚 Next Steps

1. ✅ **Read**: `memory_bank/architecture/SYSTEM_OVERVIEW.md`
2. ✅ **Review**: `memory_bank/references/AGENT_CATALOG.md`
3. ✅ **Explore**: `memory_bank/references/API_REFERENCE.md`
4. ✅ **Configure**: `memory_bank/guides/QDRANT_SETUP.md`
5. ✅ **Customize**: Modify agent instructions for your use case

---

## 🎓 Learning Path

### **Beginner**
1. Understand SSE flow
2. Test with simple queries
3. Explore generated HTML
4. Read agent catalog

### **Intermediate**
1. Customize agent instructions
2. Add new tools
3. Modify slide templates
4. Implement caching

### **Advanced**
1. Create new agents
2. Implement custom pipelines
3. Optimize performance
4. Scale horizontally

---

## 🆘 Getting Help

1. **Check**: `memory_bank/troubleshooting/COMMON_ISSUES.md`
2. **Read**: `memory_bank/architecture/CURRENT_STATE.md`
3. **Test**: Run diagnostic scripts
4. **Debug**: Enable DEBUG logging in `.env`

---

## ✅ Success Checklist

- [ ] Virtual environment activated
- [ ] Dependencies installed
- [ ] MongoDB running
- [ ] Qdrant running
- [ ] `.env` configured with all required variables
- [ ] Qdrant collection created
- [ ] Server starts without errors
- [ ] Health check passes
- [ ] Test presentation generates successfully
- [ ] Authentication works with JWT token

---

**Estimated Setup Time**: 10-15 minutes  
**Difficulty**: Easy  
**Support**: See troubleshooting guide for common issues

🎉 **Congratulations!** Your presentation service is ready!

