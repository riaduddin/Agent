# Presentation Generation Service

AI-powered presentation generation service using Google Vertex AI Agent Builder, featuring multi-agent architecture, vector search, and real-time SSE streaming.

## 🚀 Quick Start

**New to this project?** Start here:

1. **Overview**: [`memory_bank/MEMORY_BANK_GUIDE.md`](memory_bank/MEMORY_BANK_GUIDE.md) - Complete memory bank guide
2. **Navigation**: [`memory_bank/INDEX.md`](memory_bank/INDEX.md) - Documentation index
3. **Setup**: [`memory_bank/guides/QUICK_START.md`](memory_bank/guides/QUICK_START.md) - Get running in 10 minutes
4. **Architecture**: [`memory_bank/architecture/SYSTEM_OVERVIEW.md`](memory_bank/architecture/SYSTEM_OVERVIEW.md) - System design

## 📚 Documentation

### 🏗️ Architecture Documentation (WebSocket Implementation)

**NEW!** Comprehensive architecture documentation for the WebSocket-based real-time system:

- **[ARCHITECTURE_INDEX.md](ARCHITECTURE_INDEX.md)** - 📍 **START HERE** - Complete architecture documentation index
- **[architecture_diagram.html](architecture_diagram.html)** - 🎨 **Interactive visual diagram** (open in browser)
- **[ARCHITECTURE_SUMMARY.md](ARCHITECTURE_SUMMARY.md)** - ⚡ Quick reference guide
- **[ARCHITECTURE_SIMPLE.md](ARCHITECTURE_SIMPLE.md)** - 📄 Simplified architecture overview
- **[ARCHITECTURE_DIAGRAM.md](ARCHITECTURE_DIAGRAM.md)** - 📚 Complete technical documentation

### 📖 Memory Bank (Legacy Documentation)

All original project documentation is organized in the **Memory Bank**:

```
memory_bank/
├── README.md                    # Documentation overview
├── INDEX.md                     # Complete navigation index ⭐ START HERE
│
├── architecture/                # System design & current state
│   ├── SYSTEM_OVERVIEW.md      # Architecture, tech stack, flows
│   ├── CURRENT_STATE.md        # Latest status, known issues
│   ├── FULL_HISTORY.md         # Complete change history
│   └── RECENT_CHANGES.md       # Recent updates log
│
├── guides/                      # Setup & configuration
│   ├── QUICK_START.md          # 10-minute setup guide
│   ├── AUTHENTICATION.md       # JWT authentication setup
│   └── ENVIRONMENT_SETUP.txt   # .env template
│
├── references/                  # Technical documentation
│   ├── AGENT_CATALOG.md        # All 20+ agents & tools
│   ├── API_REFERENCE.md        # Complete API docs
│   └── CODE_PATTERNS.md        # Best practices & conventions
│
├── troubleshooting/             # Problem solving
│   └── COMMON_ISSUES.md        # 30+ errors & solutions
│
└── schemas/                     # Data structures
    └── DATABASE_SCHEMA.md      # MongoDB & Qdrant schemas
```

## 🎯 Key Features

- ✅ **Real-Time WebSocket Streaming** - Live updates with multi-worker support via Redis Pub/Sub
- ✅ **Multi-Agent AI System** - 20+ specialized agents for research, planning, generation
- ✅ **Vector Search** - Qdrant with Gemini embeddings (768D) for semantic retrieval
- ✅ **JWT Authentication** - Secure user authentication and multi-tenancy
- ✅ **Distributed Locking** - Redis-based coordination for multi-worker deployments
- ✅ **Reconnection Support** - Seamless reconnection with event backfilling
- ✅ **SSE Streaming** - Real-time progress updates to frontend
- ✅ **Quality Verification** - Automated slide quality checks and fixes
- ✅ **Parallel Processing** - Fast generation with concurrent agents
- ✅ **Lightweight Planning** - Efficient outline-based approach
- ✅ **Google Search Integration** - Enhanced context and research

## 🛠️ Tech Stack

- **Backend**: FastAPI (Python 3.9+)
- **AI**: Google Vertex AI Agent Builder, Gemini 2.5 Flash
- **Vector DB**: Qdrant (Docker)
- **Database**: MongoDB
- **Embeddings**: Google text-embedding-004 (768D)
- **Authentication**: JWT (PyJWT)
- **Cloud Storage**: Google Cloud Storage (GCS)

## ⚙️ Setup

### Prerequisites
- Python 3.9+
- MongoDB
- Qdrant (Docker)
- Google Cloud account (Gemini API)
- Node.js auth service (for JWT tokens)

### Quick Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start Qdrant
docker run -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage:z \
  qdrant/qdrant

# 3. Configure environment
cp memory_bank/guides/ENVIRONMENT_SETUP.txt .env
# Edit .env with your values

# 4. Run server
uvicorn main:app --reload --port 8000
```

**For detailed setup**, see [`memory_bank/guides/QUICK_START.md`](memory_bank/guides/QUICK_START.md)

## 📡 API

### Main Endpoints

- `POST /sse/presentations` - Create presentation (SSE stream)
- `GET /slides/?p_id={id}` - Get presentation slides
- `POST /upload` - Upload files (PDF, DOCX, TXT)
- `GET /health` - Health check

**For complete API documentation**, see [`memory_bank/references/API_REFERENCE.md`](memory_bank/references/API_REFERENCE.md)

## 🧪 Testing

Run tests and diagnostics:

```bash
# Diagnose infrastructure
python tests/diagnose_qdrant_issue.py

# Test core functionality
python tests/unit/test_lightweight_approach.py

# Test quality verification
python tests/unit/test_slide_quality_verification.py
```

**For all tests**, see [`tests/README.md`](tests/README.md)

## 🐛 Troubleshooting

Having issues? Check:

1. [`memory_bank/troubleshooting/COMMON_ISSUES.md`](memory_bank/troubleshooting/COMMON_ISSUES.md) - 30+ solved problems
2. [`memory_bank/architecture/CURRENT_STATE.md`](memory_bank/architecture/CURRENT_STATE.md) - Known issues section
3. Run diagnostic scripts: `python tests/diagnose_qdrant_issue.py`

## 🤖 For AI Assistants

When resuming work on this project:

1. **Always read** [`memory_bank/architecture/CURRENT_STATE.md`](memory_bank/architecture/CURRENT_STATE.md) first
2. **Check** relevant docs for technical details
3. **Update** [`memory_bank/architecture/RECENT_CHANGES.md`](memory_bank/architecture/RECENT_CHANGES.md) after changes

## 📖 Learn More

| Topic | Document |
|-------|----------|
| **Getting Started** | [QUICK_START.md](memory_bank/guides/QUICK_START.md) |
| **System Architecture** | [SYSTEM_OVERVIEW.md](memory_bank/architecture/SYSTEM_OVERVIEW.md) |
| **All Agents** | [AGENT_CATALOG.md](memory_bank/references/AGENT_CATALOG.md) |
| **API Reference** | [API_REFERENCE.md](memory_bank/references/API_REFERENCE.md) |
| **Code Patterns** | [CODE_PATTERNS.md](memory_bank/references/CODE_PATTERNS.md) |
| **Database Schema** | [DATABASE_SCHEMA.md](memory_bank/schemas/DATABASE_SCHEMA.md) |
| **Authentication** | [AUTHENTICATION.md](memory_bank/guides/AUTHENTICATION.md) |

## 📊 Project Status

**Status**: ✅ Production Ready  
**Version**: 2.0 (Lightweight Planning + Qdrant Integration)  
**Last Updated**: October 21, 2025

**Current State**: See [`memory_bank/architecture/CURRENT_STATE.md`](memory_bank/architecture/CURRENT_STATE.md)

## 📝 License

[Your License Here]

## 🤝 Contributing

1. Read [`memory_bank/references/CODE_PATTERNS.md`](memory_bank/references/CODE_PATTERNS.md)
2. Follow coding conventions
3. Update [`memory_bank/architecture/RECENT_CHANGES.md`](memory_bank/architecture/RECENT_CHANGES.md)
4. Test thoroughly

---

**Need Help?**
- 📖 **New here?** Read [`memory_bank/MEMORY_BANK_GUIDE.md`](memory_bank/MEMORY_BANK_GUIDE.md)
- 📚 **Find docs**: Browse [`memory_bank/INDEX.md`](memory_bank/INDEX.md)
- 🐛 **Issues?** Check [`memory_bank/troubleshooting/COMMON_ISSUES.md`](memory_bank/troubleshooting/COMMON_ISSUES.md)



