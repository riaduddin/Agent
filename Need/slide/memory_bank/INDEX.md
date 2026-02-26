# Memory Bank Index

Quick navigation for all memory bank documentation.

---

## 🚀 Getting Started

**New to this project?** Start here:

1. **[README.md](README.md)** - Memory bank overview
2. **[architecture/SYSTEM_OVERVIEW.md](architecture/SYSTEM_OVERVIEW.md)** - Understand the system
3. **[guides/QUICK_START.md](guides/QUICK_START.md)** - Setup in 10 minutes
4. **[references/AGENT_CATALOG.md](references/AGENT_CATALOG.md)** - Learn about agents

---

## 📁 Complete File Index

### **Getting Started**

| File | Description | When to Read |
|------|-------------|--------------|
| [README.md](README.md) | Memory bank overview | First time |
| [INDEX.md](INDEX.md) | This file - complete navigation | Always |
| [MEMORY_BANK_GUIDE.md](MEMORY_BANK_GUIDE.md) | Complete guide to using memory bank | First time, onboarding |

### **Architecture** (`architecture/`)

Core system design and current state documentation.

| File | Description | When to Read |
|------|-------------|--------------|
| [SYSTEM_OVERVIEW.md](architecture/SYSTEM_OVERVIEW.md) | High-level architecture, agent types, data flow | First time, onboarding |
| [CURRENT_STATE.md](architecture/CURRENT_STATE.md) | Latest status, known issues, priorities | Context switches, debugging |
| [FULL_HISTORY.md](architecture/FULL_HISTORY.md) | Complete change history, all features | Understanding evolution |
| [RECENT_CHANGES.md](architecture/RECENT_CHANGES.md) | Change log (update frequently) | After changes |
| [CLEANUP_SUMMARY.md](architecture/CLEANUP_SUMMARY.md) | Documentation cleanup details (71 files removed) | Reference |
| [TESTS_ORGANIZATION.md](architecture/TESTS_ORGANIZATION.md) | Test folder organization details | Reference |

### **Guides** (`guides/`)

Setup, configuration, and deployment instructions.

| File | Description | When to Use |
|------|-------------|-------------|
| [QUICK_START.md](guides/QUICK_START.md) | 10-minute setup guide | Initial setup |
| [AUTHENTICATION.md](guides/AUTHENTICATION.md) | JWT authentication setup | Setting up auth |
| [ENVIRONMENT_SETUP.txt](guides/ENVIRONMENT_SETUP.txt) | .env file template | Configuration |

### **References** (`references/`)

Detailed technical references and API documentation.

| File | Description | When to Use |
|------|-------------|-------------|
| [AGENT_CATALOG.md](references/AGENT_CATALOG.md) | All agents, tools, purposes | Understanding agents, creating new ones |
| [API_REFERENCE.md](references/API_REFERENCE.md) | Complete API documentation | Frontend integration, testing |
| [CODE_PATTERNS.md](references/CODE_PATTERNS.md) | Best practices, code conventions | Writing new code, reviewing PRs |

### **Troubleshooting** (`troubleshooting/`)

Problem solving and debugging resources.

| File | Description | When to Use |
|------|-------------|-------------|
| [COMMON_ISSUES.md](troubleshooting/COMMON_ISSUES.md) | Frequent errors and solutions | When something breaks |

### **Schemas** (`schemas/`)

Database structures and data models.

| File | Description | When to Use |
|------|-------------|-------------|
| [DATABASE_SCHEMA.md](schemas/DATABASE_SCHEMA.md) | MongoDB and Qdrant schemas | Database operations, queries |

---

## 🎯 Use Case Navigation

### **"I'm starting fresh"**
1. [SYSTEM_OVERVIEW.md](architecture/SYSTEM_OVERVIEW.md)
2. [QUICK_START.md](guides/QUICK_START.md)
3. [AUTHENTICATION.md](guides/AUTHENTICATION.md)

### **"I need to understand the code"**
1. [AGENT_CATALOG.md](references/AGENT_CATALOG.md)
2. [CODE_PATTERNS.md](references/CODE_PATTERNS.md)
3. [DATABASE_SCHEMA.md](schemas/DATABASE_SCHEMA.md)

### **"Something is broken"**
1. [COMMON_ISSUES.md](troubleshooting/COMMON_ISSUES.md)
2. [CURRENT_STATE.md](architecture/CURRENT_STATE.md) - Known Issues section
3. Diagnostic scripts in project root

### **"I need context after a break"**
1. [CURRENT_STATE.md](architecture/CURRENT_STATE.md)
2. [FULL_HISTORY.md](architecture/FULL_HISTORY.md) - Recent changes
3. [AGENT_CATALOG.md](references/AGENT_CATALOG.md) - Refresh on agents

### **"I'm integrating the API"**
1. [API_REFERENCE.md](references/API_REFERENCE.md)
2. [AUTHENTICATION.md](guides/AUTHENTICATION.md)
3. [DATABASE_SCHEMA.md](schemas/DATABASE_SCHEMA.md)

### **"I'm adding a new feature"**
1. [CODE_PATTERNS.md](references/CODE_PATTERNS.md)
2. [AGENT_CATALOG.md](references/AGENT_CATALOG.md)
3. [CURRENT_STATE.md](architecture/CURRENT_STATE.md) - Check priorities

### **"I'm deploying to production"**
1. [ENVIRONMENT_SETUP.txt](guides/ENVIRONMENT_SETUP.txt)
2. [AUTHENTICATION.md](guides/AUTHENTICATION.md)
3. [SYSTEM_OVERVIEW.md](architecture/SYSTEM_OVERVIEW.md) - Scalability section

---

## 📊 Document Status

| File | Status | Last Updated | Completeness |
|------|--------|--------------|--------------|
| SYSTEM_OVERVIEW.md | ✅ Current | Oct 21, 2025 | 100% |
| CURRENT_STATE.md | ✅ Current | Oct 21, 2025 | 100% |
| FULL_HISTORY.md | ✅ Current | Oct 21, 2025 | 100% |
| QUICK_START.md | ✅ Current | Oct 21, 2025 | 100% |
| AUTHENTICATION.md | ✅ Current | Oct 21, 2025 | 100% |
| ENVIRONMENT_SETUP.txt | ✅ Current | Oct 21, 2025 | 100% |
| AGENT_CATALOG.md | ✅ Current | Oct 21, 2025 | 100% |
| API_REFERENCE.md | ✅ Current | Oct 21, 2025 | 100% |
| CODE_PATTERNS.md | ✅ Current | Oct 21, 2025 | 100% |
| COMMON_ISSUES.md | ✅ Current | Oct 21, 2025 | 100% |
| DATABASE_SCHEMA.md | ✅ Current | Oct 21, 2025 | 100% |

---

## 🔄 Maintenance Guidelines

### **When to Update**

**SYSTEM_OVERVIEW.md**:
- Major architecture changes
- New agent types
- Technology stack updates

**CURRENT_STATE.md**:
- After significant features
- New known issues
- Priority changes
- ⚠️ **UPDATE THIS MOST FREQUENTLY**

**FULL_HISTORY.md**:
- Major milestones
- Significant bugs fixed
- New features implemented

**AGENT_CATALOG.md**:
- New agents created
- Agent instructions changed
- New tools added

**API_REFERENCE.md**:
- New endpoints
- Changed request/response formats
- Authentication changes

**CODE_PATTERNS.md**:
- New coding conventions
- Best practice updates
- Common pattern discoveries

**COMMON_ISSUES.md**:
- New bugs discovered
- New solutions found
- Workarounds implemented

**DATABASE_SCHEMA.md**:
- Schema changes
- New collections/indexes
- Query pattern updates

### **Update Checklist**

After making significant changes:

- [ ] Update `CURRENT_STATE.md` (Required)
- [ ] Add to `FULL_HISTORY.md` if major feature
- [ ] Update relevant reference docs
- [ ] Update `INDEX.md` if new files added
- [ ] Update doc status table in `INDEX.md`
- [ ] Test all diagnostic scripts still work
- [ ] Update `QUICK_START.md` if setup changed

---

## 🎓 Learning Path

### **Beginner** (1-2 hours)
1. Read SYSTEM_OVERVIEW.md
2. Follow QUICK_START.md
3. Skim AGENT_CATALOG.md
4. Test with simple query

### **Intermediate** (3-5 hours)
1. Read CODE_PATTERNS.md
2. Study AGENT_CATALOG.md in detail
3. Read API_REFERENCE.md
4. Modify an agent's instructions
5. Add a simple tool

### **Advanced** (5-10 hours)
1. Read DATABASE_SCHEMA.md
2. Study FULL_HISTORY.md
3. Create a new agent
4. Implement custom pipeline
5. Optimize performance
6. Contribute to documentation

---

## 📚 External Resources

### **Google ADK**
- Docs: https://cloud.google.com/vertex-ai/docs
- Examples: https://github.com/google/agent-dev-kit

### **Qdrant**
- Docs: https://qdrant.tech/documentation/
- Python Client: https://github.com/qdrant/qdrant-client

### **FastAPI**
- Docs: https://fastapi.tiangolo.com/
- Tutorial: https://fastapi.tiangolo.com/tutorial/

### **Gemini**
- API Docs: https://ai.google.dev/docs
- Embedding Model: https://ai.google.dev/docs/embeddings

---

## 🆘 Quick Help

**Common Questions**:

| Question | Answer |
|----------|--------|
| How do I start the service? | See [QUICK_START.md](guides/QUICK_START.md) |
| How do I create an agent? | See [CODE_PATTERNS.md](references/CODE_PATTERNS.md) - Creating an Agent |
| Why is my JWT failing? | See [COMMON_ISSUES.md](troubleshooting/COMMON_ISSUES.md) - Authentication Issues |
| How do I query Qdrant? | See [DATABASE_SCHEMA.md](schemas/DATABASE_SCHEMA.md) - Query Patterns |
| What agents exist? | See [AGENT_CATALOG.md](references/AGENT_CATALOG.md) |
| What's the API? | See [API_REFERENCE.md](references/API_REFERENCE.md) |

---

## 📞 Support

1. **Check**: This memory bank
2. **Test**: Run diagnostic scripts
3. **Search**: Grep codebase for examples
4. **Debug**: Enable DEBUG logging

---

## ✨ Tips for AI Assistants

When resuming work on this project:

1. **Always read** `CURRENT_STATE.md` first
2. **Check** relevant reference docs for technical details
3. **Update** `CURRENT_STATE.md` after significant changes
4. **Refer to** `CODE_PATTERNS.md` when writing code
5. **Consult** `COMMON_ISSUES.md` before fixing bugs

---

**Memory Bank Version**: 1.0  
**Created**: October 21, 2025  
**Last Updated**: October 21, 2025  
**Maintainer**: Development Team

