# Memory Bank - Project Context Repository

This folder contains all critical information needed to understand, maintain, and extend the Presentation Generation Service.

## 📁 Folder Structure

```
memory_bank/
├── architecture/          # System design, agent flows, data models
├── guides/               # Setup, configuration, deployment guides
├── troubleshooting/      # Common issues, fixes, debugging
├── references/           # API docs, agent configs, code patterns
└── schemas/              # Database schemas, data structures
```

## 🎯 Purpose

This memory bank serves as:
- **Onboarding Guide** for new developers
- **Context Reference** for AI assistants across sessions
- **Documentation Hub** for system architecture
- **Troubleshooting Resource** for common issues
- **Change Log** for major updates

**📖 New to memory bank?** Read [MEMORY_BANK_GUIDE.md](MEMORY_BANK_GUIDE.md) for a complete overview!

## 📖 How to Use

### For New Developers
1. Start with `architecture/SYSTEM_OVERVIEW.md`
2. Read `guides/QUICK_START.md` for setup
3. Review `references/AGENT_CATALOG.md` to understand agents
4. Check `troubleshooting/COMMON_ISSUES.md` if problems arise

### For AI Assistants / Context Windows
When resuming work on this project:
1. Read `architecture/CURRENT_STATE.md` for latest status
2. Check `architecture/RECENT_CHANGES.md` for recent updates
3. Review `references/CODE_PATTERNS.md` for conventions
4. Consult `troubleshooting/KNOWN_ISSUES.md` for active problems

### For Code Changes
Before making changes:
1. Review relevant architecture docs
2. Check `references/AGENT_CATALOG.md` for agent details
3. Update `architecture/RECENT_CHANGES.md` after significant changes
4. Add new issues to `troubleshooting/KNOWN_ISSUES.md`

## 🔄 Maintenance

**Update these files regularly:**
- `architecture/CURRENT_STATE.md` - After major features
- `architecture/RECENT_CHANGES.md` - After each significant change
- `troubleshooting/KNOWN_ISSUES.md` - When bugs are found/fixed
- `schemas/` - When database structure changes

## 📋 Quick Reference

| Need | File |
|------|------|
| System overview | `architecture/SYSTEM_OVERVIEW.md` |
| Setup instructions | `guides/QUICK_START.md` |
| Environment config | `guides/ENVIRONMENT_SETUP.md` |
| Agent details | `references/AGENT_CATALOG.md` |
| API endpoints | `references/API_REFERENCE.md` |
| Common errors | `troubleshooting/COMMON_ISSUES.md` |
| Database schema | `schemas/DATABASE_SCHEMA.md` |
| Qdrant setup | `guides/QDRANT_SETUP.md` |
| Authentication | `guides/AUTHENTICATION.md` |

## 🚀 Getting Started Checklist

- [ ] Read `architecture/SYSTEM_OVERVIEW.md`
- [ ] Follow `guides/QUICK_START.md`
- [ ] Configure `.env` using `guides/ENVIRONMENT_SETUP.md`
- [ ] Start Qdrant using `guides/QDRANT_SETUP.md`
- [ ] Test authentication using `guides/AUTHENTICATION.md`
- [ ] Review `references/AGENT_CATALOG.md`
- [ ] Run test scripts in `troubleshooting/TESTING_GUIDE.md`

---

**Last Updated**: Context organization  
**Version**: 1.0  
**Maintained By**: Development Team

