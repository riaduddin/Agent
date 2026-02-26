# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-01-11

### Added
- Initial production release
- Dual logging system (User logs: `agent_outputs_2`, Developer logs: `agent_logs`)
- Comprehensive user log filtering to exclude technical events
- Socket.IO message logging utility (disabled by default, controlled via `ENABLE_SOCKETIO_LOGGING`)
- Session service timeout improvements (increased to 30s)
- Modular slide creation agent architecture
- Redis caching for browser research results
- Multi-worker Socket.IO support with Redis pub/sub

### Fixed
- SyntaxError in `lightweight_slide_pipeline.py` (async generator return issue)
- TimeoutError warnings in session service operations
- Log duplication in `agent_outputs_2` collection

### Changed
- Refactored `slide_creation_agent.py` into modular components
- Excluded internal agent events from user-facing logs:
  - `enhanced_slide_generator` events
  - `template_selector` events
  - Internal tool calls (`keyword_research_agent`, `search_query_agent`, etc.)
  - Enhanced slide generator tool calls (`retrieve_research_context`, `search_images`)

### Technical Details
- Database: MongoDB with optimized connection pooling
- Session Management: Google ADK with centralized session service
- Real-time Communication: Socket.IO with Redis for multi-worker support
- Agent Framework: Google ADK with custom logging wrappers

## [Unreleased]

### Planned
- Automated version bumping with CI/CD
- Docker image versioning
- API versioning strategy

---

## Version Format
- **MAJOR.MINOR.PATCH** (Semantic Versioning)
- **MAJOR**: Breaking changes
- **MINOR**: New features (backwards-compatible)
- **PATCH**: Bug fixes (backwards-compatible)
