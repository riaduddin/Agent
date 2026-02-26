# Application Versioning Strategy

## Overview
This document outlines the versioning strategy for the Presentation Generation Service.

## Semantic Versioning (SemVer)
We follow [Semantic Versioning 2.0.0](https://semver.org/):

**Format**: `MAJOR.MINOR.PATCH`

- **MAJOR**: Breaking changes (incompatible API changes)
- **MINOR**: New features (backwards-compatible)
- **PATCH**: Bug fixes (backwards-compatible)

### Examples
- `1.0.0` → `1.0.1`: Bug fix
- `1.0.1` → `1.1.0`: New feature added
- `1.1.0` → `2.0.0`: Breaking change

## Version Locations

### 1. Code Version (`__version__.py`)
The canonical version is stored in `__version__.py`:
```python
__version__ = "1.0.0"
```

### 2. Git Tags
Each release is tagged in Git:
```bash
git tag -a v1.0.0 -m "Release version 1.0.0"
git push origin v1.0.0
```

### 3. API Response
The version is exposed via the `/health` endpoint for monitoring.

## Release Workflow

### Creating a New Release

1. **Update Version Number**
   ```bash
   # Edit __version__.py
   __version__ = "1.1.0"
   ```

2. **Update CHANGELOG.md**
   Document all changes in the changelog.

3. **Commit Changes**
   ```bash
   git add __version__.py CHANGELOG.md
   git commit -m "Bump version to 1.1.0"
   ```

4. **Create Git Tag**
   ```bash
   git tag -a v1.1.0 -m "Release v1.1.0: Add Socket.IO logging and refine user log filtering"
   ```

5. **Push to Remote**
   ```bash
   git push origin main
   git push origin v1.1.0
   ```

## Version Checking

### In Code
```python
from __version__ import __version__
print(f"Running version {__version__}")
```

### Via API
```bash
curl http://localhost:8060/health
```

Response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2026-01-11T11:35:00Z"
}
```

### Via Git
```bash
git describe --tags
```

## Current Version
**v1.0.0** - Initial production release

## Next Steps
- Set up automated version bumping with CI/CD
- Consider using `bump2version` or `semantic-release` for automation
- Add version to Docker image tags
