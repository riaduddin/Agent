#!/bin/bash
# Script to create a new release version
# Supports both direct push and protected branch (PR) workflows

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}   Release Version Script${NC}"
echo -e "${BLUE}========================================${NC}"

# Check if version argument is provided
if [ -z "$1" ]; then
    echo -e "${YELLOW}Usage: ./release.sh <version>${NC}"
    echo -e "${YELLOW}Example: ./release.sh 1.1.0${NC}"
    exit 1
fi

NEW_VERSION=$1

# Validate version format (MAJOR.MINOR.PATCH)
if ! [[ $NEW_VERSION =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo -e "${RED}Error: Version must be in format MAJOR.MINOR.PATCH (e.g., 1.0.0)${NC}"
    exit 1
fi

# Get current branch
CURRENT_BRANCH=$(git branch --show-current)
echo -e "${BLUE}Current branch: ${CURRENT_BRANCH}${NC}"

# Check if on main branch
if [ "$CURRENT_BRANCH" = "main" ]; then
    echo -e "${YELLOW}⚠️  You are on the main branch${NC}"
    echo -e "${YELLOW}If main is protected, you should create a release branch first.${NC}"
    echo -e "${YELLOW}Create release branch? (y/n)${NC}"
    read -r CREATE_BRANCH
    
    if [ "$CREATE_BRANCH" = "y" ]; then
        RELEASE_BRANCH="release/${NEW_VERSION}"
        echo -e "${BLUE}Creating branch: ${RELEASE_BRANCH}${NC}"
        git checkout -b "$RELEASE_BRANCH"
        CURRENT_BRANCH="$RELEASE_BRANCH"
    fi
fi

echo -e "${GREEN}Creating release for version ${NEW_VERSION}${NC}"

# Update __version__.py
echo -e "${BLUE}Updating __version__.py...${NC}"
sed -i "s/__version__ = \".*\"/__version__ = \"${NEW_VERSION}\"/" __version__.py

# Update release date
CURRENT_DATE=$(date +%Y-%m-%d)
sed -i "s/RELEASE_DATE = \".*\"/RELEASE_DATE = \"${CURRENT_DATE}\"/" __version__.py

echo -e "${GREEN}✓ Updated __version__.py${NC}"

# Prompt for changelog update
echo -e "${YELLOW}Please update CHANGELOG.md with the changes for version ${NEW_VERSION}${NC}"
echo -e "${YELLOW}Press Enter when done...${NC}"
read

# Git operations
echo -e "${BLUE}Committing changes...${NC}"
git add __version__.py CHANGELOG.md
git commit -m "Bump version to ${NEW_VERSION}"

echo -e "${BLUE}Creating Git tag...${NC}"
git tag -a "v${NEW_VERSION}" -m "Release version ${NEW_VERSION}"

echo -e "${GREEN}✓ Created tag v${NEW_VERSION}${NC}"

# Determine push strategy
if [ "$CURRENT_BRANCH" = "main" ]; then
    # Direct push to main
    echo -e "${YELLOW}Push to main and tag? (y/n)${NC}"
    read -r PUSH_CONFIRM
    
    if [ "$PUSH_CONFIRM" = "y" ]; then
        echo -e "${BLUE}Pushing to remote...${NC}"
        git push origin main
        git push origin "v${NEW_VERSION}"
        echo -e "${GREEN}✓ Pushed to remote${NC}"
    else
        echo -e "${YELLOW}Skipped push. Remember to push manually:${NC}"
        echo -e "  git push origin main"
        echo -e "  git push origin v${NEW_VERSION}"
    fi
else
    # Push release branch for PR workflow
    echo -e "${YELLOW}Push release branch for Pull Request? (y/n)${NC}"
    read -r PUSH_CONFIRM
    
    if [ "$PUSH_CONFIRM" = "y" ]; then
        echo -e "${BLUE}Pushing release branch to remote...${NC}"
        git push origin "$CURRENT_BRANCH"
        echo -e "${GREEN}✓ Pushed branch ${CURRENT_BRANCH}${NC}"
        
        echo -e ""
        echo -e "${BLUE}========================================${NC}"
        echo -e "${BLUE}   Next Steps (PR Workflow)${NC}"
        echo -e "${BLUE}========================================${NC}"
        echo -e "${YELLOW}1. Create Pull Request:${NC}"
        echo -e "   ${CURRENT_BRANCH} → main"
        echo -e ""
        echo -e "${YELLOW}2. After PR is merged, push the tag:${NC}"
        echo -e "   ${GREEN}git checkout main${NC}"
        echo -e "   ${GREEN}git pull origin main${NC}"
        echo -e "   ${GREEN}git push origin v${NEW_VERSION}${NC}"
        echo -e ""
        echo -e "${YELLOW}3. Optionally delete release branch:${NC}"
        echo -e "   ${GREEN}git branch -d ${CURRENT_BRANCH}${NC}"
        echo -e "   ${GREEN}git push origin --delete ${CURRENT_BRANCH}${NC}"
        echo -e "${BLUE}========================================${NC}"
    else
        echo -e "${YELLOW}Skipped push. Remember to:${NC}"
        echo -e "  1. Push branch: git push origin ${CURRENT_BRANCH}"
        echo -e "  2. Create PR: ${CURRENT_BRANCH} → main"
        echo -e "  3. After merge, push tag: git push origin v${NEW_VERSION}"
    fi
fi

echo -e ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}   Release ${NEW_VERSION} Prepared!${NC}"
echo -e "${GREEN}========================================${NC}"
