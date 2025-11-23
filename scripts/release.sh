#!/bin/bash
# Helper script to create a new release
# Usage: ./scripts/release.sh 0.1.1 "Bug fixes and improvements"

set -e

VERSION=$1
NOTES=$2

if [ -z "$VERSION" ]; then
    echo "Error: Version number required"
    echo "Usage: ./scripts/release.sh <version> <optional-notes>"
    echo "Example: ./scripts/release.sh 0.1.1 'Bug fixes and improvements'"
    exit 1
fi

# Validate version format
if ! [[ $VERSION =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "Error: Version must be in format X.Y.Z (e.g., 0.1.1)"
    exit 1
fi

echo "🚀 Preparing release v$VERSION"
echo "================================"

# 1. Check git status
if [[ -n $(git status -s) ]]; then
    echo "⚠️  Warning: You have uncommitted changes"
    git status -s
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 2. Update version in pyproject.toml
echo "📝 Updating version in pyproject.toml..."
sed -i '' "s/^version = \".*\"/version = \"$VERSION\"/" pyproject.toml

# 3. Update version in __init__.py
echo "📝 Updating version in __init__.py..."
sed -i '' "s/__version__ = \".*\"/__version__ = \"$VERSION\"/" src/crossvector/__init__.py

# 4. Run tests
echo "🧪 Running tests..."
uv run pytest || {
    echo "❌ Tests failed! Please fix and try again."
    exit 1
}

# 5. Build package
echo "📦 Building package..."
rm -rf dist/ build/ *.egg-info
uv pip install build twine
python -m build

# 6. Check package
echo "🔍 Checking package with twine..."
twine check dist/*

# 7. Show changes
echo ""
echo "✅ Package built successfully!"
echo ""
echo "Package contents:"
ls -lh dist/
echo ""

# 8. Commit version bump
echo "💾 Committing version bump..."
git add pyproject.toml src/crossvector/__init__.py
git commit -m "Bump version to $VERSION"

# 9. Create tag
echo "🏷️  Creating git tag v$VERSION..."
if [ -z "$NOTES" ]; then
    git tag -a "v$VERSION" -m "Release version $VERSION"
else
    git tag -a "v$VERSION" -m "$NOTES"
fi

# 10. Show next steps
echo ""
echo "✅ Release preparation complete!"
echo ""
echo "Next steps:"
echo "1. Review changes: git show v$VERSION"
echo "2. Push changes: git push origin main --tags"
echo "3. Create GitHub release: gh release create v$VERSION --generate-notes"
echo ""
echo "Or run all at once:"
echo "  git push origin main --tags && gh release create v$VERSION --generate-notes"
echo ""
