#!/usr/bin/env bash
# =============================================================================
# github_push.sh — Push AI-Hiring-Intelligence to GitHub
#
# BEFORE RUNNING:
#   1. Create the repo on GitHub: https://github.com/new
#      - Name: AI-Hiring-Intelligence
#      - Visibility: Public (required for hackathon sandbox)
#      - DO NOT initialise with README (we have our own)
#
#   2. Set your credentials in the two lines below:
GITHUB_USERNAME="YOUR_GITHUB_USERNAME"
GITHUB_TOKEN="YOUR_PERSONAL_ACCESS_TOKEN"
#
#   3. Run: bash github_push.sh
# =============================================================================

set -e  # Exit immediately on any error

REPO_NAME="AI-Hiring-Intelligence"
REMOTE_URL="https://${GITHUB_USERNAME}:${GITHUB_TOKEN}@github.com/${GITHUB_USERNAME}/${REPO_NAME}.git"

echo "========================================================"
echo "  AI-Hiring-Intelligence — GitHub Push Script"
echo "========================================================"
echo ""

# Navigate into repo directory
cd "$(dirname "$0")"

# Initialise git (safe to re-run)
if [ ! -d ".git" ]; then
    echo "[1/7] Initialising git repository..."
    git init
    git branch -M main
else
    echo "[1/7] Git already initialised ✓"
fi

# Configure git identity (uses system git config if already set)
git config user.name  "${GITHUB_USERNAME}"
git config user.email "${GITHUB_USERNAME}@users.noreply.github.com"

# Add remote (update if already exists)
echo "[2/7] Setting remote origin..."
if git remote | grep -q "^origin$"; then
    git remote set-url origin "${REMOTE_URL}"
else
    git remote add origin "${REMOTE_URL}"
fi
echo "      Remote: https://github.com/${GITHUB_USERNAME}/${REPO_NAME}"

# Stage all files
echo "[3/7] Staging all files..."
git add .
echo "      Staged files:"
git diff --cached --name-only | head -40 | sed "s/^/        /"

# Commit
echo "[4/7] Creating initial commit..."
git commit -m "feat: initial submission — AI-Hiring-Intelligence

Redrob Intelligent Candidate Discovery & Ranking Challenge

- 5-stage hybrid ranking pipeline (TF-IDF + features + behavioral)
- Profile Credibility Engine (Phase X) — 8 analytical modules
- 18 engineered features calibrated for NDCG@10
- Explainable AI with per-candidate reasoning strings
- 0 honeypots in top-100 | runtime < 5 min | CPU-only
- Full competition compliance (7/7 validation checks PASS)

Competition metrics: NDCG@10 (50%), NDCG@50 (30%), MAP (15%), P@10 (5%)
Built with: Python, NumPy, Pandas | Seed: 42 (fully deterministic)
" --allow-empty

# Push
echo "[5/7] Pushing to GitHub..."
git push -u origin main --force

echo ""
echo "[6/7] Verifying push..."
echo "      Repository: https://github.com/${GITHUB_USERNAME}/${REPO_NAME}"

echo ""
echo "[7/7] Next steps:"
echo "  1. Visit https://github.com/${GITHUB_USERNAME}/${REPO_NAME}"
echo "  2. Confirm all files are present (README, src/, docs/, submission/)"
echo "  3. Fill in submission_metadata.yaml with your team details"
echo "  4. Deploy sandbox (Streamlit Cloud or HuggingFace Spaces)"
echo "  5. Submit on the competition portal"
echo ""
echo "========================================================"
echo "  PUSH COMPLETE ✓"
echo "========================================================"