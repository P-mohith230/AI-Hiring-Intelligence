# GitHub Repository Setup Guide

Complete step-by-step instructions to push AI-Hiring-Intelligence to GitHub.

---

## Prerequisites

- A GitHub account (free at https://github.com)
- Git installed on your machine (`git --version` to check)
- Python 3.9+ installed

---

## Step 1 — Create the GitHub Repository

1. Go to https://github.com/new
2. Fill in:
   - **Repository name:** `AI-Hiring-Intelligence`
   - **Description:** `Enterprise-grade AI candidate ranking system — Redrob Challenge`
   - **Visibility:** ✅ Public (required for hackathon sandbox)
   - ⛔ **DO NOT** tick "Add a README file" (we have our own)
   - ⛔ **DO NOT** add .gitignore or license (we have our own)
3. Click **Create repository**
4. Copy the repository URL shown (e.g. `https://github.com/YOUR_USERNAME/AI-Hiring-Intelligence.git`)

---

## Step 2 — Generate a Personal Access Token (PAT)

1. Go to https://github.com/settings/tokens
2. Click **Generate new token (classic)**
3. Note: `AI-Hiring-Intelligence push`
4. Expiration: 30 days (or longer)
5. Scopes: ✅ `repo` (full control of private repositories)
6. Click **Generate token**
7. **Copy the token immediately** — it will not be shown again

---

## Step 3 — Download the Repository Files

1. Download `AI-Hiring-Intelligence.zip` from your Zerve workspace
2. Unzip it to a folder on your computer
3. Open a terminal in that folder

---

## Step 4 — Push to GitHub

### Option A: Using the Push Script (Recommended)

```bash
# Edit github_push.sh and fill in your credentials:
#   GITHUB_USERNAME="your-username"
#   GITHUB_TOKEN="ghp_xxxxxxxxxxxxx"

# Make executable and run:
chmod +x github_push.sh
bash github_push.sh
```

### Option B: Manual Git Commands

```bash
cd AI-Hiring-Intelligence

# Initialise git
git init
git branch -M main

# Add all files
git add .

# Create initial commit
git commit -m "feat: initial submission — AI-Hiring-Intelligence

Redrob Intelligent Candidate Discovery & Ranking Challenge
5-stage hybrid ranking pipeline | Profile Credibility Engine | XAI
0 honeypots | runtime < 5 min | CPU-only | Seed 42"

# Add remote (replace YOUR_USERNAME and YOUR_TOKEN)
git remote add origin https://YOUR_USERNAME:YOUR_TOKEN@github.com/YOUR_USERNAME/AI-Hiring-Intelligence.git

# Push
git push -u origin main
```

---

## Step 5 — Verify on GitHub

Visit: `https://github.com/YOUR_USERNAME/AI-Hiking-Intelligence`

Confirm these files and folders are present:
- ✅ README.md (should render with badges and tables)
- ✅ src/ (6 Python modules)
- ✅ submission/run_submission.py
- ✅ submission/submission.csv
- ✅ docs/ (4 markdown files)
- ✅ assets/architecture.png
- ✅ requirements.txt, LICENSE, .gitignore

---

## Step 6 — Fill In Submission Metadata

Edit `submission_metadata.yaml` on GitHub:
1. Click the file in the GitHub UI
2. Click the ✏️ pencil icon to edit
3. Fill in:
   - `team_name`: your team name
   - `name`: your full name
   - `email`: your email address
   - `github_repository`: the URL from Step 1
   - `sandbox_url`: your deployed app URL (Step 7)
4. Click **Commit changes**

---

## Step 7 — Deploy a Sandbox (Required for Submission)

The competition requires a live demo. Quickest option: **Streamlit Community Cloud**

1. Go to https://share.streamlit.io
2. Sign in with your GitHub account
3. Click **New app**
4. Select: `AI-Hiring-Intelligence` repo
5. Branch: `main`
6. Main file path: `submission/run_submission.py`

> **Note:** The full pipeline needs `candidates.jsonl` which is 487 MB.
> For the sandbox, you can use `sample_candidates.json` (50 records, 300 KB)
> to demonstrate the scoring and credibility engine without the full dataset.

---

## Step 8 — Final Submission Checklist

Before submitting on the competition portal:

- [ ] `submission.csv` has exactly 100 rows
- [ ] Columns: `candidate_id, rank, score, reasoning`
- [ ] Ranks 1–100, no gaps
- [ ] Scores in [0.0, 1.0], monotonically non-increasing
- [ ] All reasoning strings present (non-empty)
- [ ] `submission_metadata.yaml` fully filled in
- [ ] GitHub repository URL is correct and public
- [ ] Sandbox URL is live and accessible
- [ ] `python validate_submission.py submission.csv` passes

---

## Troubleshooting

**`remote: Support for password authentication was removed`**
→ Use a Personal Access Token (PAT) in the URL, not your GitHub password.

**`error: failed to push some refs`**
→ Use `git push --force` if you need to overwrite.

**`Permission denied (publickey)`**
→ Use HTTPS URL with PAT, not SSH.

---

## Quick Reference — Key Files

| File | Purpose |
|------|---------|
| `submission/run_submission.py` | Single command to reproduce submission |
| `submission/submission.csv` | Final competition submission file |
| `submission_metadata.yaml` | Team info for portal upload |
| `github_push.sh` | Automated push script |
| `requirements.txt` | Python dependencies |
| `README.md` | Project overview for judges |