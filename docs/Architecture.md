# System Architecture

## Overview
The AI-Hiring-Intelligence system uses a 5-stage hybrid pipeline that processes
100,000 candidate profiles and returns the top 100 ranked against a Senior AI
Engineer job description, with full per-candidate explainability.

## Pipeline Stages

### Stage 1: Hard Filters
Removes definitively ineligible candidates:
- Honeypot profiles (logically impossible data)
- Silent candidates (inactive >270 days AND response rate <5%)
- Non-India location (not willing to relocate)
- YOE outside [2, 18] years

### Stage 2: Structured JD-Fit Scoring
Applies 15-feature weighted scoring to all eligible candidates.
Weights calibrated for NDCG@10 optimisation.

### Stage 3: NLP Re-scoring (Top 5K)
Boosts for vector DB (+3%), GitHub (+2%), cosine similarity (+2%).
Penalises low availability (-4%).

### Stage 4: Behavioral Re-ranking (Top 1K)
5% behavioral trust bonus: recruiter interest + interview completion + availability.

### Stage 5: Final Top-100
Selects, normalises to [0.5, 1.0], enforces monotonicity, audits honeypots.

## Phase X: Profile Credibility Engine
Independent 8-module system estimating profile internal consistency.
Runs on all 100K candidates as a decision-support layer.

## Data Flow
```
candidates.jsonl
    → preprocessing (flatten + honeypot detection)
    → NLP analysis (TF-IDF + tech entities)
    → feature engineering (18 f_* features)
    → ranking pipeline (5 stages)
    → explainability (reasoning + waterfall)
    → submission.csv

credibility engine (independent Phase X) → recruiter dashboard
```