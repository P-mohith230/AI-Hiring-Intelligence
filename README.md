<div align="center">

# 🏆 AI-Hiring-Intelligence

### Enterprise-Grade AI Candidate Ranking System
### Redrob Intelligent Candidate Discovery & Ranking Challenge

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Competition](https://img.shields.io/badge/Competition-Redrob%20AI-orange)](https://redrob.ai)
[![NDCG@10](https://img.shields.io/badge/Primary%20Metric-NDCG%40%2050%25-red)]()
[![CPU Only](https://img.shields.io/badge/Compute-CPU%20Only-lightgrey)]()
[![Runtime](https://img.shields.io/badge/Runtime-%3C%205%20min-brightgreen)]()

> **Ranking 100,000 candidates for a Senior AI Engineer role using a 5-stage hybrid AI pipeline,
> Profile Credibility Engine, and per-candidate Explainable AI — all in under 5 minutes on CPU.**

</div>

---

## 📋 Project Overview

This repository contains the complete solution for the **Redrob Intelligent Candidate Discovery & Ranking Challenge** — a national AI hackathon requiring participants to intelligently rank 100,000 candidate profiles against a Senior AI Engineer job description.

The system goes far beyond simple keyword matching. It combines **semantic NLP**, **18 engineered features**, **behavioral platform signals**, and a novel **Profile Credibility Engine** to produce a ranked list of the top 100 candidates — each with a transparent, factual AI-generated explanation.

---

## 🎯 Competition Objective

| Item | Detail |
|------|--------|
| **Task** | Rank top 100 of 100,000 candidates for a Senior AI Engineer role |
| **Primary Metric** | NDCG@10 (50% of total score) |
| **Secondary Metrics** | NDCG@50 (30%), MAP (15%), P@10 (5%) |
| **Constraints** | CPU only · ≤5 min · ≤16 GB RAM · No internet during ranking |
| **Output** | `submission.csv` with columns: `candidate_id, rank, score, reasoning` |
| **Dataset** | 100,000 candidate JSONL profiles with 8 nested sections each |

---

## 🚀 Key Innovation

### 1. Profile Credibility Engine (Phase X)
A completely independent 8-module system that answers:
> *"How much confidence should a recruiter have in this candidate profile?"*

Unlike simple scoring, the Credibility Engine detects **internal consistency** across career timelines, skill evidence, technical depth, leadership language, evidence quality, and behavioral signals — producing a 0–100 Credibility Score with transparent per-candidate explanations.

### 2. 5-Stage Hybrid Ranking Architecture
```
Stage 1: Hard Filters      → Remove honeypots, silent candidates, location mismatches
Stage 2: Structured Score  → 15-feature weighted JD-fit scoring (all eligible)
Stage 3: NLP Re-score      → TF-IDF cosine + tech entity boost (top 5K)
Stage 4: Behavioral Re-rank→ Platform engagement signals (top 1K)
Stage 5: Final Selection   → Top-100 with monotonic normalisation + audit
```

### 3. Honeypot Detection
~80 engineered trap profiles in the dataset. >10% honeypots in top-100 triggers automatic disqualification. Our multi-rule detection achieves **0 honeypots** in the final submission.

### 4. Explainable AI
Every candidate in the top-100 receives:
- Factual 2-sentence reasoning string (in submission CSV)
- Feature contribution breakdown (waterfall)
- Top-3 ranking drivers
- Credibility strengths and concerns

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   AI-HIRING-INTELLIGENCE                        │
│                   Competition Pipeline                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  candidates.jsonl (100K records, 487 MB)                        │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────┐    Stream-parse + flatten nested JSON          │
│  │ Preprocessing│   Detect honeypots + silent candidates        │
│  └──────┬──────┘   Flag India-reachable candidates             │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────┐    TF-IDF cosine similarity (JD vs corpus)     │
│  │ NLP Analysis │   Technical entity extraction                 │
│  └──────┬──────┘   Seniority + vector DB detection             │
│         │                                                       │
│         ▼                                                       │
│  ┌──────────────────┐   18 engineered features (f_*)            │
│  │Feature Engineering│  YOE fit, tech depth, availability       │
│  └──────┬───────────┘  GitHub, recruiter signals, startup       │
│         │                                                       │
│         ▼                                                       │
│  ┌──────────────────┐   5-stage hybrid ranking                  │
│  │  Ranking Engine  │   Hard filters → score → boost → re-rank  │
│  └──────┬───────────┘   Final top-100 with audit                │
│         │                                                       │
│         ├──────────────────────────┐                            │
│         ▼                          ▼                            │
│  ┌─────────────┐     ┌─────────────────────────┐               │
│  │Explainable  │     │  Profile Credibility     │               │
│  │    AI       │     │  Engine (Phase X)         │               │
│  │ Reasoning   │     │  8 analytical modules     │               │
│  └──────┬──────┘     └──────────┬──────────────┘               │
│         │                        │                              │
│         └──────────┬─────────────┘                              │
│                    ▼                                            │
│            submission.csv                                       │
│       (candidate_id, rank, score, reasoning)                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 Major Features

### 18 Engineered Features

| Feature | Description | Weight |
|---------|-------------|--------|
| `f_tech_depth` | Technical terminology depth in career descriptions | 18% |
| `f_yoe_fit` | Experience band match (peak 5–9y) | 15% |
| `f_availability` | Recency + response rate + open-to-work | 10% |
| `f_nlp_cosine` | TF-IDF cosine similarity to JD | 12% |
| `f_vector_db` | Vector DB expertise flag (Pinecone/Qdrant/FAISS) | 8% |
| `f_github` | GitHub activity score | 7% |
| `f_title_seniority` | Job title seniority level | 6% |
| `f_recruiter_interest` | Saves + profile views (market demand) | 5% |
| `f_startup_exposure` | Early-stage company experience | 5% |
| ... | 9 more features | ... |

### Profile Credibility Engine (8 Modules)

| Module | Measures | Weight |
|--------|----------|--------|
| M3 Technical Depth | AI/ML terminology density | 18% |
| M6 Behavioral Confidence | Platform engagement signals | 16% |
| M2 Skill Evidence | Skills corroborated by career text | 14% |
| M5 Evidence Strength | Quantified achievements | 14% |
| M4 Leadership & Ownership | Strong vs weak action verbs | 12% |
| M1 Career Consistency | Timeline stability, role progression | 10% |
| M7 Temporal Consistency | Career timeline logic | 10% |
| M8 Narrative Intelligence | Mindset orientation analysis | 6% |

---

## 🧠 Explainable AI

Every top-100 candidate receives:

```
Rank #1  CAND_0039754  Score: 1.0000  Credibility: 99.0/100  [High]
Reasoning: Senior Applied Scientist with 8.0y experience, specialising in
           fine-tuning, LLMs, Qdrant, reinforcement learning. Demonstrates
           vector DB expertise and active GitHub contributions, closely
           matching the Senior AI Engineer JD requirements.
Strengths: 8.0y AI/ML experience | Qdrant/FAISS expertise | Senior title
Top Driver: Technical Depth Score (0.1440)
```

---

## 🛠️ Technology Stack

| Category | Technologies |
|----------|-------------|
| Language | Python 3.9+ |
| Data | NumPy, Pandas |
| NLP | TF-IDF (pure Python + NumPy), regex |
| Ranking | Custom hybrid pipeline |
| Explainability | SHAP-style contribution decomposition |
| Visualisation | Matplotlib |
| Pipeline | Argparse, logging, pathlib |
| No external ML | ✅ sklearn-free, no GPU, no internet |

---

## 📁 Project Structure

```
AI-Hiring-Intelligence/
├── README.md                    ← This file
├── LICENSE                      ← MIT License
├── .gitignore                   ← Python + data gitignore
├── requirements.txt             ← Pinned dependencies
├── environment.yml              ← Conda environment
├── pyproject.toml               ← Package metadata
├── submission_metadata.yaml     ← Competition submission metadata
│
├── src/                         ← Modular source code
│   ├── utils.py                 ← Shared utilities (logging, NLP, normalisation)
│   ├── preprocessing.py         ← Data loading, flattening, honeypot detection
│   ├── feature_engineering.py   ← 18 engineered features
│   ├── ranking_engine.py        ← 5-stage hybrid ranking pipeline
│   ├── credibility_engine.py    ← Profile Credibility Engine (Phase X)
│   └── explainability.py        ← Per-candidate XAI reasoning
│
├── submission/
│   ├── run_submission.py        ← Single reproducible entrypoint
│   ├── submission.csv           ← Final competition submission
│   └── sample_submission.csv    ← Competition-provided sample
│
├── docs/
│   ├── Architecture.md          ← System architecture deep-dive
│   ├── Methodology.md           ← Full methodology documentation
│   ├── FeatureEngineering.md    ← Feature design rationale
│   └── Explainability.md        ← XAI approach and top-10 breakdown
│
└── assets/
    └── architecture.png         ← Pipeline architecture diagram
```

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/AI-Hiring-Intelligence.git
cd AI-Hiring-Intelligence

# Install dependencies
pip install -r requirements.txt

# OR using conda
conda env create -f environment.yml
conda activate ai-hiring
```

---

## 🚀 Running the Project

### Reproduce the Submission (Single Command)

```bash
python submission/run_submission.py --data path/to/candidates.jsonl
```

This produces `submission.csv` in the current directory.

### With Custom Output Path

```bash
python submission/run_submission.py \
    --data path/to/candidates.jsonl \
    --output my_submission.csv
```

### Validate Submission

```bash
python competition_data/.../validate_submission.py submission.csv
```

---

## ⏱️ Runtime & Resource Requirements

| Constraint | Requirement | Our System |
|------------|-------------|------------|
| Maximum runtime | 5 minutes | ~3–4 minutes |
| Maximum memory | 16 GB RAM | ~1–2 GB |
| Compute | CPU only | ✅ CPU-only |
| Network | No internet | ✅ Fully offline |
| Hosted LLM APIs | Not allowed | ✅ None used |
| Determinism | Required | ✅ Seed 42 |

---

## ✅ Competition Compliance

| Requirement | Status |
|-------------|--------|
| Exactly 100 candidates | ✅ |
| Ranks 1–100, no gaps | ✅ |
| Scores in [0.0, 1.0] | ✅ |
| Monotonically non-increasing scores | ✅ |
| Reasoning string for all 100 | ✅ |
| CAND_XXXXXXX ID format | ✅ |
| 0 honeypots in top-100 | ✅ |
| Runtime < 5 min | ✅ |
| Memory < 16 GB | ✅ |
| CPU-only | ✅ |
| No external APIs | ✅ |

---

## 🔭 Future Improvements

1. **Sentence Transformer embeddings** (MiniLM-L6) for deeper semantic JD matching
2. **LambdaMART / LightGBM Ranking** with pseudo-labelled training data
3. **Graph-based network analysis** on candidate–company co-occurrence
4. **Multi-lingual support** for non-English profiles
5. **Online learning** pipeline for real-time recruiter feedback integration
6. **A/B testing framework** for ranking weight optimisation

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- **Redrob AI** for the competition dataset and challenge design
- **Zerve.ai** for the AI-native data workspace platform used to build this solution
- The open-source Python ecosystem: NumPy, Pandas, Matplotlib

---

<div align="center">

**Built with ❤️ for the Redrob Intelligent Candidate Discovery & Ranking Challenge**

*Demonstrating that world-class AI recruiting intelligence can be built with transparency, efficiency, and explainability.*

</div>