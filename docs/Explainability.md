# Explainability Report

## Approach
Every top-100 candidate receives a transparent explanation:

1. Feature Contribution Decomposition — per-feature weighted contribution (waterfall-style)
2. Strength Extraction — top-3 positive signals from the candidate profile
3. Risk Identification — up to 2 data concerns (never factual accusations)
4. Factual Reasoning String — 2-sentence specific explanation for submission CSV
5. Top-3 Ranking Drivers — features that contributed most to their final score

## Example Output (Rank #1)

```
Candidate:  CAND_0039754
Score:      1.0000
Credibility: 99.0/100  [High]

Reasoning:
  Senior Applied Scientist with 8.0y experience, specialising in
  fine-tuning, LLMs, Qdrant, reinforcement learning. Demonstrates
  vector DB expertise and active GitHub contributions, closely
  matching the Senior AI Engineer JD requirements.

Strengths:
  1. 8.0y of AI/ML experience as Senior Applied Scientist
  2. Hands-on vector DB experience (Pinecone/Qdrant/FAISS)
  3. Active open-source contributor (GitHub score 0.92)

Top Drivers:
  Technical Depth Score:  0.1440
  JD Semantic Similarity: 0.1320
  Active & Available:     0.0850
```

## Design Principles
- No black-box outputs: every score is fully decomposable
- No factual accusations: risks framed as items to verify
- Factual over generic: reasoning uses actual candidate data
- Consistent: same methodology applied to all 100 candidates