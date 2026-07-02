"""
explainability.py — Per-candidate reasoning and XAI explanations.

For every candidate in the top-100:
    1. Computes per-feature weighted contribution (waterfall decomposition)
    2. Extracts top-3 strengths and up to 2 risk factors
    3. Generates a factual, specific 1–2 sentence reasoning string
    4. Identifies the top 3 ranking drivers

All explanations are interpretable, factual, and free from black-box reasoning.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

logger = logging.getLogger("ai_hiring.explainability")

# Feature metadata: key → (human label, weight)
FEATURE_META: Dict[str, Tuple[str, float]] = {
    "f_yoe_fit":              ("Years of Experience Fit",        0.12),
    "f_tech_depth":           ("Technical Depth Score",          0.18),
    "f_nlp_cosine":           ("JD Semantic Similarity",         0.15),
    "f_availability":         ("Active & Available",             0.10),
    "f_github":               ("GitHub Activity",                0.08),
    "f_recruiter_interest":   ("Recruiter Interest",             0.06),
    "f_interview_completion": ("Interview Completion Rate",      0.07),
    "f_title_seniority":      ("Title Seniority Level",          0.08),
    "f_vector_db":            ("Vector DB Experience",           0.06),
    "f_career_stability":     ("Career Stability",               0.04),
    "f_startup_exposure":     ("Startup Exposure",               0.02),
    "f_education_tier":       ("Education Tier",                 0.02),
    "f_profile_completeness": ("Profile Completeness",           0.01),
    "f_india_readiness":      ("India Hybrid Readiness",         0.01),
}

FEATURE_COLS  = list(FEATURE_META.keys())
FEATURE_NAMES = [FEATURE_META[k][0] for k in FEATURE_COLS]
WEIGHTS       = np.array([FEATURE_META[k][1] for k in FEATURE_COLS])


def compute_contributions(df_top100: pd.DataFrame) -> np.ndarray:
    """Compute per-feature weighted contributions for all top-100 candidates.

    Args:
        df_top100: Top-100 DataFrame with all f_* feature columns.

    Returns:
        (100, 14) numpy array of feature contributions.
    """
    feat_matrix = df_top100[FEATURE_COLS].fillna(0).values
    return feat_matrix * WEIGHTS


def get_top_drivers(contributions: np.ndarray, row_idx: int,
                    n: int = 3) -> List[Tuple[str, float]]:
    """Return the top-N ranking drivers for a candidate by contribution.

    Args:
        contributions: (100, 14) contribution matrix.
        row_idx:       Row index in the matrix.
        n:             Number of top drivers to return.

    Returns:
        List of (feature_name, contribution_value) tuples, descending.
    """
    c    = contributions[row_idx]
    top  = np.argsort(c)[::-1][:n]
    return [(FEATURE_NAMES[i], float(c[i])) for i in top]


def make_reasoning(row: pd.Series) -> Dict:
    """Generate a transparent reasoning string and XAI breakdown for one candidate.

    Args:
        row: Candidate row from df_top100 with all feature and profile columns.

    Returns:
        Dict with reasoning (str), strengths (list), risks (list).
    """
    yoe        = round(float(row.get("years_of_experience", 0)), 1)
    title      = str(row.get("current_title", "Unknown"))
    skills_raw = str(row.get("skills_text", ""))
    skills_list = [s.strip() for s in skills_raw.split(",") if len(s.strip()) > 2][:4]
    skills_str  = ", ".join(skills_list) if skills_list else "ML/AI skills"

    github      = float(row.get("github_activity_score", 0) or 0)
    vec_db      = bool(row.get("vector_db_flag", False))
    tech_score  = float(row.get("f_tech_depth", 0))
    avail_days  = int(row.get("last_active_days_ago", 999))
    interview   = float(row.get("f_interview_completion", 0))
    seniority   = float(row.get("f_title_seniority", 0))
    edu_tier    = str(row.get("edu_tier", ""))
    honeypot    = float(row.get("f_honeypot_risk", 0))

    # Build strengths list
    strengths = []
    if yoe >= 5:
        strengths.append(f"{yoe}y of AI/ML experience as {title}")
    if vec_db:
        strengths.append("hands-on vector DB experience (Pinecone/Qdrant/FAISS)")
    if tech_score >= 0.6:
        strengths.append(f"strong technical depth in {skills_str}")
    if github >= 70:
        strengths.append(f"active open-source contributor (GitHub score {github:.0f}/100)")
    if seniority >= 0.7:
        strengths.append("senior-level title aligned to JD requirements")
    if avail_days < 30:
        strengths.append("actively job-seeking (last active < 30 days)")
    if edu_tier in ("tier_1", "tier_2"):
        strengths.append(f"top-tier education ({edu_tier})")
    if interview >= 0.8:
        strengths.append(f"high interview completion rate ({interview:.0%})")
    if not strengths:
        strengths = [f"{yoe}y experience in AI/ML domain"]
    strengths = strengths[:3]

    # Build risk factors
    risks = []
    if honeypot > 0.2:
        risks.append("minor data inconsistency flag — recommend manual verification")
    if avail_days > 90:
        risks.append(f"reduced recent activity ({avail_days}d since last platform login)")
    if yoe > 12:
        risks.append("over-experienced vs. 5–9y JD band — may expect senior+ compensation")

    # Compose reasoning sentence
    s1 = f"{title} with {yoe}y experience, specialising in {skills_str}."
    if vec_db and github >= 50:
        s2 = ("Demonstrates vector DB expertise and active GitHub contributions, "
              "closely matching the Senior AI Engineer JD requirements.")
    elif vec_db:
        s2 = ("Direct vector DB experience is a key differentiator for this "
              "embedding/retrieval-focused role.")
    elif tech_score >= 0.6:
        s2 = f"High technical depth score ({tech_score:.2f}) across JD-critical AI/ML skills."
    else:
        cosine = float(row.get("nlp_jd_cosine", 0))
        s2 = (f"Strong JD semantic alignment (cosine {cosine:.2f}) "
              "with relevant AI engineering background.")

    reasoning = f"{s1} {s2}"
    if risks:
        reasoning += f" Note: {risks[0]}."

    return {"reasoning": reasoning, "strengths": strengths, "risks": risks}


def generate_explainability_df(df_top100: pd.DataFrame) -> pd.DataFrame:
    """Generate a full explainability DataFrame for all top-100 candidates.

    Args:
        df_top100: Top-100 DataFrame with all feature columns and profile data.

    Returns:
        DataFrame with columns:
            rank, candidate_id, score, reasoning, strengths, risks, drivers
    """
    contributions = compute_contributions(df_top100)
    records = []
    for idx, (_, row) in enumerate(df_top100.iterrows()):
        exp = make_reasoning(row)
        records.append({
            "rank":         int(row["rank"]),
            "candidate_id": row["candidate_id"],
            "score":        float(row["score"]),
            "reasoning":    exp["reasoning"],
            "strengths":    exp["strengths"],
            "risks":        exp["risks"],
            "drivers":      get_top_drivers(contributions, idx),
        })

    result = pd.DataFrame(records)[
        ["rank", "candidate_id", "score", "reasoning", "strengths", "risks", "drivers"]
    ]

    generic = (result["reasoning"].str.len() < 30).sum()
    logger.info(
        "Explainability generated: %d candidates | generic=%d",
        len(result), generic
    )
    return result
