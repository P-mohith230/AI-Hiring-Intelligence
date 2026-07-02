"""
feature_engineering.py — 18 engineered features for candidate ranking.

Each feature is computed from the flat candidates DataFrame produced by
preprocessing.py. All output features are prefixed with "f_" and normalised
to [0, 1]. Features are documented with their business rationale.

Feature list:
    f_yoe_fit              — JD experience band fit (5–9y optimal)
    f_tech_depth           — Technical depth from NLP analysis
    f_nlp_cosine           — TF-IDF cosine similarity to JD
    f_availability         — Hiring availability composite
    f_github               — Open-source engineering signal
    f_recruiter_interest   — Market demand signal (views + saves)
    f_interview_completion — Process reliability signal
    f_title_seniority      — Title-level seniority score
    f_vector_db            — Vector DB expertise flag
    f_career_stability     — Average tenure consistency
    f_startup_exposure     — Early-stage company exposure
    f_education_tier       — Academic institution quality
    f_endorsement_quality  — Endorsements per skill claimed
    f_profile_completeness — Profile data completeness
    f_india_readiness      — India hybrid location fit
    f_network_strength     — Professional network size
    f_honeypot_risk        — Inverse credibility signal (penalty)
    f_job_search_activity  — Active job-seeking behaviour
"""

import re
import logging
import numpy as np
import pandas as pd
from typing import Dict

logger = logging.getLogger("ai_hiring.features")

# Seniority title map: keyword → normalised score
SENIORITY_MAP: Dict[str, float] = {
    "chief": 1.0, "vp": 1.0, "vice president": 1.0,
    "director": 0.9, "head of": 0.9,
    "principal": 0.85, "staff": 0.85,
    "senior": 0.75, "sr.": 0.75, "sr ": 0.75,
    "lead": 0.7, "tech lead": 0.8, "team lead": 0.75,
    "manager": 0.65, "engineering manager": 0.8,
    "engineer": 0.5, "scientist": 0.5, "analyst": 0.3,
    "intern": 0.0, "trainee": 0.0, "junior": 0.1, "jr.": 0.1,
    "associate": 0.2,
}

EDUCATION_TIER_MAP: Dict[str, float] = {
    "tier_1": 1.0,
    "tier_2": 0.80,
    "tier_3": 0.60,
    "tier_4": 0.40,
}

VECTOR_DB_TERMS = [
    "faiss", "pinecone", "qdrant", "weaviate", "chromadb",
    "milvus", "vespa", "pgvector", "annoy", "hnsw", "opensearch",
]


def _seniority_score(title: str) -> float:
    """Map a job title string to a normalised seniority score [0, 1].

    Args:
        title: Raw job title string.

    Returns:
        Float seniority score in [0.0, 1.0].
    """
    tl = str(title).lower()
    best = 0.4
    for kw, sc in SENIORITY_MAP.items():
        if kw in tl:
            best = max(best, sc)
    return best


def _has_vector_db(text: str) -> int:
    """Return 1 if text contains any vector database term, else 0.

    Args:
        text: Concatenated skills + career description text.

    Returns:
        Binary int (0 or 1).
    """
    tl = str(text).lower()
    return int(any(t in tl for t in VECTOR_DB_TERMS))


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all 18 engineered features and append them to the DataFrame.

    All features are float columns in [0, 1] prefixed with "f_".
    The input DataFrame is NOT modified; a copy is returned.

    Args:
        df: Flat candidates DataFrame from preprocessing + NLP stages.

    Returns:
        DataFrame with 18 new "f_*" feature columns appended.
    """
    df = df.copy()
    logger.info("Engineering 18 features on %d candidates...", len(df))

    # F1 — YOE Fit: penalise under 4y and over 13y; peak at 5–9y
    _yoe = df["years_of_experience"].fillna(0).clip(0, 20)
    df["f_yoe_fit"] = np.where(
        _yoe < 4,  (_yoe / 4.0).clip(0, 1) * 0.6,
        np.where(_yoe <= 9, 0.6 + (_yoe - 4) / 5 * 0.4,
        np.where(_yoe <= 13, 1.0 - (_yoe - 9) / 4 * 0.3, 0.7))
    ).clip(0, 1)

    # F2 — Technical Depth (from NLP phase)
    df["f_tech_depth"] = df.get("tech_depth_score", pd.Series(0, index=df.index)).fillna(0).clip(0, 1)

    # F3 — NLP JD Cosine (normalised by p99)
    _p99 = float(df["nlp_jd_cosine"].quantile(0.99)) if "nlp_jd_cosine" in df.columns else 1.0
    df["f_nlp_cosine"] = (df.get("nlp_jd_cosine", pd.Series(0, index=df.index)).fillna(0) / max(_p99, 1e-6)).clip(0, 1)

    # F4 — Availability: recency + response rate + open-to-work flag
    _login = df["last_active_days_ago"].fillna(365).clip(0, 400)
    _resp  = df["recruiter_response_rate"].fillna(0).clip(0, 1)
    _otw   = df["open_to_work_flag"].fillna(False).astype(int)
    df["f_availability"] = ((1 - _login / 400).clip(0, 1) * 0.45 + _resp * 0.35 + _otw * 0.20).clip(0, 1)

    # F5 — GitHub Activity
    _gh = df["github_activity_score"].fillna(-1)
    df["f_github"] = np.where(_gh < 0, 0.0, (_gh / 100.0).clip(0, 1))

    # F6 — Recruiter Interest
    _saved = df["saved_by_recruiters_30d"].fillna(0).clip(0, 80)
    _views = df["profile_views_30d"].fillna(0).clip(0, 500)
    df["f_recruiter_interest"] = ((_saved / 80.0) * 0.65 + (_views / 500.0) * 0.35).clip(0, 1)

    # F7 — Interview Completion
    _icr   = df["interview_completion_rate"].fillna(0.5).clip(0, 1)
    _oar   = df["offer_acceptance_rate"].fillna(-1)
    _offer = np.where(_oar < 0, 0.5, _oar.clip(0, 1).values)
    df["f_interview_completion"] = (_icr * 0.65 + _offer * 0.35).clip(0, 1)

    # F8 — Title Seniority
    df["f_title_seniority"] = df["current_title"].apply(_seniority_score).clip(0, 1)

    # F9 — Vector DB Flag
    _combined_text = df["skills_text"].fillna("") + " " + df["career_desc_text"].fillna("")
    df["f_vector_db"] = _combined_text.apply(_has_vector_db).astype(float)

    # F10 — Career Stability: avg tenure (years per role)
    _roles = df["career_roles_count"].fillna(1).clip(1, 15)
    _yoe2  = df["years_of_experience"].fillna(1).clip(0.5, 20)
    _avg_t = _yoe2 / _roles
    df["f_career_stability"] = np.where(
        _avg_t < 0.5, 0.1,
        np.where(_avg_t < 1.0, 0.3,
        np.where(_avg_t < 1.5, 0.6,
        np.where(_avg_t < 2.5, 0.9,
        np.where(_avg_t < 4.0, 1.0, 0.8))))
    ).clip(0, 1)

    # F11 — Startup Exposure
    df["f_startup_exposure"] = df["has_startup"].astype(float)

    # F12 — Education Tier
    df["f_education_tier"] = df["edu_tier"].map(EDUCATION_TIER_MAP).fillna(0.40)

    # F13 — Endorsement Quality (endorsements per claimed skill)
    _endorse = df["endorsements_received"].fillna(0).clip(0, 500)
    _skills  = df["skills_count"].fillna(1).clip(1, 80)
    df["f_endorsement_quality"] = ((_endorse / _skills) / 50.0).clip(0, 1)

    # F14 — Profile Completeness
    df["f_profile_completeness"] = (
        (df["headline"].fillna("").str.len() > 10).astype(float) * 0.15 +
        (df["summary"].fillna("").str.len() > 50).astype(float)  * 0.25 +
        (df["skills_count"] > 5).astype(float)                   * 0.20 +
        (df["career_roles_count"] > 1).astype(float)             * 0.20 +
        (df["cert_count"] > 0).astype(float)                     * 0.10 +
        df["linkedin_connected"].fillna(False).astype(float)     * 0.10
    ).clip(0, 1)

    # F15 — India Readiness (location + relocation + work mode)
    _in_india  = df["country"].str.lower().str.contains("india", na=False).astype(float)
    _relocate  = df["willing_to_relocate"].fillna(False).astype(float)
    _hybrid_ok = df["preferred_work_mode"].str.lower().str.contains("hybrid|onsite", na=False).astype(float)
    df["f_india_readiness"] = (_in_india * 0.50 + _relocate * 0.35 + _hybrid_ok * 0.15).clip(0, 1)

    # F16 — Network Strength
    df["f_network_strength"] = (df["connection_count"].fillna(0).clip(0, 2000) / 2000.0).clip(0, 1)

    # F17 — Honeypot Risk (penalty feature — lower is better for candidate)
    _dates_f = df["invalid_dates_count"].fillna(0).clip(0, 5)
    _dur_f   = df["impossible_duration_count"].fillna(0).clip(0, 5)
    _yoe_f   = ((df["years_of_experience"].fillna(0) > 50) |
                (df["years_of_experience"].fillna(0) < 0)).astype(float)
    df["f_honeypot_risk"] = (
        (_dates_f / 5.0) * 0.40 + (_dur_f / 5.0) * 0.35 + _yoe_f * 0.25
    ).clip(0, 1)

    # F18 — Job Search Activity
    _apps   = df["applications_30d"].fillna(0).clip(0, 25)
    _search = df["search_appearance_30d"].fillna(0).clip(0, 500)
    df["f_job_search_activity"] = ((_apps / 25.0) * 0.60 + (_search / 500.0) * 0.40).clip(0, 1)

    feat_cols = [c for c in df.columns if c.startswith("f_")]
    logger.info("Feature engineering complete: %d f_ columns", len(feat_cols))
    return df
