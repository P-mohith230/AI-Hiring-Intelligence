"""
ranking_engine.py — 5-stage hybrid candidate ranking pipeline.

Architecture:
    Stage 1: Hard filters (honeypots, silent candidates, YOE range, location)
    Stage 2: Structured JD-fit weighted scoring (all eligible candidates)
    Stage 3: Enhanced NLP re-scoring (top 5K)
    Stage 4: Behavioral signal re-ranking (top 1K)
    Stage 5: Final top-100 selection, normalisation, honeypot audit

Ranking weights are calibrated for NDCG@10 (50% of evaluation score).
"""

import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, Tuple

logger = logging.getLogger("ai_hiring.ranking")

# Feature weights — calibrated to maximise NDCG@10
# Critical JD signals receive highest weight
RANKING_WEIGHTS: Dict[str, float] = {
    "f_yoe_fit":               0.15,
    "f_tech_depth":            0.18,
    "f_nlp_cosine":            0.12,
    "f_vector_db":             0.08,
    "f_github":                0.07,
    "f_availability":          0.10,
    "f_title_seniority":       0.06,
    "f_startup_exposure":      0.05,
    "f_recruiter_interest":    0.05,
    "f_interview_completion":  0.05,
    "f_india_readiness":       0.03,
    "f_career_stability":      0.02,
    "f_education_tier":        0.02,
    "f_profile_completeness":  0.01,
    "f_job_search_activity":   0.01,
}

assert abs(sum(RANKING_WEIGHTS.values()) - 1.0) < 0.001, (
    f"Weights must sum to 1.0, got {sum(RANKING_WEIGHTS.values()):.4f}"
)


def apply_hard_filters(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Remove candidates who are definitively ineligible.

    Hard filter rules:
        - Honeypot risk score > 0.3 (competition disqualification protection)
        - Silent candidates (inactive > 270 days AND response rate < 5%)
        - Non-India location AND not willing to relocate
        - Years of experience outside [2, 18]

    Args:
        df: Full candidates DataFrame with all features.

    Returns:
        Tuple of (filtered_df, filter_stats_dict).
    """
    _hp      = df["honeypot_risk_score"].fillna(0) > 0.3
    _silent  = (
        (df["last_active_days_ago"].fillna(9999) > 270) &
        (df["recruiter_response_rate"].fillna(0) < 0.05)
    )
    _no_loc  = (
        (~df["country"].str.lower().str.contains("india", na=False)) &
        (df["willing_to_relocate"] != True)
    )
    _yoe_ext = (
        (df["years_of_experience"].fillna(0) < 2) |
        (df["years_of_experience"].fillna(0) > 18)
    )
    _filtered_out = _hp | _silent | _no_loc | _yoe_ext

    stats = {
        "total_before":    len(df),
        "honeypots":       int(_hp.sum()),
        "silent":          int(_silent.sum()),
        "non_india":       int(_no_loc.sum()),
        "yoe_extreme":     int(_yoe_ext.sum()),
        "total_eligible":  int((~_filtered_out).sum()),
    }

    logger.info(
        "Hard filters: %d → %d eligible  "
        "(hp=%d silent=%d loc=%d yoe=%d)",
        stats["total_before"], stats["total_eligible"],
        stats["honeypots"], stats["silent"],
        stats["non_india"], stats["yoe_extreme"]
    )
    return df[~_filtered_out].copy(), stats


def compute_structured_score(df: pd.DataFrame,
                              weights: Dict[str, float] = None) -> pd.DataFrame:
    """Compute weighted structured JD-fit score for all eligible candidates.

    Args:
        df:      Eligible candidates DataFrame (post hard-filter).
        weights: Feature weight dict (default: RANKING_WEIGHTS).

    Returns:
        DataFrame with raw_score column added.
    """
    if weights is None:
        weights = RANKING_WEIGHTS
    df = df.copy()
    score = np.zeros(len(df), dtype=np.float64)
    for feat, w in weights.items():
        if feat in df.columns:
            score += df[feat].fillna(0).values * w
    df["raw_score"] = score
    logger.info(
        "Structured scores: min=%.4f  med=%.4f  max=%.4f",
        score.min(), np.median(score), score.max()
    )
    return df


def enhanced_rescore_top5k(df: pd.DataFrame) -> pd.DataFrame:
    """Apply NLP and skill boosts to the top-5K candidates.

    Boosts:
        +3% for vector DB experience
        +2% for high GitHub activity
        +2% for strong NLP cosine similarity
    Penalty:
        -4% for very low availability

    Args:
        df: DataFrame with raw_score column (all eligible candidates).

    Returns:
        Top-5K DataFrame with enhanced_score column added.
    """
    df_top5k = df.nlargest(5000, "raw_score").copy()
    df_top5k["enhanced_score"] = (
        df_top5k["raw_score"].values
        + df_top5k["f_vector_db"].values     * 0.03
        + df_top5k["f_github"].values        * 0.02
        + df_top5k["f_nlp_cosine"].values    * 0.02
        - (1 - df_top5k["f_availability"].values).clip(0) * 0.04
    ).clip(0, 1)
    logger.info(
        "Top-5K re-scored: min=%.4f  max=%.4f",
        df_top5k["enhanced_score"].min(),
        df_top5k["enhanced_score"].max()
    )
    return df_top5k


def behavioral_rerank_top1k(df_top5k: pd.DataFrame) -> pd.DataFrame:
    """Re-rank the top-1K candidates using behavioral trust signals.

    Behavioral bonus = 5% of composite recruiter trust score:
        40% recruiter interest + 30% interview completion + 30% availability

    Args:
        df_top5k: Top-5K DataFrame with enhanced_score.

    Returns:
        Top-1K DataFrame with final_score column added.
    """
    df_top1k = df_top5k.nlargest(1000, "enhanced_score").copy()
    _trust = (
        df_top1k["f_recruiter_interest"].values  * 0.40 +
        df_top1k["f_interview_completion"].values * 0.30 +
        df_top1k["f_availability"].values         * 0.30
    )
    df_top1k["final_score"] = (
        df_top1k["enhanced_score"].values + _trust * 0.05
    ).clip(0, 1)
    logger.info(
        "Top-1K behavioral re-rank: min=%.4f  max=%.4f",
        df_top1k["final_score"].min(),
        df_top1k["final_score"].max()
    )
    return df_top1k


def select_final_top100(df_top1k: pd.DataFrame) -> pd.DataFrame:
    """Select, rank, normalise, and validate the final top-100 candidates.

    Score normalisation: [min, max] → [0.5, 1.0]
    Ensures strict monotonically non-increasing scores by rank.

    Args:
        df_top1k: Top-1K DataFrame with final_score.

    Returns:
        Top-100 DataFrame with rank (1–100) and normalised score columns.
        Raises AssertionError if honeypot count exceeds competition threshold.
    """
    df_top100 = df_top1k.nlargest(100, "final_score").copy()
    df_top100 = df_top100.reset_index(drop=True)
    df_top100["rank"] = range(1, 101)

    _s_min = float(df_top100["final_score"].min())
    _s_max = float(df_top100["final_score"].max())
    df_top100["score"] = (
        0.5 + (df_top100["final_score"] - _s_min) / max(_s_max - _s_min, 1e-6) * 0.5
    ).round(6)

    # Enforce strict monotonicity
    df_top100 = df_top100.sort_values("rank")
    for i in range(1, len(df_top100)):
        if df_top100.iloc[i]["score"] > df_top100.iloc[i - 1]["score"]:
            df_top100.loc[df_top100.index[i], "score"] = df_top100.iloc[i - 1]["score"]

    # Honeypot audit — competition DQ threshold is >10%
    hp_count = int((df_top100["honeypot_risk_score"].fillna(0) > 0.3).sum())
    assert hp_count <= 10, (
        f"CRITICAL: {hp_count} honeypots in top-100 — competition DQ threshold exceeded!"
    )

    logger.info(
        "Final top-100 selected | score=[%.4f, %.4f] | honeypots=%d",
        df_top100["score"].min(), df_top100["score"].max(), hp_count
    )
    return df_top100


def run_ranking_pipeline(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Execute the complete 5-stage ranking pipeline end-to-end.

    Args:
        df: Full candidates DataFrame with all engineered features.

    Returns:
        Tuple of (top_100_df, pipeline_stats_dict).

    Pipeline stages:
        1. Hard filters
        2. Structured JD-fit scoring
        3. Enhanced NLP re-scoring (top 5K)
        4. Behavioral re-ranking (top 1K)
        5. Final top-100 selection
    """
    t0 = time.time()
    logger.info("Starting 5-stage ranking pipeline on %d candidates...", len(df))

    df_eligible, filter_stats = apply_hard_filters(df)
    df_scored   = compute_structured_score(df_eligible)
    df_top5k    = enhanced_rescore_top5k(df_scored)
    df_top1k    = behavioral_rerank_top1k(df_top5k)
    df_top100   = select_final_top100(df_top1k)

    elapsed = time.time() - t0
    stats = {
        **filter_stats,
        "runtime_seconds": round(elapsed, 2),
        "top100_score_max": float(df_top100["score"].max()),
        "top100_score_min": float(df_top100["score"].min()),
    }
    logger.info("Pipeline complete in %.1fs", elapsed)
    return df_top100, stats
