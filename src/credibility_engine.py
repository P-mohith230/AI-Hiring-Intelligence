"""
credibility_engine.py — Profile Credibility Engine (Phase X).

Answers the question:
    "How much confidence should a recruiter have in this candidate profile
     based on available evidence?"

Produces a Profile Credibility Score (0–100) from 8 analytical modules:
    M1  Career Consistency    — timeline stability, role progression
    M2  Skill Evidence        — skills corroborated by career descriptions
    M3  Technical Depth       — depth of AI/ML engineering language
    M4  Leadership & Ownership — strong vs weak action verb analysis
    M5  Evidence Strength     — quantified achievements and metrics
    M6  Behavioral Confidence — platform engagement signals
    M7  Temporal Consistency  — career timeline logic and gap analysis
    M8  Narrative Intelligence — mindset orientation (technical/research/product)

Important: This module estimates INTERNAL CONSISTENCY only.
It makes no factual accusations and does not access external data sources.
"""

import re
import math
import logging
import numpy as np
import pandas as pd
from typing import Dict, Tuple

logger = logging.getLogger("ai_hiring.credibility")

# Module weights — calibrated for Senior AI Engineer role
CREDIBILITY_WEIGHTS: Dict[str, float] = {
    "m1_career_consistency":     0.10,
    "m2_skill_evidence":         0.14,
    "m3_technical_depth":        0.18,
    "m4_leadership":             0.12,
    "m5_evidence_strength":      0.14,
    "m6_behavioral_confidence":  0.16,
    "m7_temporal_consistency":   0.10,
    "m8_narrative_intelligence": 0.06,
}

MODULE_LABELS: Dict[str, str] = {
    "m1_career_consistency":     "Career Consistency",
    "m2_skill_evidence":         "Skill Evidence",
    "m3_technical_depth":        "Technical Depth",
    "m4_leadership":             "Leadership & Ownership",
    "m5_evidence_strength":      "Evidence Strength",
    "m6_behavioral_confidence":  "Behavioral Confidence",
    "m7_temporal_consistency":   "Temporal Consistency",
    "m8_narrative_intelligence": "Narrative Intelligence",
}

# Technical term lexicons
TECH_TERMS = {
    "python","java","sql","scala","spark","kafka","airflow","docker","kubernetes",
    "tensorflow","pytorch","scikit","hugging","transformers","bert","gpt","llm",
    "faiss","pinecone","qdrant","weaviate","milvus","opensearch","elasticsearch",
    "langchain","ray","mlflow","aws","gcp","azure","fastapi","postgres","mongodb",
    "redis","databricks","snowflake","bigquery",
}

DEEP_TECH = {
    "distributed","microservices","architecture","latency","throughput","sharding",
    "embedding","retrieval","fine-tuning","rlhf","rag","quantization","distillation",
    "attention","transformer","encoder","decoder","contrastive","self-supervised",
    "zero-shot","few-shot","vector","semantic","cosine","ann","hnsw","approximate",
    "nearest","ndcg","mrr","pipeline","orchestration","inference","serving","batching",
    "streaming","asynchronous","concurrent","parallelism","cuda","tensorrt","onnx",
    "profiling","benchmarking","ablation","experiment","published","arxiv","paper",
    "conference","research","dataset","benchmark","evaluation","baseline",
}

BUZZWORDS = {
    "passionate","enthusiastic","dynamic","innovative","synergy","leverage",
    "cutting-edge","state-of-the-art","rockstar","ninja","guru","wizard",
    "proactive","team-player","self-starter","detail-oriented",
}

STRONG_VERBS = {
    "built","designed","architected","led","managed","owned","delivered","scaled",
    "launched","founded","created","implemented","developed","drove","established",
    "defined","pioneered","spearheaded","orchestrated","directed","mentored",
    "hired","grew","transformed","migrated","automated","reduced","improved",
    "increased","optimised","refactored","deployed","shipped","released","published",
}

WEAK_VERBS = {
    "assisted","supported","helped","familiar","exposure","participated",
    "contributed","involved","responsible",
}

QUANTITY_PATTERNS = [
    r"\b\d+[%x]\b", r"\b\d+[km+]\b", r"\b\$\s*\d+",
    r"\b\d+\s*(ms|seconds?|hrs?)\b",
    r"\b\d+\s*(million|billion|crore|lakh)\b",
    r"\b\d+\s*(users?|requests?|qps|rps|tps)\b",
    r"\breduced\s+by\b", r"\bimproved\s+by\b", r"\bincreased\s+by\b",
    r"\b(ndcg|mrr|map|f1|auc|rmse|mae)\s*[=@]\s*\d",
]


def _safe_text(*parts) -> str:
    cleaned = []
    for p in parts:
        if p is None:
            continue
        if isinstance(p, float) and math.isnan(p):
            continue
        cleaned.append(str(p).lower().strip())
    return " ".join(cleaned)


def _minmax(series: pd.Series) -> pd.Series:
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - lo) / (hi - lo)


def compute_all_modules(df: pd.DataFrame) -> pd.DataFrame:
    """Run all 8 credibility scoring modules on the full candidate pool.

    Args:
        df: Full candidates DataFrame with profile + career + signals data.

    Returns:
        Copy of df with 8 new m*_* score columns added (0–100 each).
    """
    df = df.copy()
    n = len(df)
    logger.info("Running 8 credibility modules on %d candidates...", n)

    # M1 — Career Consistency
    roles = df["career_roles_count"].fillna(1).clip(1, 20)
    yoe   = df["years_of_experience"].fillna(1).clip(1, 30)
    role_per_yr = roles / yoe
    stability   = 1.0 - np.abs(role_per_yr - 0.4).clip(0, 1)
    ind_div     = df["career_industries_count"].fillna(1).clip(1, 10)
    ind_score   = 1.0 - ((ind_div - 2).clip(0, 8) / 8) * 0.4
    date_pen    = (df["invalid_dates_count"].fillna(0).clip(0, 5) / 5) * 0.5
    dur_pen     = (df["impossible_duration_count"].fillna(0).clip(0, 5) / 5) * 0.5
    is_senior   = df["current_title"].fillna("").str.lower().str.contains(
        r"senior|staff|lead|principal|vp|director|head", regex=True)
    yoe_gap     = np.where(is_senior & (yoe < 4), 0.3, 0.0)
    career_raw  = stability*0.4 + ind_score*0.2 - (date_pen+dur_pen).clip(0,1)*0.3 - yoe_gap*0.1
    df["m1_career_consistency"] = _minmax(pd.Series(career_raw, index=df.index)) * 100

    # M2 — Skill Evidence
    def _skill_ev(row):
        sk = set(re.findall(r"\b\w+\b", str(row.get("skills_text","")).lower()))
        ca = set(re.findall(r"\b\w+\b", str(row.get("career_desc_text","")).lower()))
        cross = len(sk & ca & TECH_TERMS)
        tech_sk = max(len(sk & TECH_TERMS), 1)
        adv_ratio = float(row.get("advanced_skills",0)) / max(float(row.get("skills_count",1)),1)
        cert_boost = min(float(row.get("cert_count",0)) * 0.05, 0.25)
        return float(np.clip(cross/tech_sk*0.5 + adv_ratio*0.3 + cert_boost*0.2, 0, 1))
    df["m2_skill_evidence"] = _minmax(df.apply(_skill_ev, axis=1)) * 100

    # M3 — Technical Depth
    def _tech_dep(row):
        text = _safe_text(row.get("career_desc_text",""), row.get("headline",""), row.get("summary",""))
        tokens = re.findall(r"\b\w[\w-]*\b", text)
        if not tokens:
            return 0.0
        total = len(tokens)
        deep  = sum(1 for t in tokens if t in DEEP_TECH)
        buzz  = sum(1 for t in tokens if t in BUZZWORDS)
        phase4 = float(row.get("tech_depth_score", 0) or 0)
        return float(np.clip(deep/total*0.5 + phase4*0.4 - min(buzz/total*5, 0.3)*0.1, 0, 1))
    df["m3_technical_depth"] = _minmax(df.apply(_tech_dep, axis=1)) * 100

    # M4 — Leadership & Ownership
    def _lead(row):
        text = _safe_text(row.get("career_desc_text",""), row.get("summary",""))
        tokens = set(re.findall(r"\b\w+\b", text))
        strong = len(tokens & STRONG_VERBS)
        weak   = len(tokens & WEAK_VERBS)
        total  = max(len(tokens), 1)
        sr     = float(row.get("title_seniority_score", 0) or 0)
        ratio  = (strong - weak * 0.5) / total * 20
        return float(np.clip(ratio*0.6 + sr*0.4, 0, 1))
    df["m4_leadership"] = _minmax(df.apply(_lead, axis=1)) * 100

    # M5 — Evidence Strength
    def _evid(row):
        text = _safe_text(row.get("career_desc_text",""), row.get("summary",""))
        if not text.strip():
            return 0.0
        hits = sum(len(re.findall(p, text, re.IGNORECASE)) for p in QUANTITY_PATTERNS)
        density = hits / max(len(text.split()), 1) * 100
        endorse = min(float(row.get("total_endorsements", 0) or 0) / 200, 1.0)
        return float(np.clip(density*0.6 + endorse*0.4, 0, 1))
    df["m5_evidence_strength"] = _minmax(df.apply(_evid, axis=1)) * 100

    # M6 — Behavioral Confidence (from redrob signals)
    last_active = df["last_active_days_ago"].fillna(200).clip(36, 276)
    activity    = 1.0 - (last_active - 36) / (276 - 36)
    resp        = df["recruiter_response_rate"].fillna(0.0).clip(0, 1)
    icr         = df["interview_completion_rate"].fillna(0.5).clip(0, 1)
    gh          = df["github_activity_score"].fillna(-1)
    gh_norm     = np.where(gh < 0, 0.4, (gh / 100.0).clip(0, 1))
    saved       = (df["saved_by_recruiters_30d"].fillna(0).clip(0, 80) / 80)
    apps        = (df["applications_30d"].fillna(0).clip(0, 25) / 25)
    otw         = df["open_to_work_flag"].fillna(False).astype(int)
    verified    = ((df["verified_email"].fillna(False).astype(int) +
                    df["verified_phone"].fillna(False).astype(int)) / 2)
    assess      = df["skill_assessment_score"].fillna(-1)
    assess_norm = np.where(assess < 0, 0.5, (assess / 100.0).clip(0, 1))
    behav_raw   = (
        activity*0.18 + resp*0.18 + icr*0.15 + gh_norm*0.15 +
        saved*0.12 + apps*0.08 + otw*0.06 + verified*0.05 + assess_norm*0.03
    )
    df["m6_behavioral_confidence"] = _minmax(pd.Series(behav_raw.values, index=df.index)) * 100

    # M7 — Temporal Consistency
    invalid_dates = df["invalid_dates_count"].fillna(0).clip(0, 5)
    impossible_d  = df["impossible_duration_count"].fillna(0).clip(0, 5)
    avg_tenure    = df["years_of_experience"].fillna(1).clip(0.5, 20) /                     df["career_roles_count"].fillna(1).clip(1, 20)
    tenure_score  = np.where(
        avg_tenure < 0.5, 0.1,
        np.where(avg_tenure < 1.0, 0.4,
        np.where(avg_tenure < 3.0, 1.0,
        np.where(avg_tenure < 5.0, 0.8, 0.6)))
    )
    date_flag_pen = (invalid_dates / 5) * 0.5
    dur_flag_pen  = (impossible_d / 5) * 0.5
    temporal_raw  = tenure_score * 0.6 - date_flag_pen * 0.25 - dur_flag_pen * 0.15
    df["m7_temporal_consistency"] = _minmax(pd.Series(temporal_raw, index=df.index)) * 100

    # M8 — Narrative Intelligence (mindset orientation)
    MINDSET_LEXICONS = {
        "technical":  {"algorithm","model","system","api","framework","library","code","debug","test"},
        "research":   {"paper","study","experiment","analysis","hypothesis","findings","published","arxiv"},
        "product":    {"user","customer","product","roadmap","feature","launch","growth","retention"},
        "leadership": {"team","hire","mentor","strategy","vision","direction","stakeholder","executive"},
        "engineering":{"architecture","pipeline","deploy","scale","performance","reliability","monitoring"},
        "innovation": {"novel","patent","prototype","zero-to-one","first","invented","pioneered","created"},
    }
    ACTION_VERBS = STRONG_VERBS
    def _narr(row):
        text = _safe_text(row.get("career_desc_text",""), row.get("summary",""), row.get("headline",""))
        tokens = re.findall(r"\b\w+\b", text)
        if not tokens:
            return 0.5
        token_set = set(tokens)
        total = len(tokens)
        mindset_hits = {m: len(token_set & lex) for m, lex in MINDSET_LEXICONS.items()}
        dominant = max(mindset_hits.values()) if mindset_hits else 0
        action_density = len(token_set & ACTION_VERBS) / total * 10
        mindset_diversity = len([v for v in mindset_hits.values() if v > 0]) / len(MINDSET_LEXICONS)
        return float(np.clip(dominant/10*0.4 + action_density*0.35 + mindset_diversity*0.25, 0, 1))
    df["m8_narrative_intelligence"] = _minmax(df.apply(_narr, axis=1)) * 100

    logger.info("All 8 modules computed ✓")
    return df


def compute_credibility_score(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 8 module scores into final credibility score + confidence tier.

    Normalises the weighted sum using p2–p98 clipping for robustness.

    Args:
        df: DataFrame with m1–m8 columns (output of compute_all_modules).

    Returns:
        DataFrame with credibility_score (0–100) and confidence_tier columns.
    """
    score_cols = list(CREDIBILITY_WEIGHTS.keys())
    weights    = np.array([CREDIBILITY_WEIGHTS[c] for c in score_cols])
    cred_matrix = df[score_cols].fillna(0).values
    raw_score   = cred_matrix @ weights

    lo = np.percentile(raw_score, 2)
    hi = np.percentile(raw_score, 98)
    norm = np.clip((raw_score - lo) / max(hi - lo, 1e-6) * 100, 0, 100)

    df = df.copy()
    df["credibility_score"] = np.round(norm, 2)
    df["confidence_tier"]   = df["credibility_score"].apply(_assign_tier)

    logger.info(
        "Credibility scores: mean=%.1f  std=%.1f  High=%d  Watchlist=%d",
        df["credibility_score"].mean(), df["credibility_score"].std(),
        (df["confidence_tier"] == "High").sum(),
        (df["confidence_tier"] == "Watchlist").sum()
    )
    return df


def _assign_tier(score: float) -> str:
    if score >= 75:  return "High"
    if score >= 55:  return "Medium"
    if score >= 35:  return "Low"
    return "Watchlist"


def generate_credibility_explanation(row: pd.Series) -> Dict:
    """Generate a transparent per-candidate credibility explanation.

    Args:
        row: Single candidate row with m1–m8 scores, credibility_score, confidence_tier.

    Returns:
        Dict with credibility_score, confidence_tier, key_strengths,
        potential_concerns, why_this_score, top_contributing_feature.
    """
    score_cols = list(CREDIBILITY_WEIGHTS.keys())
    scores     = {col: float(row.get(col, 0)) for col in score_cols}
    contribs   = {col: CREDIBILITY_WEIGHTS[col] * scores[col] for col in score_cols}
    sorted_c   = sorted(contribs.items(), key=lambda x: x[1], reverse=True)

    strengths = [
        f"{MODULE_LABELS[c]} ({scores[c]:.1f}/100)"
        for c, _ in sorted_c[:3] if scores[c] >= 50
    ]
    concerns = [
        f"{MODULE_LABELS[c]} below confidence threshold ({scores[c]:.1f}/100)"
        for c, _ in sorted_c[-2:] if scores[c] < 35
    ]

    why_parts = []
    if scores.get("m3_technical_depth", 0) >= 60:
        why_parts.append("demonstrated technical depth in career descriptions")
    if scores.get("m6_behavioral_confidence", 0) >= 60:
        why_parts.append("strong platform engagement and recruiter signals")
    if scores.get("m7_temporal_consistency", 0) >= 70:
        why_parts.append("logically consistent career timeline")
    if scores.get("m2_skill_evidence", 0) >= 50:
        why_parts.append("skills supported by career role descriptions")
    if scores.get("m5_evidence_strength", 0) >= 50:
        why_parts.append("quantified achievements present")
    if not why_parts:
        why_parts.append("limited available evidence across credibility dimensions")

    top_col = sorted_c[0][0]
    return {
        "credibility_score":         float(row.get("credibility_score", 0)),
        "confidence_tier":           str(row.get("confidence_tier", "Watchlist")),
        "top_contributing_feature":  f"{MODULE_LABELS[top_col]} ({scores[top_col]:.1f}/100)",
        "key_strengths":             "; ".join(strengths) or "No dominant strengths identified",
        "potential_concerns":        "; ".join(concerns) or "No significant concerns identified",
        "why_this_score":            "Score driven by: " + "; ".join(why_parts[:3]),
    }
