"""
preprocessing.py — Data loading, flattening, and honeypot detection.

Responsibilities:
    - Stream-parse 100K JSONL candidate records (memory-efficient)
    - Flatten deeply nested JSON into a flat pandas DataFrame
    - Detect honeypot / disqualification profiles
    - Flag silent candidates
    - Profile key columns (nulls, distributions, cardinality)
"""

import json
import gc
import re
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, date
from typing import Dict, Any, Optional

logger = logging.getLogger("ai_hiring.preprocessing")

TODAY: date = date.today()

# ── Salary helper ─────────────────────────────────────────────────────────────

def _parse_salary(sal) -> Optional[float]:
    """Parse expected_salary_range_inr_lpa (dict or scalar) to midpoint float.

    Args:
        sal: Raw salary value from redrob_signals.

    Returns:
        Float midpoint in LPA, or None if unparseable.
    """
    if isinstance(sal, dict):
        lo = sal.get("min", 0) or 0
        hi = sal.get("max", 0) or 0
        return float((lo + hi) / 2)
    if isinstance(sal, (int, float)):
        return float(sal)
    return None


def _parse_assessment(sas) -> Optional[float]:
    """Parse skill_assessment_scores (dict or scalar) to average float.

    Args:
        sas: Raw assessment value from redrob_signals.

    Returns:
        Float average score, or None if unparseable.
    """
    if isinstance(sas, dict):
        vals = [v for v in sas.values() if isinstance(v, (int, float))]
        return float(np.mean(vals)) if vals else None
    if isinstance(sas, (int, float)):
        return float(sas)
    return None


# ── Core flattening ───────────────────────────────────────────────────────────

def flatten_candidate(c: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a single nested candidate JSON record into a flat dict.

    Args:
        c: Parsed candidate dict from JSONL.

    Returns:
        Flat dict with ~45 scalar/string fields.
    """
    p   = c.get("profile", {}) or {}
    sig = c.get("redrob_signals", {}) or {}

    skills        = c.get("skills", []) or []
    skill_names   = [s.get("name", "") for s in skills]
    skill_adv     = sum(1 for s in skills if s.get("proficiency", "") == "advanced")
    skill_months  = sum(s.get("duration_months", 0) or 0 for s in skills)
    total_endorse = sum(s.get("endorsements", 0) or 0 for s in skills)

    career     = c.get("career_history", []) or []
    co_sizes   = [r.get("company_size", "") for r in career]
    industries = list({r.get("industry", "") for r in career if r.get("industry")})
    all_titles = [r.get("title", "") or "" for r in career]
    all_desc   = " ".join((r.get("description", "") or "")[:300] for r in career)

    _inv_dates, _imp_dur = 0, 0
    for r in career:
        dur = r.get("duration_months", 0) or 0
        if dur > 600:
            _imp_dur += 1
        for dk in ("start_date", "end_date"):
            dv = r.get(dk, "")
            if dv:
                try:
                    if datetime.strptime(str(dv)[:10], "%Y-%m-%d").date() > TODAY:
                        _inv_dates += 1
                except Exception:
                    pass

    edu      = c.get("education", []) or []
    edu_tier = edu[0].get("tier", "tier_4") if edu else "tier_4"
    edu_deg  = edu[0].get("degree", "") if edu else ""

    last_active_days: Optional[int] = None
    _las = sig.get("last_active_date", "")
    if _las:
        try:
            last_active_days = (TODAY - datetime.strptime(str(_las)[:10], "%Y-%m-%d").date()).days
        except Exception:
            pass

    return {
        "candidate_id":              c.get("candidate_id", ""),
        "years_of_experience":       p.get("years_of_experience"),
        "current_title":             p.get("current_title", "") or "",
        "current_company":           p.get("current_company", "") or "",
        "current_company_size":      p.get("current_company_size", "") or "",
        "current_industry":          p.get("current_industry", "") or "",
        "location":                  p.get("location", "") or "",
        "country":                   p.get("country", "") or "",
        "headline":                  p.get("headline", "") or "",
        "summary":                   (p.get("summary", "") or "")[:400],
        "skills_count":              len(skills),
        "advanced_skills":           skill_adv,
        "skill_total_months":        skill_months,
        "total_endorsements":        total_endorse,
        "skills_text":               " ".join(skill_names[:50]),
        "career_roles_count":        len(career),
        "career_industries_count":   len(industries),
        "career_desc_text":          all_desc[:1200],
        "all_titles_text":           " ".join(all_titles[:10]),
        "has_startup":               any(s in ["1-10","11-50","51-200","201-500"] for s in co_sizes),
        "edu_tier":                  edu_tier,
        "edu_degree":                edu_deg,
        "cert_count":                len(c.get("certifications", []) or []),
        "languages_count":           len(c.get("languages", []) or []),
        "invalid_dates_count":       _inv_dates,
        "impossible_duration_count": _imp_dur,
        "last_active_days_ago":      last_active_days,
        "open_to_work_flag":         sig.get("open_to_work_flag"),
        "profile_views_30d":         sig.get("profile_views_received_30d"),
        "applications_30d":          sig.get("applications_submitted_30d"),
        "recruiter_response_rate":   sig.get("recruiter_response_rate"),
        "avg_response_time_hours":   sig.get("avg_response_time_hours"),
        "skill_assessment_score":    _parse_assessment(sig.get("skill_assessment_scores")),
        "connection_count":          sig.get("connection_count"),
        "endorsements_received":     sig.get("endorsements_received"),
        "notice_period_days":        sig.get("notice_period_days"),
        "expected_salary_lpa":       _parse_salary(sig.get("expected_salary_range_inr_lpa")),
        "preferred_work_mode":       sig.get("preferred_work_mode") or "",
        "willing_to_relocate":       sig.get("willing_to_relocate"),
        "github_activity_score":     sig.get("github_activity_score"),
        "search_appearance_30d":     sig.get("search_appearance_30d"),
        "saved_by_recruiters_30d":   sig.get("saved_by_recruiters_30d"),
        "interview_completion_rate": sig.get("interview_completion_rate"),
        "offer_acceptance_rate":     sig.get("offer_acceptance_rate"),
        "verified_email":            sig.get("verified_email"),
        "verified_phone":            sig.get("verified_phone"),
        "linkedin_connected":        sig.get("linkedin_connected"),
    }


def load_candidates(jsonl_path: Path, limit: Optional[int] = None) -> pd.DataFrame:
    """Stream-load candidates.jsonl into a flat pandas DataFrame.

    Memory-efficient: processes one line at a time and garbage-collects.

    Args:
        jsonl_path: Path to candidates.jsonl file.
        limit:      Optional max number of records to load (for testing).

    Returns:
        DataFrame with one row per candidate (~45 columns).

    Raises:
        FileNotFoundError: If jsonl_path does not exist.
    """
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Dataset not found: {jsonl_path}")

    logger.info("Streaming candidates from %s", jsonl_path)
    rows = []
    with open(jsonl_path, "r") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if line:
                rows.append(flatten_candidate(json.loads(line)))
            if limit and i >= limit - 1:
                break
            if (i + 1) % 20000 == 0:
                logger.info("  Loaded %d records...", i + 1)

    df = pd.DataFrame(rows)
    del rows
    gc.collect()
    logger.info("Loaded %d candidates | %d columns | %.1f MB",
                len(df), df.shape[1],
                df.memory_usage(deep=True).sum() / 1e6)
    return df


def detect_honeypots(df: pd.DataFrame) -> pd.DataFrame:
    """Add honeypot_flag and honeypot_risk_score columns to the DataFrame.

    Honeypot profiles have logically impossible data designed to catch
    systems that do not validate input quality. More than 10% honeypots
    in the top-100 submission triggers automatic disqualification.

    Detection rules:
        - Future-dated career entries (invalid_dates_count > 0)
        - Impossible role durations > 50 years (impossible_duration_count > 0)
        - Years of experience > 50 or < 0
        - Unrealistically high skill count (> 80)
        - Keyword stuffers: many skills but near-zero endorsements

    Args:
        df: Candidates DataFrame (output of load_candidates).

    Returns:
        DataFrame with two new columns: honeypot_flag (int 0/1),
        honeypot_risk_score (float 0–1).
    """
    _hp_dates  = df["invalid_dates_count"] > 0
    _hp_dur    = df["impossible_duration_count"] > 0
    _hp_yoe    = (df["years_of_experience"].fillna(0) > 50) |                  (df["years_of_experience"].fillna(0) < 0)
    _hp_skills = df["skills_count"] > 80
    _hp_kw     = (df["skills_count"] > 30) & (df["total_endorsements"] < 5)
    _hp_any    = _hp_dates | _hp_dur | _hp_yoe | _hp_skills | _hp_kw

    df = df.copy()
    df["honeypot_flag"] = _hp_any.astype(int)
    df["honeypot_risk_score"] = (
        _hp_dates.astype(float)  * 0.35 +
        _hp_dur.astype(float)    * 0.30 +
        _hp_yoe.astype(float)    * 0.20 +
        _hp_skills.astype(float) * 0.10 +
        _hp_kw.astype(float)     * 0.10
    ).clip(0, 1)

    logger.info("Honeypots flagged: %d (%.2f%%)", _hp_any.sum(),
                _hp_any.mean() * 100)
    return df


def flag_silent_candidates(df: pd.DataFrame,
                           max_inactive_days: int = 270,
                           min_response_rate: float = 0.05) -> pd.DataFrame:
    """Add silent_flag column for candidates who are unlikely to respond.

    Args:
        df:                DataFrame with last_active_days_ago and recruiter_response_rate.
        max_inactive_days: Days threshold beyond which candidate is considered inactive.
        min_response_rate: Minimum recruiter response rate threshold.

    Returns:
        DataFrame with silent_flag (int 0/1) column added.
    """
    _silent = (
        (df["last_active_days_ago"].fillna(9999) > max_inactive_days) &
        (df["recruiter_response_rate"].fillna(0) < min_response_rate)
    )
    df = df.copy()
    df["silent_flag"] = _silent.astype(int)
    logger.info("Silent candidates flagged: %d (%.1f%%)", _silent.sum(),
                _silent.mean() * 100)
    return df
