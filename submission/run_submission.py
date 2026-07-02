#!/usr/bin/env python3
"""
run_submission.py - End-to-end reproducible submission pipeline.
AI-Hiring-Intelligence: Redrob Intelligent Candidate Discovery & Ranking

Usage:
    python submission/run_submission.py --data path/to/candidates.jsonl

Runtime:    < 5 minutes on any modern CPU
Memory:     < 2 GB RAM
GPU:        Not required (CPU-only)
Network:    Not required (fully offline)
Seed:       42 (fully deterministic)
"""

import sys, time, math, re, json, gc, logging, argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, date
from collections import Counter
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("run_submission")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
TODAY = date.today()

JD_TEXT = """
Senior AI Engineer Founding Team Redrob AI Series A AI-native talent intelligence platform
Pune Noida India Hybrid five to nine years experience embedding retrieval ranking LLMs fine-tuning
production deployment vector database FAISS Pinecone Qdrant Weaviate ChromaDB Python machine learning
MLOps evaluation NDCG MRR MAP BM25 hybrid retrieval dense sparse fusion sentence transformers
semantic search re-ranking LLM PyTorch transformers huggingface training inference startup founding
"""

RANKING_WEIGHTS = {
    "f_yoe_fit": 0.15, "f_tech_depth": 0.18, "f_nlp_cosine": 0.12,
    "f_vector_db": 0.08, "f_github": 0.07, "f_availability": 0.10,
    "f_title_seniority": 0.06, "f_startup_exposure": 0.05,
    "f_recruiter_interest": 0.05, "f_interview_completion": 0.05,
    "f_india_readiness": 0.03, "f_career_stability": 0.02,
    "f_education_tier": 0.02, "f_profile_completeness": 0.01,
    "f_job_search_activity": 0.01,
}

STOPWORDS = set("a an the is are was were be been have has had do does did will would could should may might must can of in on at to for from with by about as into through during before after this that these those i me my we our you your he she it they them what which who when where why how all each every also just only very most more than any now use used using make made".split())

ML_KEYWORDS = {
    "faiss": 4, "pinecone": 4, "qdrant": 4, "weaviate": 4, "chromadb": 4,
    "milvus": 4, "bm25": 4, "ndcg": 4, "embedding": 3, "retrieval": 3,
    "ranking": 3, "llm": 3, "fine-tuning": 3, "transformer": 3,
    "huggingface": 3, "pytorch": 3, "python": 3, "bert": 2, "mrr": 3,
    "tensorflow": 2, "kubernetes": 2, "docker": 2, "mlflow": 2,
}

VECTOR_DB_TERMS = ["faiss","pinecone","qdrant","weaviate","chromadb","milvus","vespa","pgvector","annoy","hnsw","opensearch"]
SENIORITY_MAP = {"chief":1.0,"vp":1.0,"director":0.9,"head of":0.9,"principal":0.85,"staff":0.85,"senior":0.75,"lead":0.7,"tech lead":0.8,"manager":0.65,"engineer":0.5,"intern":0.0,"trainee":0.0,"junior":0.1,"associate":0.2}
EDU_TIER_MAP = {"tier_1":1.0,"tier_2":0.80,"tier_3":0.60,"tier_4":0.40}


def tokenize(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\\s\\-]", " ", text)
    return re.findall(r"\\b[a-z][a-z0-9\\-]{2,}\\b", text)

def build_tfidf(docs, max_features=8000):
    df_counts = Counter()
    for doc in docs:
        df_counts.update(set(tokenize(doc)) - STOPWORDS)
    N = len(docs)
    vocab_terms = [w for w,_ in df_counts.most_common(max_features) if df_counts[w]>=5]
    vocab = {w:i for i,w in enumerate(vocab_terms)}
    idf = np.array([math.log((N+1)/(df_counts[w]+1))+1 for w in vocab_terms], dtype=np.float32)
    return vocab, idf

def encode_text(text, vocab, idf):
    tokens = [t for t in tokenize(text) if t not in STOPWORDS]
    tf = Counter(tokens)
    total = max(1, len(tokens))
    vec = np.zeros(len(vocab), dtype=np.float32)
    for w, c in tf.items():
        if w in vocab:
            vec[vocab[w]] = (c/total)*idf[vocab[w]]
    norm = np.linalg.norm(vec)
    return vec/norm if norm>0 else vec

def flatten_candidate(c):
    p = c.get("profile",{}) or {}
    sig = c.get("redrob_signals",{}) or {}
    skills = c.get("skills",[]) or []
    career = c.get("career_history",[]) or []
    edu    = c.get("education",[]) or []
    co_sizes   = [r.get("company_size","") for r in career]
    industries = list({r.get("industry","") for r in career if r.get("industry")})
    all_desc   = " ".join((r.get("description","") or "")[:300] for r in career)
    inv_d, imp_d = 0, 0
    for r in career:
        if (r.get("duration_months") or 0) > 600: imp_d += 1
        for dk in ("start_date","end_date"):
            dv = r.get(dk,"")
            if dv:
                try:
                    if datetime.strptime(str(dv)[:10],"%Y-%m-%d").date() > TODAY: inv_d += 1
                except: pass
    las = sig.get("last_active_date","")
    lad = None
    if las:
        try: lad = (TODAY - datetime.strptime(str(las)[:10],"%Y-%m-%d").date()).days
        except: pass
    sal = sig.get("expected_salary_range_inr_lpa")
    sal_mid = float(((sal.get("min",0) or 0)+(sal.get("max",0) or 0))/2) if isinstance(sal,dict) else (float(sal) if isinstance(sal,(int,float)) else None)
    sas = sig.get("skill_assessment_scores")
    sas_avg = None
    if isinstance(sas,dict):
        vals=[v for v in sas.values() if isinstance(v,(int,float))]
        sas_avg = float(np.mean(vals)) if vals else None
    elif isinstance(sas,(int,float)): sas_avg = float(sas)
    return {
        "candidate_id":c.get("candidate_id",""), "years_of_experience":p.get("years_of_experience"),
        "current_title":p.get("current_title","") or "", "country":p.get("country","") or "",
        "headline":p.get("headline","") or "", "summary":(p.get("summary","") or "")[:400],
        "skills_count":len(skills), "advanced_skills":sum(1 for s in skills if s.get("proficiency")=="advanced"),
        "total_endorsements":sum(s.get("endorsements",0) or 0 for s in skills),
        "skills_text":" ".join(s.get("name","") for s in skills[:50]),
        "career_roles_count":len(career), "career_industries_count":len(industries),
        "career_desc_text":all_desc[:1200], "all_titles_text":" ".join(r.get("title","") or "" for r in career[:10]),
        "has_startup":any(s in ["1-10","11-50","51-200","201-500"] for s in co_sizes),
        "edu_tier":edu[0].get("tier","tier_4") if edu else "tier_4",
        "cert_count":len(c.get("certifications",[]) or []),
        "invalid_dates_count":inv_d, "impossible_duration_count":imp_d,
        "last_active_days_ago":lad, "open_to_work_flag":sig.get("open_to_work_flag"),
        "profile_views_30d":sig.get("profile_views_received_30d"),
        "applications_30d":sig.get("applications_submitted_30d"),
        "recruiter_response_rate":sig.get("recruiter_response_rate"),
        "github_activity_score":sig.get("github_activity_score"),
        "saved_by_recruiters_30d":sig.get("saved_by_recruiters_30d"),
        "interview_completion_rate":sig.get("interview_completion_rate"),
        "offer_acceptance_rate":sig.get("offer_acceptance_rate"),
        "verified_email":sig.get("verified_email"), "verified_phone":sig.get("verified_phone"),
        "linkedin_connected":sig.get("linkedin_connected"),
        "connection_count":sig.get("connection_count"),
        "endorsements_received":sig.get("endorsements_received"),
        "search_appearance_30d":sig.get("search_appearance_30d"),
        "willing_to_relocate":sig.get("willing_to_relocate"),
        "preferred_work_mode":sig.get("preferred_work_mode") or "",
        "skill_assessment_score":sas_avg, "expected_salary_lpa":sal_mid,
    }

def load_candidates(jsonl_path):
    logger.info("Loading candidates from %s", jsonl_path)
    rows = []
    with open(jsonl_path,"r") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if line: rows.append(flatten_candidate(json.loads(line)))
            if (i+1)%25000==0: logger.info("  Loaded %d...", i+1)
    df = pd.DataFrame(rows); del rows; gc.collect()
    logger.info("Loaded %d candidates", len(df))
    return df

def run_nlp(df):
    logger.info("Running NLP analysis...")
    texts = (df["headline"].fillna("")+" "+df["summary"].fillna("")+" "+
             df["skills_text"].fillna("")+" "+df["all_titles_text"].fillna("")+" "+
             df["career_desc_text"].fillna("")).tolist()
    vocab, idf = build_tfidf(texts)
    jd_vec = encode_text(JD_TEXT, vocab, idf)
    cos = np.zeros(len(texts), dtype=np.float32)
    B = 5000
    for i in range(0,len(texts),B):
        bv = np.stack([encode_text(t,vocab,idf) for t in texts[i:i+B]])
        cos[i:i+B] = bv @ jd_vec
        if (i//B)%4==0: logger.info("  NLP: %d/%d", min(i+B,len(texts)), len(texts))
    df = df.copy()
    df["nlp_jd_cosine"] = cos
    ts = np.array([sum(w for k,w in ML_KEYWORDS.items() if k in t.lower()) for t in texts], dtype=np.float32)
    df["tech_depth_score"] = (ts / max(float(ts.max()),1)).clip(0,1)
    def _sen(t):
        tl=str(t).lower()
        return max((sc for kw,sc in SENIORITY_MAP.items() if kw in tl), default=0.4)
    df["title_seniority_score"] = df["current_title"].apply(_sen)
    combined = df["skills_text"].fillna("")+" "+df["career_desc_text"].fillna("")
    df["vector_db_flag"] = combined.apply(lambda t: int(any(v in str(t).lower() for v in VECTOR_DB_TERMS)))
    logger.info("NLP complete")
    return df

def engineer_features(df):
    logger.info("Engineering features...")
    df = df.copy()
    _yoe = df["years_of_experience"].fillna(0).clip(0,20)
    df["f_yoe_fit"] = np.where(_yoe<4,(_yoe/4.0).clip(0,1)*0.6,np.where(_yoe<=9,0.6+(_yoe-4)/5*0.4,np.where(_yoe<=13,1.0-(_yoe-9)/4*0.3,0.7))).clip(0,1)
    _p99 = float(df["nlp_jd_cosine"].quantile(0.99))
    df["f_nlp_cosine"] = (df["nlp_jd_cosine"].fillna(0)/max(_p99,1e-6)).clip(0,1)
    df["f_tech_depth"] = df["tech_depth_score"].fillna(0).clip(0,1)
    _login=df["last_active_days_ago"].fillna(365).clip(0,400); _resp=df["recruiter_response_rate"].fillna(0); _otw=df["open_to_work_flag"].fillna(False).astype(int)
    df["f_availability"] = ((1-_login/400)*0.45+_resp*0.35+_otw*0.20).clip(0,1)
    _gh=df["github_activity_score"].fillna(-1)
    df["f_github"] = np.where(_gh<0,0.0,(_gh/100.0).clip(0,1))
    _saved=df["saved_by_recruiters_30d"].fillna(0).clip(0,80); _views=df["profile_views_30d"].fillna(0).clip(0,500)
    df["f_recruiter_interest"] = ((_saved/80)*0.65+(_views/500)*0.35).clip(0,1)
    _icr=df["interview_completion_rate"].fillna(0.5).clip(0,1); _oar=df["offer_acceptance_rate"].fillna(-1); _off=np.where(_oar<0,0.5,_oar.clip(0,1).values)
    df["f_interview_completion"] = (_icr*0.65+_off*0.35).clip(0,1)
    df["f_title_seniority"] = df["title_seniority_score"].fillna(0.4)
    df["f_vector_db"] = df["vector_db_flag"].astype(float)
    _roles=df["career_roles_count"].fillna(1).clip(1,15); _yoe2=df["years_of_experience"].fillna(1).clip(0.5,20); _avgt=_yoe2/_roles
    df["f_career_stability"] = np.where(_avgt<0.5,0.1,np.where(_avgt<1.0,0.3,np.where(_avgt<1.5,0.6,np.where(_avgt<2.5,0.9,np.where(_avgt<4.0,1.0,0.8))))).clip(0,1)
    df["f_startup_exposure"] = df["has_startup"].astype(float)
    df["f_education_tier"] = df["edu_tier"].map(EDU_TIER_MAP).fillna(0.40)
    _end=df["endorsements_received"].fillna(0).clip(0,500); _sk=df["skills_count"].fillna(1).clip(1,80)
    df["f_endorsement_quality"] = ((_end/_sk)/50.0).clip(0,1)
    df["f_profile_completeness"] = ((df["headline"].fillna("").str.len()>10).astype(float)*0.15+(df["summary"].fillna("").str.len()>50).astype(float)*0.25+(df["skills_count"]>5).astype(float)*0.20+(df["career_roles_count"]>1).astype(float)*0.20+(df["cert_count"]>0).astype(float)*0.10+df["linkedin_connected"].fillna(False).astype(float)*0.10).clip(0,1)
    _inin=df["country"].str.lower().str.contains("india",na=False).astype(float); _rel=df["willing_to_relocate"].fillna(False).astype(float); _hyb=df["preferred_work_mode"].str.lower().str.contains("hybrid|onsite",na=False).astype(float)
    df["f_india_readiness"] = (_inin*0.50+_rel*0.35+_hyb*0.15).clip(0,1)
    df["f_network_strength"] = (df["connection_count"].fillna(0).clip(0,2000)/2000.0).clip(0,1)
    _df2=df["invalid_dates_count"].fillna(0).clip(0,5); _drf=df["impossible_duration_count"].fillna(0).clip(0,5); _yf=((df["years_of_experience"].fillna(0)>50)|(df["years_of_experience"].fillna(0)<0)).astype(float)
    df["f_honeypot_risk"] = ((_df2/5)*0.40+(_drf/5)*0.35+_yf*0.25).clip(0,1)
    _apps=df["applications_30d"].fillna(0).clip(0,25); _srch=df["search_appearance_30d"].fillna(0).clip(0,500)
    df["f_job_search_activity"] = ((_apps/25)*0.60+(_srch/500)*0.40).clip(0,1)
    _hp=((df["invalid_dates_count"]>0)|(df["impossible_duration_count"]>0)|(df["years_of_experience"].fillna(0)>50)|(df["skills_count"]>80)|((df["skills_count"]>30)&(df["total_endorsements"]<5)))
    df["honeypot_flag"] = _hp.astype(int)
    df["honeypot_risk_score"] = ((df["invalid_dates_count"]>0).astype(float)*0.35+(df["impossible_duration_count"]>0).astype(float)*0.30+(df["years_of_experience"].fillna(0)>50).astype(float)*0.20+(df["skills_count"]>80).astype(float)*0.10+((df["skills_count"]>30)&(df["total_endorsements"]<5)).astype(float)*0.10).clip(0,1)
    logger.info("Features done")
    return df

def run_ranking(df):
    logger.info("Running ranking pipeline...")
    _hp=df["honeypot_risk_score"].fillna(0)>0.3; _sil=(df["last_active_days_ago"].fillna(9999)>270)&(df["recruiter_response_rate"].fillna(0)<0.05)
    _noloc=(~df["country"].str.lower().str.contains("india",na=False))&(df["willing_to_relocate"]!=True); _yoex=(df["years_of_experience"].fillna(0)<2)|(df["years_of_experience"].fillna(0)>18)
    df_elig=df[~(_hp|_sil|_noloc|_yoex)].copy()
    logger.info("Eligible: %d", len(df_elig))
    score=np.zeros(len(df_elig))
    for feat,w in RANKING_WEIGHTS.items():
        if feat in df_elig.columns: score+=df_elig[feat].fillna(0).values*w
    df_elig["raw_score"]=score
    df5k=df_elig.nlargest(5000,"raw_score").copy()
    df5k["enhanced_score"]=(df5k["raw_score"]+df5k["f_vector_db"]*0.03+df5k["f_github"]*0.02+df5k["f_nlp_cosine"]*0.02-(1-df5k["f_availability"]).clip(0)*0.04).clip(0,1)
    df1k=df5k.nlargest(1000,"enhanced_score").copy()
    trust=(df1k["f_recruiter_interest"]*0.40+df1k["f_interview_completion"]*0.30+df1k["f_availability"]*0.30)
    df1k["final_score"]=(df1k["enhanced_score"]+trust*0.05).clip(0,1)
    df100=df1k.nlargest(100,"final_score").reset_index(drop=True)
    df100["rank"]=range(1,101)
    s_min,s_max=float(df100["final_score"].min()),float(df100["final_score"].max())
    df100["score"]=(0.5+(df100["final_score"]-s_min)/max(s_max-s_min,1e-6)*0.5).round(6)
    df100=df100.sort_values("rank")
    for i in range(1,len(df100)):
        if df100.iloc[i]["score"]>df100.iloc[i-1]["score"]: df100.loc[df100.index[i],"score"]=df100.iloc[i-1]["score"]
    hp100=int((df100["honeypot_risk_score"].fillna(0)>0.3).sum())
    assert hp100<=10, f"CRITICAL: {hp100} honeypots in top-100!"
    logger.info("Top-100 selected | score=[%.4f,%.4f] | hp=%d", df100["score"].min(), df100["score"].max(), hp100)
    return df100

def make_reasoning(row):
    yoe=round(float(row.get("years_of_experience",0)),1); title=str(row.get("current_title","Unknown"))
    skills_str=", ".join(s.strip() for s in str(row.get("skills_text","")).split(",") if len(s.strip())>2)[:80] or "ML/AI skills"
    github=float(row.get("github_activity_score",0) or 0); vec_db=bool(row.get("vector_db_flag",False)); tech=float(row.get("f_tech_depth",0))
    s1=f"{title} with {yoe}y experience, specialising in {skills_str}."
    if vec_db and github>=50: s2="Demonstrates vector DB expertise and active GitHub contributions, closely matching the Senior AI Engineer JD."
    elif vec_db: s2="Direct vector DB experience is a key differentiator for this embedding/retrieval-focused role."
    elif tech>=0.6: s2=f"High technical depth score ({tech:.2f}) across JD-critical AI/ML skills."
    else: s2=f"Strong JD semantic alignment (cosine {float(row.get('nlp_jd_cosine',0)):.2f}) with relevant AI engineering background."
    return f"{s1} {s2}"

def validate_submission(df):
    ok = True
    for desc, result in [
        ("100 rows", len(df)==100), ("unique IDs", df["candidate_id"].nunique()==100),
        ("ranks 1-100", set(df["rank"].tolist())==set(range(1,101))),
        ("score [0,1]", (df["score"]>=0).all() and (df["score"]<=1).all()),
        ("monotonic", df.sort_values("rank")["score"].is_monotonic_decreasing),
        ("no missing reasoning", df["reasoning"].notna().all()),
        ("ID format", df["candidate_id"].str.match(r"^CAND_\\d{7}$").all()),
    ]:
        logger.info("  [%s] %s", "PASS" if result else "FAIL", desc)
        if not result: ok = False
    return ok

def main():
    parser = argparse.ArgumentParser(description="AI-Hiring-Intelligence Submission Pipeline")
    parser.add_argument("--data", default="competition_data/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl")
    parser.add_argument("--output", default="submission.csv")
    args = parser.parse_args()
    t0 = time.time()
    logger.info("="*60)
    logger.info("AI-HIRING-INTELLIGENCE - Submission Pipeline | Seed=%d", RANDOM_SEED)
    data_path = Path(args.data)
    if not data_path.exists():
        logger.error("Dataset not found: %s", data_path); sys.exit(1)
    df = load_candidates(data_path)
    df = run_nlp(df)
    df = engineer_features(df)
    df100 = run_ranking(df)
    df100 = df100.copy()
    df100["reasoning"] = df100.apply(make_reasoning, axis=1)
    sub = df100[["candidate_id","rank","score","reasoning"]].copy()
    sub["score"] = sub["score"].clip(0,1).round(4)
    sub["rank"] = sub["rank"].astype(int)
    sub = sub.sort_values("rank").reset_index(drop=True)
    logger.info("Validating...")
    if not validate_submission(sub):
        logger.error("VALIDATION FAILED"); sys.exit(1)
    sub.to_csv(args.output, index=False)
    logger.info("Saved: %s | Runtime: %.1fs | PASS", args.output, time.time()-t0)

if __name__ == "__main__":
    main()
