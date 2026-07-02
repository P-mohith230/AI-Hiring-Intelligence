# Methodology

## Approach: Hybrid Ensemble (3 signal types)

1. **Structured features** — 18 handcrafted features from profile data
2. **Semantic NLP** — TF-IDF cosine similarity between candidate text and JD
3. **Behavioral signals** — 23 redrob platform engagement metrics

## Why Hybrid?
- Pure rule-based: misses semantic meaning
- Pure NLP (TF-IDF): favours keyword density over real depth
- Pure behavioral: ignores technical fit

The hybrid pipeline ensures each weakness is compensated by complementary signals.

## Feature Weight Design
Calibrated for NDCG@10 (50% of total score):
- f_tech_depth (18%) — most discriminative for Senior AI Engineer
- f_yoe_fit (15%) — JD specifies 5-9y
- f_nlp_cosine (12%) — semantic JD alignment
- f_availability (10%) — hiring velocity for Series A

## Honeypot Detection Strategy
Multi-rule detection guards against ~80 trap profiles:
- Future-dated career entries
- Role durations > 50 years
- YOE > 50 or < 0
- Skills count > 80
- High skill count + near-zero endorsements

Gate threshold: honeypot_risk_score > 0.3 — excluded from eligible pool.
Result: 0 honeypots in final top-100.