# Feature Engineering Report

## 18 Engineered Features (all normalised to [0, 1])

| # | Feature | Type | Business Rationale |
|---|---------|------|-------------------|
| 1 | f_yoe_fit | Numeric | JD requires 5-9y; peak score at that band |
| 2 | f_tech_depth | NLP | AI/ML keyword depth adjusted for buzzwords |
| 3 | f_nlp_cosine | NLP | TF-IDF cosine similarity to full JD text |
| 4 | f_availability | Composite | Recency + response rate + open-to-work |
| 5 | f_github | Signal | Engineering credibility (open-source) |
| 6 | f_recruiter_interest | Signal | Market demand (saves + profile views) |
| 7 | f_interview_completion | Signal | Process compatibility signal |
| 8 | f_title_seniority | Categorical | Title-level seniority alignment |
| 9 | f_vector_db | Binary | Core JD requirement: Pinecone/Qdrant/FAISS |
| 10 | f_career_stability | Numeric | Average tenure consistency |
| 11 | f_startup_exposure | Binary | JD preference for early-stage experience |
| 12 | f_education_tier | Ordinal | Academic institution quality proxy |
| 13 | f_endorsement_quality | Ratio | Endorsements per claimed skill |
| 14 | f_profile_completeness | Composite | Platform engagement seriousness |
| 15 | f_india_readiness | Composite | Location + relocation + work mode |
| 16 | f_network_strength | Numeric | Professional network size |
| 17 | f_honeypot_risk | Penalty | Inverse credibility signal (disqualifier) |
| 18 | f_job_search_activity | Composite | Active search behaviour signal |

## Validation
All features validated for:
- Distribution spread (no degenerate all-zero features)
- Correlation with final ranking (confirm discriminative power)
- Independence (reduce redundancy across features)