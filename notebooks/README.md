# 📒 Notebooks

This directory contains the complete experimentation, research, and development workflow for the **Redrob AI Candidate Ranking System**.

Each notebook corresponds to a major stage of the candidate ranking pipeline, documenting the progression from raw candidate data to the final ranked submission.

---

## 📂 Notebook Overview

### 📌 00 — Data Audit & Schema (`00_data_audit_and_schema.ipynb`)

- Dataset exploration
- Schema validation
- Missing value analysis
- Duplicate detection
- Candidate profiling
- Data quality assessment

---

### 📌 01 — JD Compass (`01_jd_compass.ipynb`)

- Job description understanding
- Role summary extraction
- Required skills extraction
- Preferred skills extraction
- Scoring dimensions generation

---

### 📌 02 — Candidate Parsing & Feature Engineering (`02_candidate_parsing_and_features.ipynb`)

- Candidate profile parsing
- Experience feature engineering
- Skills engineering
- Education & certification features
- Recruiter signal engineering
- GitHub activity features
- Profile completeness features

---

### 📌 03 — Hybrid Retrieval Baseline (`03_hybrid_retrieval_baseline.ipynb`)

Implements the first-stage retrieval pipeline using:

- BM25 lexical retrieval
- Dense semantic retrieval (BGE embeddings)
- Reciprocal Rank Fusion (RRF)
- Retrieval evaluation

---

### 📌 04 — Cross-Encoder Reranker (`04_reranker_and_trap_filter.ipynb`)

- Cross-Encoder reranking
- Candidate shortlist refinement
- Retrieval optimization
- Final ranking score generation

---

### 📌 05 — Reasoning Generation & Submission (`05_reasoning_generation_and_submission.ipynb`)

- Explainable candidate reasoning generation
- Final score normalization
- Candidate ranking
- Competition submission generation

---

### 📌 06 — Evaluation & Ablation (`06_evaluation_and_ablation.ipynb`)

- Retrieval evaluation
- Reranking evaluation
- Component-wise ablation studies
- End-to-end pipeline analysis

---

# 🔄 End-to-End Pipeline

```text
Candidate Dataset
        │
        ▼
Data Audit
        │
        ▼
JD Understanding
        │
        ▼
Candidate Feature Engineering
        │
        ▼
Hybrid Retrieval
(BM25 + Dense Semantic Retrieval)
        │
        ▼
Reciprocal Rank Fusion (RRF)
        │
        ▼
Cross-Encoder Reranking
        │
        ▼
Reasoning Generation
        │
        ▼
Final Ranked Submission
```

---

## 🏗️ Production Pipeline

The experimentation performed in these notebooks has been modularized into the production-ready implementation located in the **`src/`** directory.

The production pipeline powers:

- **`rank.py`** – Command-line ranking pipeline
- **`streamlit_app.py`** – Interactive Streamlit demonstration

This separation keeps the repository organized while preserving the complete research and development workflow.
