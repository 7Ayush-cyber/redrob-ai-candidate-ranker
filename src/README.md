# Source Code

This directory contains the production implementation of the **Redrob AI Candidate Ranking System**. The modules are organized according to the end-to-end ranking pipeline, from data loading and auditing to hybrid retrieval, reranking, and final reasoning generation.

---

## Project Modules

### `load_data.py`
Loads candidate datasets, job descriptions, and intermediate artifacts used throughout the pipeline.

### `audit.py`
Performs data validation, schema verification, missing value analysis, duplicate detection, and candidate quality auditing.

### `jd_compass.py`
Processes the job description to extract structured hiring requirements, required skills, preferred skills, and scoring dimensions that guide candidate ranking.

### `candidate_features.py`
Parses candidate profiles and generates structured features including experience, skills, education, certifications, recruiter signals, GitHub activity, and profile completeness.

### `retrieval.py`
Implements the first-stage candidate retrieval pipeline using:

- BM25 lexical retrieval
- Dense semantic retrieval using BGE embeddings
- Reciprocal Rank Fusion (RRF)

### `reranker.py`
Applies Cross-Encoder reranking to refine the hybrid retrieval results and produce a high-quality shortlist.

### `reasoning.py`
Generates explainable candidate summaries, computes final ranking scores, assigns candidate ranks, and produces the submission CSV.

### `config.py`
Contains project-wide configuration variables, model names, retrieval parameters, file paths, and ranking constants.

### `utils.py`
Provides reusable helper functions shared across multiple modules.

### `__init__.py`
Marks the directory as a Python package.

---

# Pipeline Architecture

```text
Candidate Dataset
        │
        ▼
Data Loading
        │
        ▼
Data Audit
        │
        ▼
JD Compass
        │
        ▼
Candidate Feature Engineering
        │
        ▼
Hybrid Retrieval
(BM25 + Dense Retrieval)
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

## Design Philosophy

The source code is intentionally modular, with each stage encapsulated in its own module. This design improves readability, maintainability, reproducibility, and enables individual pipeline components to be developed, tested, and extended independently.
