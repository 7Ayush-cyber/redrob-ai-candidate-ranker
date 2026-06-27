# Source Code

This directory contains the production implementation of the Redrob AI Candidate Ranking System.

## Modules

### audit.py
Performs schema validation, duplicate detection, missing value analysis, and candidate data auditing.

### candidate_features.py
Extracts structured candidate features used for retrieval and ranking.

### retrieval.py
Implements the hybrid retrieval pipeline using:
- BM25 lexical retrieval
- Dense semantic retrieval (BGE embeddings)
- Reciprocal Rank Fusion (RRF)

### reranker.py
Uses a Cross-Encoder model to rerank the hybrid retrieval results and generate the final shortlist.

### reasoning.py
Generates explainable candidate summaries, computes final ranking scores, and produces the submission CSV.

### config.py
Contains project-wide configuration variables and constants.

### load_data.py
Utility functions for loading candidate datasets and intermediate artifacts.

### utils.py
Common helper functions shared across the pipeline.
