# 🤖 Redrob AI Candidate Ranking System

An end-to-end AI-powered candidate ranking platform built for the **Redrob AI Candidate Discovery & Ranking Challenge**.

The system takes a job description and a candidate dataset, then ranks candidates using a multi-stage pipeline that combines:

* Job Description Understanding
* Candidate Auditing and Feature Engineering
* Hybrid Retrieval (BM25 + Dense Embeddings)
* Reciprocal Rank Fusion (RRF)
* Cross-Encoder Reranking
* Explainable Reasoning Generation

> **Note:** This project is specifically designed around the **challenge-provided Job Description and Candidate Schema** supplied by the hackathon organizers. The ranking logic is optimized for this hiring task and data format.

---

# 🚀 Live Demo

**Streamlit Application**

https://redrob-ai-candidate-ranker.streamlit.app/

---

# 📂 GitHub Repository

https://github.com/7Ayush-cyber/redrob-ai-candidate-ranker

---

# 📌 Project Overview

Traditional resume screening often relies on keyword matching, which frequently overlooks semantic relevance, career progression, recruiter signals, and overall candidate suitability.

This project addresses these limitations through a multi-stage AI ranking pipeline that combines lexical retrieval, semantic search, feature engineering, Cross-Encoder reranking, and evidence-based reasoning to identify the strongest candidates for a given role.

The repository contains:

* 📓 Research notebooks documenting the complete development workflow.
* ⚙️ Production-ready Python modules.
* 🖥️ Interactive Streamlit demonstration.
* 📦 Final submission generation pipeline.

---

# ✨ Key Features

* 📄 Job Description Compass Generation
* 🔍 Candidate Audit & Schema Validation
* 👤 Rich Candidate Feature Engineering
* 📚 BM25 Lexical Retrieval
* 🧠 Dense Semantic Retrieval using Sentence Transformers
* 🔀 Reciprocal Rank Fusion (RRF)
* 🎯 Cross-Encoder Reranking
* ⚠️ Candidate Risk & Trap Filtering
* 💬 Explainable Candidate Reasoning
* 📊 Interactive Streamlit Dashboard
* 📥 Submission CSV Generation
* ⚡ Supports both small demo datasets and the original competition dataset (locally)

---

# 🏗️ Pipeline Architecture

```text
Job Description
        │
        ▼
JD Compass Generation
        │
        ▼
Candidate Audit
        │
        ▼
Candidate Feature Engineering
        │
        ▼
Hybrid Retrieval
(BM25 + Dense Embeddings)
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

# 🧠 Models & Methods

| Component         | Model / Method                       |
| ----------------- | ------------------------------------ |
| Dense Embeddings  | BAAI/bge-small-en-v1.5               |
| Cross-Encoder     | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Lexical Retrieval | BM25                                 |
| Dense Retrieval   | FAISS                                |
| Fusion Strategy   | Reciprocal Rank Fusion (RRF)         |

---

# 📂 Repository Structure

```text
redrob-ai-candidate-ranker/
│
├── notebooks/                  # Research and experimentation notebooks
├── src/                        # Production source code
├── artifacts/                  # Intermediate pipeline artifacts
├── outputs/                    # Final submission and reports
├── data/                       # User-provided datasets (not included)
│
├── streamlit_app.py            # Streamlit demo application
├── rank.py                     # CLI entry point
├── requirements.txt
├── submission_metadata.yaml
└── README.md
```

---

# ⚠️ Dataset Information

The original **Redrob Challenge dataset is NOT included** in this repository.

This is because:

* The original dataset contains approximately **100,000 candidate profiles**, making it too large for a standard GitHub repository.
* The dataset is provided as part of the competition and should be obtained from the challenge organizers.

To reproduce the complete pipeline locally, create a folder named **`data/`** in the project root and place the required files inside it.

> **Note:** The Streamlit application is intended as an interactive demonstration. Users are encouraged to upload smaller candidate datasets (e.g., 50–2,000 candidates) for faster processing. Larger datasets are supported locally but may require significantly longer preprocessing time.

---

# ⚙️ Installation

## Clone the Repository

```bash
git clone https://github.com/7Ayush-cyber/redrob-ai-candidate-ranker.git

cd redrob-ai-candidate-ranker
```

## Create a Virtual Environment

### Windows

```bash
python -m venv .venv

.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv

source .venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Streamlit Demo

```bash
streamlit run streamlit_app.py
```

The Streamlit application allows users to:

* Upload a candidate dataset
* Execute the complete ranking pipeline
* View the final leaderboard
* Search ranked candidates
* Inspect score distributions
* Download the generated submission CSV

For the best experience, smaller datasets are recommended during demonstration.

---

# ▶️ Running the Command-Line Pipeline

```bash
python rank.py
```

This command generates the final competition submission CSV using the precomputed pipeline artifacts.

---

# 📊 Generated Outputs

The pipeline produces:

* Candidate Audit Reports
* Engineered Feature Tables
* Hybrid Retrieval Shortlists
* Cross-Encoder Reranked Candidates
* Final Reasoning Shortlists
* Submission Summary
* Final Competition Submission CSV

---

# 📓 Development Workflow

The notebooks document the complete development lifecycle:

1. Data Audit & Schema Validation
2. Job Description Analysis (JD Compass)
3. Candidate Parsing & Feature Engineering
4. Hybrid Retrieval Baseline
5. Cross-Encoder Reranking
6. Reasoning Generation & Submission
7. Evaluation & Ablation Studies

---

