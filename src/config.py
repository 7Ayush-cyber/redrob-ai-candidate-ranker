from pathlib import Path


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SRC_DIR = PROJECT_ROOT / "src"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Create these folders when needed
REQUIRED_DIRS = [
    DATA_DIR,
    NOTEBOOKS_DIR,
    SRC_DIR,
    ARTIFACTS_DIR,
    OUTPUTS_DIR,
]


# Input files
CANDIDATES_JSONL = DATA_DIR / "candidates.jsonl"
CANDIDATES_JSONL_GZ = DATA_DIR / "candidates.jsonl.gz"
SAMPLE_CANDIDATES_JSON = DATA_DIR / "sample_candidates.json"
CANDIDATE_SCHEMA_JSON = DATA_DIR / "candidate_schema.json"
JOB_DESCRIPTION_DOCX = DATA_DIR / "job_description.docx"
REDROB_SIGNALS_DOCX = DATA_DIR / "redrob_signals_doc.docx"
SUBMISSION_SPEC_DOCX = DATA_DIR / "submission_spec.docx"
SAMPLE_SUBMISSION_CSV = DATA_DIR / "sample_submission.csv"
VALIDATE_SUBMISSION_PY = DATA_DIR / "validate_submission.py"

# Artifact files
AUDIT_PARQUET = ARTIFACTS_DIR / "candidate_audit_with_flags.parquet"
AUDIT_CSV = ARTIFACTS_DIR / "candidate_audit_with_flags.csv"
MISSING_VALUE_REPORT = ARTIFACTS_DIR / "missing_value_report.csv"
AUDIT_SUMMARY_JSON = ARTIFACTS_DIR / "audit_summary.json"

# Job Description Artifacts
JOB_DESCRIPTION_DOCX = DATA_DIR / "job_description.docx"
JOB_DESCRIPTION_CLEANED = ARTIFACTS_DIR / "job_description_cleaned.txt"
JD_COMPASS_JSON = ARTIFACTS_DIR / "jd_compass.json"

FEATURES_PARQUET = ARTIFACTS_DIR / "candidate_features.parquet"
FEATURES_CSV = ARTIFACTS_DIR / "candidate_features.csv"
FEATURES_SUMMARY_JSON = ARTIFACTS_DIR / "candidate_features_summary.json"

RETRIEVAL_SHORTLIST_PARQUET = ARTIFACTS_DIR / "candidate_retrieval_shortlist.parquet"
RETRIEVAL_SHORTLIST_CSV = ARTIFACTS_DIR / "candidate_retrieval_shortlist.csv"
RETRIEVAL_METADATA_JSON = ARTIFACTS_DIR / "retrieval_metadata.json"

CANDIDATE_EMBEDDINGS = ARTIFACTS_DIR / "candidate_embeddings.npy"
FAISS_INDEX = ARTIFACTS_DIR / "candidate_faiss.index" 


RERANKED_PARQUET = ARTIFACTS_DIR / "reranked_candidates.parquet"
RERANKED_CSV = ARTIFACTS_DIR / "reranked_candidates.csv"
FINAL_SHORTLIST_PARQUET = ARTIFACTS_DIR / "final_shortlist_for_reasoning.parquet"
RERANK_SUMMARY_JSON = ARTIFACTS_DIR / "rerank_summary.json"

EVALUATION_SUMMARY_JSON = OUTPUTS_DIR / "evaluation_summary.json"
EVALUATION_REPORT_CSV = OUTPUTS_DIR / "evaluation_report_table.csv"

# Submission settings
TEAM_ID = "team_PPT_Se_Placement" 


# Final submission files
FINAL_SUBMISSION_CSV = OUTPUTS_DIR / f"{TEAM_ID}.csv"
SUBMISSION_SUMMARY_JSON = OUTPUTS_DIR / "submission_summary.json"

# Model names
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Retrieval settings
BM25_TOP_K = 2000
DENSE_TOP_K = 2000
FINAL_SHORTLIST_SIZE = 1500
SUBMISSION_TOP_K = 100

RRF_K = 60

# Scoring / filtering thresholds
IDEAL_YEARS_MIN = 5
IDEAL_YEARS_MAX = 9
WIDE_YEARS_MIN = 3
WIDE_YEARS_MAX = 12

HIGH_RISK_THRESHOLD = 4
HIGH_TRAP_THRESHOLD = 2.5

LOW_RESPONSE_RATE_THRESHOLD = 0.25
SLOW_RESPONSE_THRESHOLD_HOURS = 150
SHORT_NOTICE_THRESHOLD_DAYS = 90


# Utility
def ensure_directories() -> None:
    """Create project folders if they do not already exist."""
    for directory in REQUIRED_DIRS:
        directory.mkdir(parents=True, exist_ok=True)




if __name__ == "__main__":
    ensure_directories()

    print("=" * 60)
    print("Project Configuration Loaded Successfully")
    print("=" * 60)

    print("\nProject Root:")
    print(PROJECT_ROOT)

    print("\nData Directory:")
    print(DATA_DIR)

    print("\nArtifacts Directory:")
    print(ARTIFACTS_DIR)

    print("\nOutputs Directory:")
    print(OUTPUTS_DIR)

    print("\nEmbedding Model:")
    print(EMBEDDING_MODEL_NAME)

    print("\nCross Encoder:")
    print(CROSS_ENCODER_MODEL_NAME)

    print("\nSubmission Size:")
    print(SUBMISSION_TOP_K)

    print("\nAll required folders are ready.")
