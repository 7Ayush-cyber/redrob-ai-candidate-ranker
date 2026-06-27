# src/load_data.py

import json
import gzip

import pandas as pd
from docx import Document

from src.config import (
    CANDIDATES_JSONL,
    CANDIDATES_JSONL_GZ,
    SAMPLE_CANDIDATES_JSON,
    CANDIDATE_SCHEMA_JSON,
    JOB_DESCRIPTION_DOCX,
    REDROB_SIGNALS_DOCX,
    SUBMISSION_SPEC_DOCX,
    SAMPLE_SUBMISSION_CSV,
    VALIDATE_SUBMISSION_PY,
)

# Small file readers
def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return records


def read_gz_jsonl(path):
    records = []

    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return records


def read_docx(path):
    doc = Document(path)
    parts = []

    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            parts.append(text)

    return "\n".join(parts)


def read_csv(path):
    return pd.read_csv(path)

# Required file checks
def check_required_files():
    print("=" * 70)
    print("Checking required files")
    print("=" * 70)

    if CANDIDATES_JSONL.exists():
        print(f"{CANDIDATES_JSONL.name:<35} FOUND")
    elif CANDIDATES_JSONL_GZ.exists():
        print(f"{CANDIDATES_JSONL_GZ.name:<35} FOUND")
    else:
        print(f"{'Candidate dataset':<35} MISSING")

    required_files = [
        SAMPLE_CANDIDATES_JSON,
        CANDIDATE_SCHEMA_JSON,
        JOB_DESCRIPTION_DOCX,
        REDROB_SIGNALS_DOCX,
        SUBMISSION_SPEC_DOCX,
        SAMPLE_SUBMISSION_CSV,
        VALIDATE_SUBMISSION_PY,
    ]

    for file_path in required_files:
        status = "FOUND"
        if not file_path.exists():
            status = "MISSING"
        print(f"{file_path.name:<35} {status}")

    print("=" * 70)

# Dataset loaders
def load_candidates():
    """
    Load the full candidate pool.

    We prefer the plain .jsonl file if it exists.
    If not, we fall back to the gzipped version.
    """
    if CANDIDATES_JSONL.exists():
        print(f"Loading candidates from: {CANDIDATES_JSONL}")
        return read_jsonl(CANDIDATES_JSONL)

    if CANDIDATES_JSONL_GZ.exists():
        print(f"Loading candidates from: {CANDIDATES_JSONL_GZ}")
        return read_gz_jsonl(CANDIDATES_JSONL_GZ)

    raise FileNotFoundError("Could not find candidates.jsonl or candidates.jsonl.gz")


def load_sample_candidates():
    print(f"Loading sample candidates from: {SAMPLE_CANDIDATES_JSON}")
    return read_json(SAMPLE_CANDIDATES_JSON)

def load_candidate_sample(n=3):
    candidates = load_candidates()
    return candidates[:n]

def load_candidate_schema():
    print(f"Loading candidate schema from: {CANDIDATE_SCHEMA_JSON}")
    return read_json(CANDIDATE_SCHEMA_JSON)


def load_job_description():
    print(f"Loading job description from: {JOB_DESCRIPTION_DOCX}")
    return read_docx(JOB_DESCRIPTION_DOCX)


def load_redrob_signals_doc():
    print(f"Loading signals doc from: {REDROB_SIGNALS_DOCX}")
    return read_docx(REDROB_SIGNALS_DOCX)


def load_submission_spec():
    print(f"Loading submission spec from: {SUBMISSION_SPEC_DOCX}")
    return read_docx(SUBMISSION_SPEC_DOCX)


def load_sample_submission():
    print(f"Loading sample submission from: {SAMPLE_SUBMISSION_CSV}")
    return read_csv(SAMPLE_SUBMISSION_CSV)


def load_validator_script():
    print(f"Loading validator script path: {VALIDATE_SUBMISSION_PY}")
    return VALIDATE_SUBMISSION_PY

# Helpful summary
def print_dataset_summary():
    """
    Print a small summary of the main files.
    """
    print("=" * 70)
    print("Dataset Summary")
    print("=" * 70)

    candidates = load_candidates()
    sample_candidates = load_sample_candidates()
    schema = load_candidate_schema()
    jd_text = load_job_description()

    print(f"Total candidates loaded: {len(candidates)}")
    print(f"Sample candidates loaded: {len(sample_candidates)}")

    if isinstance(schema, dict):
        print(f"Schema top-level keys: {list(schema.keys())}")

    print(f"Job description length : {len(jd_text):,} characters")
    print(f"Job description words  : {len(jd_text.split()):,}")
    print("=" * 70)

# Quick run
if __name__ == "__main__":
    print("\nRedrob AI Candidate Ranking")
    print("-" * 70)

    check_required_files()

    print()

    print_dataset_summary()

    print("\nLoad Data module executed successfully.")