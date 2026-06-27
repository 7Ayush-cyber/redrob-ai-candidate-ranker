"""
rank.py

Fast final entry point for the Redrob ranking submission.

This script assumes the heavy stages have already been precomputed:
audit -> jd_compass -> candidate_features -> retrieval -> reranker

It only generates the final top-100 submission CSV from the
precomputed shortlist and reasoning stage.
"""

import argparse
from pathlib import Path
from shutil import copyfile

from src import reasoning
from src.config import FINAL_SHORTLIST_PARQUET, FINAL_SUBMISSION_CSV, JD_COMPASS_JSON


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate the final Redrob submission CSV."
    )
    parser.add_argument(
        "--candidates",
        type=str,
        default=None,
        help="Optional path to candidates.jsonl or candidates.jsonl.gz. Kept for CLI compatibility.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(FINAL_SUBMISSION_CSV),
        help="Output CSV path. Default is the configured final submission file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out_path = Path(args.out)

    print("=" * 70)
    print("REDROB AI CANDIDATE RANKER")
    print("=" * 70)

    if args.candidates:
        candidates_path = Path(args.candidates)
        if not candidates_path.exists():
            raise FileNotFoundError(f"Candidates file not found: {candidates_path}")
        print("Candidates file:", candidates_path)

    if not FINAL_SHORTLIST_PARQUET.exists():
        raise FileNotFoundError(
            f"Missing final shortlist artifact: {FINAL_SHORTLIST_PARQUET}. "
            "Run reranker.py first."
        )

    if not JD_COMPASS_JSON.exists():
        raise FileNotFoundError(
            f"Missing JD compass artifact: {JD_COMPASS_JSON}. "
            "Run jd_compass.py first."
        )

    print("Final shortlist exists:", FINAL_SHORTLIST_PARQUET.exists())
    print("JD compass exists:", JD_COMPASS_JSON.exists())
    print("\nRunning final submission pipeline...\n")

    # reasoning.main()
    reasoning.main(strict_submission=True)

    produced_path = Path(FINAL_SUBMISSION_CSV)

    if out_path.resolve() != produced_path.resolve():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        copyfile(produced_path, out_path)
        print("Copied final submission to:", out_path)
    else:
        print("Final submission written to:", produced_path)

    print("\n" + "=" * 70)
    print("Pipeline completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()