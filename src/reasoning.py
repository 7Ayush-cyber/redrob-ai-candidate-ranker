import re
import subprocess
import sys

import numpy as np
import pandas as pd
from src.load_data import *
from src.config import *
from src.utils import *


def main(strict_submission=True):
        
    print("=" * 70)
    print("REASONING GENERATION AND SUBMISSION")
    print("=" * 70)

    check_required_files()

    print("Final shortlist exists:", FINAL_SHORTLIST_PARQUET.exists())
    print("JD compass exists:", JD_COMPASS_JSON.exists())

    final_shortlist_df = load_parquet(FINAL_SHORTLIST_PARQUET)
    jd_compass = load_json(JD_COMPASS_JSON)

    print("final_shortlist_df shape:", final_shortlist_df.shape)
    print("JD role:", jd_compass["jd_summary"]["role_title"])

    """Helper functions"""

    def pick_column(df, columns):
        for col in columns:
           if col in df.columns:
               return col
        raise  KeyError(f"None of these columns were found: {columns}" )

    def unique_ordered(items):
        seen = set()
        result = []
        for item in items:
            item = clean_text(item)
            if item and item.lower() not in seen:
                seen.add(item.lower())
                result.append(item)
        return result

    def count_sentences(text):
        text = clean_text(text)

        if not text:
            return 0

        # Count . only if it ends a real sentence:
        # followed by whitespace + uppercase letter, or end of string.
        sentence_endings = re.findall(
            r"[!?]+|(?<!\d)\.(?!\d)(?=\s+[A-Z]|$)",
            text
        )

        return len(sentence_endings) if sentence_endings else 1

    def extract_matches(text, keywords, max_items=3):
        text = clean_text(text).lower()
        matches = []
        for kw in keywords:
            if kw.lower() in text:
                matches.append(kw)
        return unique_ordered(matches)[:max_items]

    candidate_id_col = pick_column(final_shortlist_df, ["candidate_id"])
    title_col = pick_column(final_shortlist_df, ["current_title_full", "current_title_retrieval", "current_title"])
    company_col = pick_column(final_shortlist_df, ["current_company_full", "current_company_retrieval", "current_company"])
    location_col = pick_column(final_shortlist_df, ["location_full", "location_retrieval", "location"])
    years_col = pick_column(final_shortlist_df, ["years_of_experience_full", "years_of_experience_retrieval", "years_of_experience"])

    profile_text_col = pick_column(final_shortlist_df, ["profile_text_full", "profile_text_retrieval", "profile_text"])
    skill_text_col = pick_column(final_shortlist_df, ["skill_text_full", "skill_text_retrieval", "skill_text"])

    score_col = pick_column(final_shortlist_df, ["final_rerank_score", "cross_encoder_score", "retrieval_stage_score"])

    open_to_work_col = pick_column(final_shortlist_df, ["open_to_work_flag_full", "open_to_work_flag_retrieval", "open_to_work_flag"])
    response_rate_col = pick_column(final_shortlist_df, ["recruiter_response_rate_full", "recruiter_response_rate_retrieval", "recruiter_response_rate"])
    response_time_col = pick_column(final_shortlist_df, ["avg_response_time_hours_full", "avg_response_time_hours_retrieval", "avg_response_time_hours"])
    notice_col = pick_column(final_shortlist_df, ["notice_period_days_full", "notice_period_days_retrieval", "notice_period_days"])
    completeness_col = pick_column(final_shortlist_df, ["profile_completeness_score_full", "profile_completeness_score_retrieval", "profile_completeness_score"])
    seniority_col = pick_column(final_shortlist_df, ["seniority_fit_score_full", "seniority_fit_score_retrieval", "seniority_fit_score"])
    risk_col = pick_column(final_shortlist_df, ["total_risk_score_full", "total_risk_score_retrieval", "total_risk_score"])
    availability_col = pick_column(final_shortlist_df, ["availability_strength_full", "availability_strength_retrieval", "availability_strength"])
    location_fit_col = pick_column(final_shortlist_df, ["location_fit_flag_full", "location_fit_flag_retrieval", "location_fit_flag"])

    print("Rows in shortlist:", len(final_shortlist_df))
    print("Unique candidate ids:", final_shortlist_df[candidate_id_col].nunique())
    print("Duplicate candidate ids:", final_shortlist_df[candidate_id_col].duplicated().sum())

    if len(final_shortlist_df) == 0:
         raise ValueError("Final shortlist is empty.")

    if strict_submission and len(final_shortlist_df) < SUBMISSION_TOP_K:
         raise ValueError("Need at least 100 candidates for the official submission path.")

    assert final_shortlist_df[candidate_id_col].nunique() == len(final_shortlist_df), "Duplicate candidate IDs found."

    print(
        final_shortlist_df[
            [
                candidate_id_col,
                title_col,
                years_col,
                score_col,
                risk_col
            ]
        ].head(10)
    )

    core_keywords = jd_compass["skill_groups"]["core_ai_ml"]
    retrieval_keywords = jd_compass["skill_groups"]["retrieval_ranking"]
    production_keywords = jd_compass["skill_groups"]["production_engineering"]
    evaluation_keywords = jd_compass["skill_groups"]["evaluation_and_quality"]

    def build_reasoning(row):
        title = clean_text(row[title_col]) or "Candidate"
        company = clean_text(row[company_col])
        location = clean_text(row[location_col])
        years = safe_float(row[years_col])

        profile_text = clean_text(row[profile_text_col])
        skill_text = clean_text(row[skill_text_col])
        combined_text = f"{profile_text} {skill_text}"

        matched_core = extract_matches(combined_text, core_keywords, max_items=3)
        matched_retrieval = extract_matches(combined_text, retrieval_keywords, max_items=3)
        matched_production = extract_matches(combined_text, production_keywords, max_items=3)
        matched_evaluation = extract_matches(combined_text, evaluation_keywords, max_items=2)

        open_to_work = bool(int(safe_float(row[open_to_work_col], 0)))
        response_rate = safe_float(row[response_rate_col], 0.0)
        response_time = safe_float(row[response_time_col], 0.0)
        notice_period = safe_float(row[notice_col], 0.0)
        completeness = safe_float(row[completeness_col], 0.0)
        seniority = safe_float(row[seniority_col], 0.0)
        risk_score = safe_float(row[risk_col], 0.0)
        location_fit = int(safe_float(row[location_fit_col], 0.0))

        positive_parts = []
        if matched_retrieval:
            positive_parts.append("retrieval/ranking experience around " + ", ".join(matched_retrieval))
        if matched_production:
            positive_parts.append("production evidence with " + ", ".join(matched_production))
        if matched_evaluation:
            positive_parts.append("evaluation exposure through " + ", ".join(matched_evaluation))
        if matched_core:
            positive_parts.append("core AI/ML background with " + ", ".join(matched_core))

        if years >= 5 and years <= 9 and (matched_retrieval or matched_production):
            sentence1 = (
                f"{title} with {years:.1f} years looks like a strong fit for the Senior AI Engineer role"
                f"{' at ' + company if company else ''}"
                f"{' from ' + location if location else ''}, with "
                + "; ".join(positive_parts[:2])
                + "."
            )
        elif years > 12 and (matched_retrieval or matched_production):
            sentence1 = (
                f"{title} with {years:.1f} years is a bit above the ideal 5 to 9 year band, "
                f"but still relevant because of "
                + "; ".join(positive_parts[:2])
                + "."
            )
        elif years < 5 and (matched_retrieval or matched_production):
            evidence_text = "; ".join(positive_parts[:2])

            sentence1 = (
                f"{title} with {years:.1f} years is slightly junior for the target seniority band, "
                f"but shows relevant evidence through {evidence_text}."
            )
        else:
            sentence1 = (
                f"{title} with {years:.1f} years has some useful signals, "
                f"but the evidence is lighter than the strongest shortlisted profiles."
            )

        concerns = []
        if not open_to_work:
            concerns.append("the candidate is not marked open to work")
        if response_rate < 0.25:
            concerns.append(f"recruiter response rate is only {response_rate:.2f}")
        if response_time > 150:
            concerns.append(f"response time is {response_time:.0f} hours")
        if notice_period > 120:
            concerns.append(f"notice period is {notice_period:.0f} days")
        if completeness < 45:
            concerns.append(f"profile completeness is {completeness:.1f}")
        if risk_score >= 4:
            concerns.append("the profile carries stronger trust risk than cleaner fits")
        if location_fit == 0:
            concerns.append("location is not an obvious match for the preferred cities")

        if open_to_work and response_rate >= 0.5 and notice_period <= 90 and risk_score < 4:
            sentence2 = (
                f"Availability signals look workable, with open-to-work status, "
                f"{response_rate:.2f} recruiter response rate, and {notice_period:.0f}-day notice period."
            )
        elif concerns:
            sentence2 = "The main concern is " + "; ".join(concerns[:2]) + "."
        else:
            sentence2 = "The profile looks usable, with no major red flags from the observed signals."

        reasoning = sentence1.strip()
        reasoning = reasoning + " " + sentence2.strip()
        return reasoning.strip()

    final_shortlist_df = final_shortlist_df.copy()
    final_shortlist_df["reasoning"] = final_shortlist_df.apply(build_reasoning, axis=1)

    print(
        final_shortlist_df[
            [
                candidate_id_col,
                title_col,
                years_col,
                score_col,
                "reasoning"
            ]
        ].head(10)
    )

    submission_path = FINAL_SUBMISSION_CSV

    target_k = min(SUBMISSION_TOP_K, len(final_shortlist_df))

    submission_df = final_shortlist_df[[candidate_id_col, score_col, "reasoning"]].copy()
    submission_df = submission_df.head(target_k).copy()

    submission_df = submission_df.sort_values(
        by=[score_col, candidate_id_col],
        ascending=[False, True]
    ).reset_index(drop=True)

    submission_df["rank"] = np.arange(1, len(submission_df) + 1)

    scores = submission_df[score_col].astype(float).to_numpy()

    if scores.max() > scores.min():
        submission_df["score"] = (scores - scores.min()) / (scores.max() - scores.min())
    else:
        submission_df["score"] = np.ones_like(scores)

    submission_df = submission_df[["candidate_id", "rank", "score", "reasoning"]].copy()
    submission_df.columns = ["candidate_id", "rank", "score", "reasoning"]

    print(submission_df.head(10))

    submission_df["reasoning_sentence_count"] = submission_df["reasoning"].apply(count_sentences)
    print(submission_df["reasoning_sentence_count"].value_counts().sort_index())

    print(
        submission_df.loc[
            submission_df["reasoning_sentence_count"] > 2,
            ["candidate_id", "reasoning", "reasoning_sentence_count"]
        ]
    )

    print("Final submission rows:", len(submission_df))
    print("Unique candidate ids:", submission_df["candidate_id"].nunique())
    print("Duplicate candidate ids:", submission_df["candidate_id"].duplicated().sum())
    print("Ranks:", submission_df["rank"].min(), "to", submission_df["rank"].max())

    if strict_submission:
        assert len(submission_df) == SUBMISSION_TOP_K, "Submission must have exactly 100 rows."
        assert submission_df["candidate_id"].nunique() == SUBMISSION_TOP_K, "Candidate IDs must be unique."
        assert list(submission_df["rank"]) == list(range(1, SUBMISSION_TOP_K + 1)), "Ranks must be 1 through 100."
    else:
        assert submission_df["candidate_id"].nunique() == len(submission_df), "Candidate IDs must be unique."
        assert list(submission_df["rank"]) == list(range(1, len(submission_df) + 1)), "Ranks must be consecutive."

    assert submission_df["score"].is_monotonic_decreasing, "Scores must be non-increasing."
    assert submission_df["reasoning"].str.strip().ne("").all(), "Reasoning cannot be empty."
    assert submission_df["reasoning_sentence_count"].between(1, 2).all(), "Reasoning must be 1 to 2 sentences."

    print(submission_df[["candidate_id", "rank", "score", "reasoning"]].head(10))

    final_submission_df = submission_df[["candidate_id", "rank", "score", "reasoning"]].copy()

    save_csv(final_submission_df, FINAL_SUBMISSION_CSV)

    print("Saved clean submission to:", FINAL_SUBMISSION_CSV)

    print("\nFinal CSV columns:")
    print(final_submission_df.columns.tolist())

    print(final_submission_df.head())

    submission_summary = {
        "rows": int(len(submission_df)),
        "candidate_ids_unique": bool(submission_df["candidate_id"].nunique() == len(submission_df)),
        "rank_min": int(submission_df["rank"].min()),
        "rank_max": int(submission_df["rank"].max()),
        "score_min": float(submission_df["score"].min()),
        "score_max": float(submission_df["score"].max()),
        "output_file": str(FINAL_SUBMISSION_CSV),
    }

    save_json(submission_summary, SUBMISSION_SUMMARY_JSON)

    print(submission_summary)
    print("Saved clean submission to:", FINAL_SUBMISSION_CSV)
    print("Saved:", SUBMISSION_SUMMARY_JSON)


    validator_path = VALIDATE_SUBMISSION_PY

    if validator_path.exists():
        print("Using validator:", validator_path)
        result = subprocess.run(
            [sys.executable, str(validator_path), str(FINAL_SUBMISSION_CSV)],
            capture_output=True,
            text=True
        )
        print("STDOUT:\n", result.stdout)
        print("STDERR:\n", result.stderr)
    else:
        print("Validator script not found:", validator_path)


    print(submission_df[["candidate_id", "rank", "score", "reasoning"]].head(20))

if __name__ == "__main__":
    main()