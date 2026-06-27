import pandas as pd

from sentence_transformers import CrossEncoder

from src.config import *
from src.utils import *
from src.load_data import *


def main():
        
    print("="*70)
    print("RERANKER AND TRAP FILTER")
    print("="*70)

    check_required_files()

    print("Retrieval shortlist exists:", RETRIEVAL_SHORTLIST_PARQUET.exists())
    print("JD compass exists:", JD_COMPASS_JSON.exists())

    features_df = load_parquet(FEATURES_PARQUET)
    retrieval_df = load_parquet(RETRIEVAL_SHORTLIST_PARQUET)

    jd_compass = load_json(JD_COMPASS_JSON)

    print("features_df shape:", features_df.shape)
    print("retrieval_df shape:", retrieval_df.shape)

    rerank_df = retrieval_df.merge(
        features_df,
        on="candidate_id",
        how="left",
        suffixes=("_retrieval", "_full")
    )

    print("rerank_df shape:", rerank_df.shape)
    print(rerank_df.head())

    print(retrieval_df.shape)
    print(rerank_df.shape)

    print([c for c in rerank_df.columns if "profile_text" in c or "skill_text" in c])

    # Cross encoder reranking setup
    jd_text = (
        jd_compass["jd_summary"]["role_title"] + ". " +
        jd_compass["jd_summary"]["role_goal"] + ". " +
        "Must have: " + ", ".join(jd_compass["jd_summary"]["must_have_skills"]) + ". " +
        "Preferred: " + ", ".join(jd_compass["jd_summary"]["strong_preferred_skills"])
    )
    # This builds one long job-description string having role title, role goal, must have skills and prefered skills


    rerank_df["rerank_candidate_text"] = (
        rerank_df["profile_text_full"].fillna("") + " " +
        rerank_df["skill_text_full"].fillna("")
    )
    # This creates one combined text per candidate from full profile text and full skill text
    # while fillna("") prevents NaN from breaking string concatenation

    print(jd_text[:1000])
    print("\nCandidate text example:\n")
    print(rerank_df["rerank_candidate_text"].iloc[0][:1000])

    cross_encoder = CrossEncoder( CROSS_ENCODER_MODEL_NAME)
    print("Cross-encoder loaded.")

    pairs = [
        (jd_text, text)
        for text in rerank_df["rerank_candidate_text"].tolist()
    ]

    # This creates a list of tuples and each tuple is (job_description_text, candidate_text)
    # This prepares the input for the crossencoder the JD never changes only the candidate profile changes
    print("Total pairs:", len(pairs))

    # Because a Cross Encoder always compares two texts together the input format is:
    # (Text A, Text B) or (Sentence A, Sentence B)
    rerank_scores = cross_encoder.predict(
        pairs,
        batch_size=32,
        show_progress_bar=True
    )

    rerank_df["cross_encoder_score"] = rerank_scores

    print(
        rerank_df[
            [
                "candidate_id",
                "current_title_retrieval",
                "years_of_experience_full",
                "cross_encoder_score"
            ]
        ].head(20)
    )

    print(rerank_df["cross_encoder_score"].describe())

    """Trap filter"""

    rerank_df["flag_extreme_experience_mismatch"] = (
        (rerank_df["years_of_experience_full"] > 12) &
        (rerank_df["seniority_fit_score_full"] == 0)
    )

    rerank_df["flag_weak_role_evidence"] = (
        rerank_df["role_evidence_score_full"] < 2.5
    )

    rerank_df["flag_high_risk_profile"] = (
        rerank_df["total_risk_score_full"] >= 4
    )

    rerank_df["flag_inactive_candidate"] = (
        (rerank_df["open_to_work_flag_full"] == 0) &
        (rerank_df["recruiter_response_rate_full"] < 0.25) &
        (rerank_df["avg_response_time_hours_full"] > 150)
    )

    rerank_df["flag_possible_framework_only_profile"] = (
        rerank_df["flag_possible_framework_only_profile_full"].fillna(False)
    )

    rerank_df["flag_possible_non_ir_ai_profile"] = (
        rerank_df["flag_possible_non_ir_ai_profile_full"].fillna(False)
    )

    rerank_df["trap_score"] = (
        rerank_df["flag_extreme_experience_mismatch"].astype(int) * 2.0 +
        rerank_df["flag_weak_role_evidence"].astype(int) * 1.5 +
        rerank_df["flag_high_risk_profile"].astype(int) * 2.0 +
        rerank_df["flag_inactive_candidate"].astype(int) * 1.5 +
        rerank_df["flag_possible_framework_only_profile"].astype(int) * 1.5 +
        rerank_df["flag_possible_non_ir_ai_profile"].astype(int) * 1.5
    )

    print(
        rerank_df[
            [
                "candidate_id",
                "trap_score",
                "flag_extreme_experience_mismatch",
                "flag_weak_role_evidence",
                "flag_high_risk_profile",
                "flag_inactive_candidate"
            ]
        ].head(20)
    )

    rerank_df["final_rerank_score"] = (
        rerank_df["cross_encoder_score"] * 3.0 +
        rerank_df["role_evidence_score_full"] * 1.5 +
        rerank_df["retrieval_stage_score"] * 1.0 +
        rerank_df["availability_strength_full"] * 0.02 +
        rerank_df["seniority_fit_score_full"] * 0.75 -
        rerank_df["trap_score"] * 2.5 -
        rerank_df["total_risk_score_full"] * 0.5
    )


    print(
        rerank_df[
            [
                "candidate_id",
                "current_title_retrieval",
                "cross_encoder_score",
                "role_evidence_score_full",
                "trap_score",
                "total_risk_score_full",
                "final_rerank_score"
            ]
        ].head(20)
    )

    rerank_df = rerank_df.sort_values(
        ["final_rerank_score", "cross_encoder_score", "candidate_id"],
        ascending=[False, False, True]
    ).reset_index(drop=True)

    print(
        rerank_df[
            [
                "candidate_id",
                "current_title_retrieval",
                "years_of_experience_full",
                "final_rerank_score",
                "cross_encoder_score",
                "role_evidence_score_full",
                "trap_score",
                "total_risk_score_full"
            ]
        ].head(30)
    )

    top_candidates = rerank_df.head(200).copy()

    print(
        top_candidates[
            [
                "candidate_id",
                "current_title_retrieval",
                "current_company_retrieval",
                "location_full",
                "years_of_experience_full",
                "final_rerank_score",
                "cross_encoder_score",
                "role_evidence_score_full",
                "trap_score",
                "total_risk_score_full"
            ]
        ].head(30)
    )

    final_shortlist_df = rerank_df.head(SUBMISSION_TOP_K).copy()
    print("Final shortlist size:", final_shortlist_df.shape[0])

    print(
        final_shortlist_df[
            [
                "candidate_id",
                "current_title_retrieval",
                "current_company_retrieval",
                "years_of_experience_full",
                "final_rerank_score",
                "cross_encoder_score",
                "trap_score"
            ]
        ]
    )

    save_parquet(  rerank_df,  RERANKED_PARQUET)
    save_csv( rerank_df, RERANKED_CSV)
    save_parquet( final_shortlist_df, FINAL_SHORTLIST_PARQUET)

    print("Saved:", RERANKED_PARQUET)             
    print("Saved:", RERANKED_CSV)
    print("Saved:", FINAL_SHORTLIST_PARQUET)

    rerank_summary = {
        "retrieval_pool_size": int(rerank_df.shape[0]),
        "final_shortlist_size": int(final_shortlist_df.shape[0]),
        "top_final_score": float(rerank_df["final_rerank_score"].max()),
        "embedding_reranker_model": CROSS_ENCODER_MODEL_NAME
    }

    save_json(rerank_summary,  RERANK_SUMMARY_JSON)
    print(rerank_summary)

if __name__ == "__main__":
    main()



