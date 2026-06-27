import json

import pandas as pd
from pprint import pprint
from tqdm.auto import tqdm

from src.config import *
from src.load_data import load_candidates
from src.utils import *


def main():
        
    print("Candidates exists:", CANDIDATES_JSONL.exists())
    print("Flags exists:", AUDIT_PARQUET.exists())
    print("JD compass exists:", JD_COMPASS_JSON.exists())

    audit_df_flags = load_parquet(AUDIT_PARQUET)
    jd_compass = load_json(JD_COMPASS_JSON)

    print("audit_df_flags shape:", audit_df_flags.shape)
    print("JD compass keys:", jd_compass.keys())

    # Loading all the raw candidate data
    candidates = load_candidates()
    print("Total candidates loaded:", len(candidates))
    assert len(candidates) == audit_df_flags.shape[0], "Number of candidates and flags do not match"

    for key , value in candidates[0].items():
     pprint(f"{key}: {value}")


    rows = []

    for candidate in tqdm(candidates, desc="Building feature rows"):
        profile = candidate.get("profile", {}) # If missing → return empty dictionary
        career_history = candidate.get("career_history", [])
        education = candidate.get("education", [])
        skills = candidate.get("skills", [])
        certifications = candidate.get("certifications", [])
        languages = candidate.get("languages", [])
        signals = candidate.get("redrob_signals", {})

        rows.append({
            "candidate_id": candidate.get("candidate_id"),
            "anonymized_name": clean_text(profile.get("anonymized_name")),
            "headline": clean_text(profile.get("headline")),
            "summary": clean_text(profile.get("summary")),
            "location": clean_text(profile.get("location")),
            "country": clean_text(profile.get("country")),
            "years_of_experience": safe_float(profile.get("years_of_experience")),
            "current_title": clean_text(profile.get("current_title")),
            "current_company": clean_text(profile.get("current_company")),
            "current_company_size": clean_text(profile.get("current_company_size")),
            "current_industry": clean_text(profile.get("current_industry")),

            "num_career_roles": len(career_history),
            "num_education": len(education),
            "num_skills": len(skills),
            "num_certifications": len(certifications),
            "num_languages": len(languages),

            "profile_completeness_score": safe_float(signals.get("profile_completeness_score")),
            "open_to_work_flag": int(bool(signals.get("open_to_work_flag"))),
            "recruiter_response_rate": safe_float(signals.get("recruiter_response_rate")),
            "avg_response_time_hours": safe_float(signals.get("avg_response_time_hours")),
            "notice_period_days": safe_float(signals.get("notice_period_days")),
            "github_activity_score": safe_float(signals.get("github_activity_score"), default=-1),
            "interview_completion_rate": safe_float(signals.get("interview_completion_rate")),
            "search_appearance_30d": safe_float(signals.get("search_appearance_30d")),
            "saved_by_recruiters_30d": safe_float(signals.get("saved_by_recruiters_30d")),
            "verified_email": int(bool(signals.get("verified_email"))),
            "verified_phone": int(bool(signals.get("verified_phone"))),
            "linkedin_connected": int(bool(signals.get("linkedin_connected"))),
        })

    features_df = pd.DataFrame(rows)
    print("Feature dataframe shape:", features_df.shape)
    print(features_df.head())

    #  career history analysis
    def extract_skill_names(skill_list):
        if not isinstance(skill_list, list):
            return []
        return [clean_text(skill.get("name")) for skill in skill_list if clean_text(skill.get("name"))]

    skill_texts = []

    for candidate in candidates:
        skills = candidate.get("skills", [])
        skill_names = extract_skill_names(skills)
        skill_texts.append(" | ".join(skill_names))

    features_df["skill_text"] = skill_texts
    print(features_df[["candidate_id", "skill_text"]].head())

    profile_texts = []

    for candidate in candidates:
        profile = candidate.get("profile", {})
        career_history = candidate.get("career_history", [])

        career_parts = []
        for role in career_history:
            title = clean_text(role.get("title"))
            company = clean_text(role.get("company"))
            industry = clean_text(role.get("industry"))
            desc = clean_text(role.get("description"))
            career_parts.append(f"{title} at {company} in {industry}. {desc}")

        profile_text = " ".join([
            clean_text(profile.get("headline")),
            clean_text(profile.get("summary")),
            clean_text(profile.get("current_title")),
            clean_text(profile.get("current_company")),
            clean_text(profile.get("current_industry")),
            " ".join(career_parts)
        ])

        profile_texts.append(profile_text)

    features_df["profile_text"] = profile_texts
    print(features_df[["candidate_id", "profile_text"]].head())

    technical_keywords = [
        "python", "ml", "machine learning", "llm", "rag", "retrieval", "ranking",
        "vector", "faiss", "elasticsearch", "pytorch", "tensorflow", "langchain",
        "langgraph", "embeddings", "search", "nlp", "recommendation"
    ]

    def count_technical_keywords(text):
        text = clean_text(text).lower()
        count = 0
        for keyword in technical_keywords:
            if keyword in text:
                count += 1
        return count

    features_df["technical_keyword_count"] = features_df["profile_text"].apply(count_technical_keywords)
    print(features_df[["candidate_id", "technical_keyword_count"]].head())

    flags_subset = audit_df_flags[
        [
            "candidate_id",
            "hard_risk_count",
            "soft_risk_count",
            "total_risk_score",
            "flag_many_skills",
            "flag_low_experience_many_roles",
            "flag_bad_profile_completeness",
            "flag_very_low_response_rate",
            "flag_very_slow_response",
            "flag_long_notice_period",
            "flag_no_github_for_technical_profile",
            "flag_low_interview_completion",
        ]
    ]

    features_df = features_df.merge(flags_subset, on="candidate_id", how="left")
    print("After merge shape:", features_df.shape)
    print(features_df.head())

    numeric_cols = features_df.select_dtypes(include=["int64", "float64"]).columns

    for col in numeric_cols:
        features_df[col] = features_df[col].fillna(0)

    print("Missing values after fill:")
    print(features_df.isna().sum().sort_values(ascending=False).head(20))

    features_df["signal_strength"] = (
        features_df["profile_completeness_score"] * 0.3 +
        features_df["recruiter_response_rate"] * 100 * 0.2 +
        features_df["interview_completion_rate"] * 100 * 0.2 +
        features_df["saved_by_recruiters_30d"] * 0.1 +
        features_df["technical_keyword_count"] * 5
    )

    features_df["availability_strength"] = (
        features_df["open_to_work_flag"] * 20 +
        (1 / (1 + features_df["notice_period_days"])) * 50 +
        features_df["recruiter_response_rate"] * 30
    )

    print(features_df[[
        "candidate_id",
        "signal_strength",
        "availability_strength"
    ]].head())

    summary_cols = [
        "candidate_id",
        "years_of_experience",
        "num_career_roles",
        "num_skills",
        "profile_completeness_score",
        "recruiter_response_rate",
        "avg_response_time_hours",
        "notice_period_days",
        "github_activity_score",
        "interview_completion_rate",
        "technical_keyword_count",
        "hard_risk_count",
        "soft_risk_count",
        "total_risk_score",
        "signal_strength",
        "availability_strength",
    ]

    print(features_df[summary_cols].head(20))

    """The previous cells created general candidate features and now we will create role-specific features based on the Senior AI Engineer JD.
    - retrieval and ranking evidence
    - production engineering evidence
    - evaluation evidence
    - seniority fit
    - location fit
    - possible disqualifier signals
    """

    skill_groups = jd_compass["skill_groups"]
    jd_summary = jd_compass["jd_summary"]

    print("JD role:", jd_summary["role_title"])
    print("Skill groups:", list(skill_groups.keys()))

    def count_group_matches(text, keywords):
        text = clean_text(text).lower()

        matched_keywords = []

        for keyword in keywords:
            if keyword.lower() in text:
                matched_keywords.append(keyword)

        return len(set(matched_keywords))

    features_df["core_ai_ml_match_count"] = features_df["profile_text"].apply(
        lambda text: count_group_matches(text, skill_groups["core_ai_ml"])
    )

    features_df["retrieval_ranking_match_count"] = features_df["profile_text"].apply(
        lambda text: count_group_matches(text, skill_groups["retrieval_ranking"])
    )

    features_df["production_engineering_match_count"] = features_df["profile_text"].apply(
        lambda text: count_group_matches(text, skill_groups["production_engineering"])
    )

    features_df["evaluation_quality_match_count"] = features_df["profile_text"].apply(
        lambda text: count_group_matches(text, skill_groups["evaluation_and_quality"])
    )

    print(
        features_df[
            [
                "candidate_id",
                "core_ai_ml_match_count",
                "retrieval_ranking_match_count",
                "production_engineering_match_count",
                "evaluation_quality_match_count",
            ]
        ].head(10)
    )

    features_df["seniority_fit_score"] = 0.0

    features_df.loc[
        features_df["years_of_experience"].between(5, 9, inclusive="both"),
        "seniority_fit_score"
    ] = 1.0

    features_df.loc[
        features_df["years_of_experience"].between(4, 10, inclusive="both") &
        (features_df["seniority_fit_score"] == 0),
        "seniority_fit_score"
    ] = 0.6

    features_df.loc[
        features_df["years_of_experience"].between(3, 12, inclusive="both") &
        (features_df["seniority_fit_score"] == 0),
        "seniority_fit_score"
    ] = 0.3

    print(
        features_df[
            ["candidate_id", "years_of_experience", "seniority_fit_score"]
        ].head(10)
    )

    preferred_locations = [
        "pune",
        "noida",
        "delhi",
        "gurgaon",
        "gurugram",
        "mumbai",
        "hyderabad",
        "bangalore",
        "bengaluru",
    ]

    features_df["location_fit_flag"] = features_df["location"].apply(
        lambda location: int(
            any(city in clean_text(location).lower() for city in preferred_locations)
        )
    )

    print(
        features_df[
            ["candidate_id", "location", "location_fit_flag"]
        ].head(10)
    )

    features_df["flag_possible_framework_only_profile"] = (
        features_df["profile_text"].str.lower().str.contains(
            "langchain|openai",
            regex=True,
            na=False
        )
        &
        (features_df["retrieval_ranking_match_count"] <= 1)
        &
        (features_df["production_engineering_match_count"] <= 1)
    )

    features_df["flag_possible_non_ir_ai_profile"] = (
        features_df["profile_text"].str.lower().str.contains(
            "computer vision|cv engineer|speech|robotics",
            regex=True,
            na=False
        )
        &
        (features_df["retrieval_ranking_match_count"] == 0)
    )

    print(
        features_df[
            [
                "candidate_id",
                "flag_possible_framework_only_profile",
                "flag_possible_non_ir_ai_profile",
            ]
        ].head(20)
    )

    features_df["role_evidence_score"] = (
        features_df["core_ai_ml_match_count"] * 1.0
        + features_df["retrieval_ranking_match_count"] * 2.0
        + features_df["production_engineering_match_count"] * 1.5
        + features_df["evaluation_quality_match_count"] * 2.0
        + features_df["seniority_fit_score"] * 3.0
        + features_df["location_fit_flag"] * 0.5
        - features_df["flag_possible_framework_only_profile"].astype(int) * 2.0
        - features_df["flag_possible_non_ir_ai_profile"].astype(int) * 2.0
    )

    print(
        features_df[
            [
                "candidate_id",
                "role_evidence_score",
                "retrieval_ranking_match_count",
                "production_engineering_match_count",
                "evaluation_quality_match_count",
                "seniority_fit_score",
                "total_risk_score",
            ]
        ]
        .sort_values("role_evidence_score", ascending=False)
        .head(20)
    )

    top_role_evidence = features_df.sort_values(
        "role_evidence_score",
        ascending=False
    ).head(10)

    print(
        top_role_evidence[
            [
                "candidate_id",
                "current_title",
                "current_company",
                "location",
                "years_of_experience",
                "role_evidence_score",
                "retrieval_ranking_match_count",
                "production_engineering_match_count",
                "evaluation_quality_match_count",
                "total_risk_score",
            ]
        ]
    )

    summary = {
        "rows": int(features_df.shape[0]),
        "columns": int(features_df.shape[1]),
        "avg_years_of_experience": float(features_df["years_of_experience"].mean()),
        "avg_profile_completeness": float(features_df["profile_completeness_score"].mean()),
        "avg_total_risk_score": float(features_df["total_risk_score"].mean()),
        "num_candidates_open_to_work": int(features_df["open_to_work_flag"].sum()),
    }

    print(summary)

    save_parquet( features_df, FEATURES_PARQUET)
    save_csv( features_df, FEATURES_CSV)
    save_json( summary, FEATURES_SUMMARY_JSON)

    print("Updated candidate feature files saved.")
    print("Final feature shape:", features_df.shape)

if __name__ == "__main__":
    main()