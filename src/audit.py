from tqdm.auto import tqdm

from src.config import *
from src.load_data import *
from src.utils import *

import json
import matplotlib.pyplot as plt
import pandas as pd
from pprint import pprint
 
def main():
        
    print("Project folder ready:", PROJECT_ROOT)
    print("Data folder ready:", DATA_DIR)
    print("Notebooks folder ready:", NOTEBOOKS_DIR)
    print("Source code folder ready:", SRC_DIR)
    print("Outputs folder ready:", OUTPUTS_DIR)
    print("Artifacts folder ready:", ARTIFACTS_DIR)


    check_required_files()
    candidate_schema = load_candidate_schema()

    print("Schema title:", candidate_schema.get("title"))
    print("Required top-level fields:", candidate_schema.get("required"))
    print("\nTop-level properties:")
    print(list(candidate_schema["properties"].keys()))

    profile_fields = candidate_schema["properties"]["profile"]["properties"]

    print("Profile fields:\n")

    for field_name, field_info in profile_fields.items():
        print(f"- {field_name}")

    signal_fields = candidate_schema["properties"]["redrob_signals"]["properties"]

    print("Redrob signal fields:\n")

    for field_name in signal_fields.keys():
        print(f"- {field_name}")


    sample_candidates = load_candidate_sample(3)

    print("Sample candidates loaded:", len(sample_candidates))
    print("\nTop-level keys:")
    print(sample_candidates[0].keys())


    candidates = load_candidates()

    print("Loaded candidates:", len(candidates))

    rows = []

    for candidate in tqdm(candidates, desc="Creating audit table"):
        profile = candidate.get("profile", {})
        signals = candidate.get("redrob_signals", {})

        rows.append({
            "candidate_id": candidate.get("candidate_id"),

            "years_of_experience": profile.get("years_of_experience"),
            "current_title": profile.get("current_title"),
            "current_industry": profile.get("current_industry"),
            "country": profile.get("country"),
            "location": profile.get("location"),

            "num_career_roles": len(candidate.get("career_history", [])),
            "num_skills": len(candidate.get("skills", [])),
            "num_education": len(candidate.get("education", [])),
            "num_certifications": len(candidate.get("certifications", [])),

            "profile_completeness_score": signals.get("profile_completeness_score"),
            "last_active_date": signals.get("last_active_date"),
            "open_to_work_flag": signals.get("open_to_work_flag"),
            "recruiter_response_rate": signals.get("recruiter_response_rate"),
            "avg_response_time_hours": signals.get("avg_response_time_hours"),
            "notice_period_days": signals.get("notice_period_days"),
            "github_activity_score": signals.get("github_activity_score"),
            "interview_completion_rate": signals.get("interview_completion_rate"),
            "preferred_work_mode": signals.get("preferred_work_mode"),
            "willing_to_relocate": signals.get("willing_to_relocate"),
        })

    audit_df = pd.DataFrame(rows)

    print("Shape:", audit_df.shape)
    print("Unique candidate IDs:", audit_df["candidate_id"].nunique())
    print("Duplicate IDs:", audit_df["candidate_id"].duplicated().sum())

    # assert audit_df["candidate_id"].nunique() == 100000 
    unique_ids = audit_df["candidate_id"].nunique()
    duplicate_ids = audit_df["candidate_id"].duplicated().sum()

    print("Shape:", audit_df.shape)
    print("Unique candidate IDs:", unique_ids)
    print("Duplicate IDs:", duplicate_ids)

    assert duplicate_ids == 0, "Duplicate candidate IDs found." #

    print(audit_df[
        [
            "years_of_experience",
            "num_career_roles",
            "num_skills",
            "num_education",
            "num_certifications",
            "profile_completeness_score",
            "recruiter_response_rate",
            "avg_response_time_hours",
            "notice_period_days",
            "github_activity_score",
            "interview_completion_rate",
        ]
    ].describe())

    print("Top 20 current titles:\n")
    print(audit_df["current_title"].value_counts().head(20))

    print("Top 20 locations:\n")
    print(audit_df["location"].value_counts().head(20))

    print("Work mode distribution:\n")
    print(audit_df["preferred_work_mode"].value_counts(dropna=False))

    missing_report = (
        audit_df.isna()
        .mean()
        .mul(100)
        .sort_values(ascending=False)
        .reset_index()
    )

    missing_report.columns = ["field", "missing_percent"]

    print(missing_report)

    plt.figure(figsize=(10, 5))
    plt.hist(audit_df["years_of_experience"].dropna(), bins=30)
    plt.title("Years of Experience")
    plt.xlabel("Years")
    plt.ylabel("Number of Candidates")
    plt.savefig(
        OUTPUTS_DIR / "years_of_experience_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    plt.figure(figsize=(10, 5))
    plt.hist(audit_df["num_skills"].dropna(), bins=30)
    plt.title("Number of Skills per Candidate")
    plt.xlabel("Skills")
    plt.ylabel("Number of Candidates")
    plt.savefig(
        OUTPUTS_DIR / "skills_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    """ ## Let's Create flag to judge the profile of the candidates

    We create flags for three reasons:

    Trap detection
    Some candidate profiles may look good superficially but contain impossible or inconsistent patterns.
    Risk scoring
    Flags help us add penalties later in ranking.
    Interpretability
    When you later explain why a candidate was ranked up or down, flags give you a clear reason.

    """

    audit_df_flags = audit_df.copy()
    print("Initial shape:", audit_df_flags.shape)

    # Lets flag many skills
    audit_df_flags["flag_many_skills"] = audit_df_flags["num_skills"] >= 18

    print("After flag_many_skills shape:", audit_df_flags.shape)
    print(audit_df_flags[["candidate_id", "num_skills", "flag_many_skills"]].sample(10))

    print(audit_df_flags["flag_many_skills"].value_counts(dropna=False))

    # candidates with low experience and many roles
    audit_df_flags["flag_low_experience_many_roles"] = (
        (audit_df_flags["years_of_experience"] <= 3) &
        (audit_df_flags["num_career_roles"] >= 4)
    )

    print("After flag_low_experience_many_roles shape:", audit_df_flags.shape)
    print(audit_df_flags[["candidate_id", "years_of_experience", "num_career_roles", "flag_low_experience_many_roles"]].sample(10))
    print(audit_df_flags["flag_low_experience_many_roles"].value_counts(dropna=False))

    # candidate with ow profile completeness,
    # incomplete profile is less trustworthy and gives fewer useful signals for ranking.
    audit_df_flags["flag_bad_profile_completeness"] = audit_df_flags["profile_completeness_score"] < 35

    print("After flag_bad_profile_completeness shape:", audit_df_flags.shape)
    print(audit_df_flags[["candidate_id", "profile_completeness_score", "flag_bad_profile_completeness"]].head(10))
    print(audit_df_flags["flag_bad_profile_completeness"].value_counts(dropna=False))

    # low recruiter response rate
    audit_df_flags["flag_very_low_response_rate"] = audit_df_flags["recruiter_response_rate"] < 0.15

    print("After flag_very_low_response_rate shape:", audit_df_flags.shape)
    print(audit_df_flags[["candidate_id", "recruiter_response_rate", "flag_very_low_response_rate"]].head(10))
    print(audit_df_flags["flag_very_low_response_rate"].value_counts(dropna=False))

    # very slow response time
    # If someone takes too long to respond, the candidate may be less useful in a fast hiring cycle.
    audit_df_flags["flag_very_slow_response"] = audit_df_flags["avg_response_time_hours"] > 240

    print("After flag_very_slow_response shape:", audit_df_flags.shape)
    print(audit_df_flags[["candidate_id", "avg_response_time_hours", "flag_very_slow_response"]].head(10))
    print(audit_df_flags["flag_very_slow_response"].value_counts(dropna=False))

    # long notice period

    audit_df_flags["flag_long_notice_period"] = audit_df_flags["notice_period_days"] >= 120

    print("After flag_long_notice_period shape:", audit_df_flags.shape)
    print(audit_df_flags[["candidate_id", "notice_period_days", "flag_long_notice_period"]].head(10))
    print(audit_df_flags["flag_long_notice_period"].value_counts(dropna=False))

    # missing GitHub for technical profiles
    audit_df_flags["flag_no_github_for_technical_profile"] = (
        (audit_df_flags["github_activity_score"] == -1) &
        (audit_df_flags["current_title"].fillna("").str.contains(
            "engineer|developer|data|ml|ai|backend|software|full stack|frontend|machine learning",
            case=False,
            regex=True
        ))
    )

    print("After flag_no_github_for_technical_profile shape:", audit_df_flags.shape)
    print(audit_df_flags[["candidate_id", "current_title", "github_activity_score", "flag_no_github_for_technical_profile"]].head(15))
    print(audit_df_flags["flag_no_github_for_technical_profile"].value_counts(dropna=False))

    # low interview completion rate
    audit_df_flags["flag_low_interview_completion"] = audit_df_flags["interview_completion_rate"] < 0.5

    print("After flag_low_interview_completion shape:", audit_df_flags.shape)
    print(audit_df_flags[["candidate_id", "interview_completion_rate", "flag_low_interview_completion"]].head(10))
    print(audit_df_flags["flag_low_interview_completion"].value_counts(dropna=False))

    hard_flag_cols = [
        "flag_many_skills",
        "flag_low_experience_many_roles",
        "flag_bad_profile_completeness",
        "flag_very_low_response_rate",
        "flag_very_slow_response",
    ]

    soft_flag_cols = [
        "flag_long_notice_period",
        "flag_no_github_for_technical_profile",
        "flag_low_interview_completion",
    ]

    audit_df_flags["hard_risk_count"] = audit_df_flags[hard_flag_cols].sum(axis=1)
    audit_df_flags["soft_risk_count"] = audit_df_flags[soft_flag_cols].sum(axis=1)
    audit_df_flags["total_risk_score"] = audit_df_flags["hard_risk_count"] * 2 + audit_df_flags["soft_risk_count"]

    print("Final shape:", audit_df_flags.shape)
    print(audit_df_flags[[
        "candidate_id",
        "hard_risk_count",
        "soft_risk_count",
        "total_risk_score"
    ]].sample(10))

    print(audit_df_flags["total_risk_score"].describe())

    print(audit_df_flags["total_risk_score"].value_counts(dropna=False))

    print(audit_df.shape)
    print(audit_df_flags.shape)

    print("Original audit_df shape:", audit_df.shape)
    print("Flagged audit_df_flags shape:", audit_df_flags.shape)

    print("\nColumns added after flagging:")
    new_columns = [
        column
        for column in audit_df_flags.columns
        if column not in audit_df.columns
    ]

    pprint(new_columns)

    risk_view = audit_df_flags.sort_values(
        by=["total_risk_score", "hard_risk_count", "soft_risk_count"],
        ascending=[False, False, False]
    )

    print(
        risk_view[
            [
                "candidate_id",
                "current_title",
                "years_of_experience",
                "num_career_roles",
                "num_skills",
                "profile_completeness_score",
                "recruiter_response_rate",
                "avg_response_time_hours",
                "notice_period_days",
                "github_activity_score",
                "interview_completion_rate",
                "hard_risk_count",
                "soft_risk_count",
                "total_risk_score",
            ]
        ].head(20)
    )

    candidate_lookup = {
        candidate["candidate_id"]: candidate
        for candidate in candidates
    }

    highest_risk_candidate_id = risk_view.iloc[0]["candidate_id"]

    print("Highest-risk candidate ID:", highest_risk_candidate_id)

    pprint(candidate_lookup[highest_risk_candidate_id])

    print("Risk-score distribution:")
    print(
        audit_df_flags["total_risk_score"]
        .value_counts()
        .sort_index()
    )

    print("\nRisk-score summary:")
    print(
        audit_df_flags[
            ["hard_risk_count", "soft_risk_count", "total_risk_score"]
        ].describe()
    )


    save_parquet(
        audit_df_flags,
        AUDIT_PARQUET
    )

    save_csv(
        audit_df_flags,
        AUDIT_CSV
    )

    save_csv(
        missing_report,
        MISSING_VALUE_REPORT
    )

    notebook_1_summary = {
        "total_candidates": int(len(audit_df_flags)),
        "original_audit_shape": list(audit_df.shape),
        "flagged_audit_shape": list(audit_df_flags.shape),
        "number_of_added_columns": int(len(new_columns)),
        "flag_columns_added": new_columns,
        "max_total_risk_score": int(audit_df_flags["total_risk_score"].max()),
        "mean_total_risk_score": round(float(audit_df_flags["total_risk_score"].mean()), 4),
        "candidates_with_any_risk": int(
            (audit_df_flags["total_risk_score"] > 0).sum()
        ),
    }

    save_json(
        notebook_1_summary,
        AUDIT_SUMMARY_JSON
    )

    print("Saved successfully:")
    print("- candidate_audit_with_flags.parquet")
    print("- candidate_audit_with_flags.csv")
    print("- missing_value_report.csv")
    print("- notebook_1_summary.json")

    saved_files = [
        AUDIT_PARQUET.name,
        AUDIT_CSV.name,
        MISSING_VALUE_REPORT.name,
        AUDIT_SUMMARY_JSON.name,
    ]

    for file_name in saved_files:
        file_path = ARTIFACTS_DIR / file_name
        print(f"{file_name:<45} {'SAVED' if file_path.exists() else 'MISSING'}")


if __name__ == "__main__":
    main()