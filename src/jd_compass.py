from docx import Document
import json
import pandas as pd
from pprint import pprint

from src.config import *
from src.load_data import *
from src.utils import *

def main(): 

    print("=" * 70)
    print("JOB DESCRIPTION COMPASS")
    print("=" * 70)

    print("Project folder:", PROJECT_ROOT)
    print("Artifacts folder:", ARTIFACTS_DIR)

    check_required_files()

    print("JD path:", JOB_DESCRIPTION_DOCX)
    # We will be using the jd to under the roles required by the company and organise
    # our efforts to find out signals that matter the most in ranking

    document = Document(JOB_DESCRIPTION_DOCX)

    jd_paragraphs = []
    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if text:
            jd_paragraphs.append(text)

    jd_text = "\n".join(jd_paragraphs)

    print(jd_text[:500])

    # creating a clean copy of the jd so that it can be used later
    JOB_DESCRIPTION_CLEANED.write_text(
           jd_text,
           encoding="utf-8"
    )

    print("Saved:", JOB_DESCRIPTION_CLEANED)

    # We manually turn the JD into a structured summary.
    jd_compass = {
        "role_title": "Senior AI Engineer",

        "role_goal": (
            "Build and ship practical AI systems involving retrieval, ranking, "
            "LLMs, evaluation, and production engineering."
        ),

        "must_have_skills": [
            "python",
            "machine learning",
            "llm",
            "embeddings",
            "retrieval",
            "vector search",
            "ranking",
            "evaluation",
        ],

        "strong_preferred_skills": [
            "rag",
            "fine tuning",
            "transformers",
            "pytorch",
            "faiss",
            "elasticsearch",
            "docker",
            "aws",
            "gcp",
            "langchain",
            "langgraph",
        ],

        "experience_preferences": {
            "minimum_years": 3,
            "ideal_years_min": 4,
            "ideal_years_max": 10,
            "production_experience_required": True,
            "recent_production_code_preferred": True,
        },

        "behavioral_preferences": {
            "open_to_work_bonus": True,
            "recent_activity_bonus": True,
            "high_response_rate_bonus": True,
            "short_notice_period_bonus": True,
        },

        "negative_signals": [
            "only_recent_langchain_or_openai_experience",
            "no_recent_production_code",
            "keyword_stuffing",
            "career_timeline_inconsistency",
        ],
    }

    for key, value in jd_compass.items():
        print(f"\n{'=' * 80}")
        print(key.upper())
        print("=" * 80)
        print(value)

    skill_groups = {
        "core_ai_ml": [
            "machine learning",
            "deep learning",
            "llm",
            "large language model",
            "transformers",
            "pytorch",
            "tensorflow",
        ],

        "retrieval_ranking": [
            "embeddings",
            "retrieval",
            "vector search",
            "vector database",
            "faiss",
            "elasticsearch",
            "bm25",
            "ranking",
            "reranking",
            "semantic search",
            "rag",
        ],

        "production_engineering": [
            "python",
            "docker",
            "aws",
            "gcp",
            "azure",
            "api",
            "fastapi",
            "deployment",
            "production",
            "mlops",
        ],

        "evaluation_and_quality": [
            "evaluation",
            "benchmarking",
            "testing",
            "monitoring",
            "experimentation",
            "ab testing",
            "metrics",
        ],
    }

    for group_name, skills in skill_groups.items():
        print(f"\n{group_name.upper()}")
        print("=" * 80)
        pprint(skills)

    scoring_dimensions = {
        "semantic_role_fit": {
            "description": "How closely the candidate profile text matches the Senior AI Engineer role.",
            "weight": 0.30,
        },

        "must_have_skill_match": {
            "description": "Match on core AI, ML, retrieval, ranking, and evaluation skills.",
            "weight": 0.20,
        },

        "career_and_seniority_fit": {
            "description": "Whether experience level, title progression, and role history fit a senior AI engineering role.",
            "weight": 0.15,
        },

        "production_evidence": {
            "description": "Evidence of building and shipping real systems, APIs, deployment, cloud, or MLOps work.",
            "weight": 0.15,
        },

        "behavioral_readiness": {
            "description": "Open-to-work status, activity, recruiter responsiveness, notice period, and interview completion.",
            "weight": 0.10,
        },

        "trust_and_consistency": {
            "description": "Profile completeness, career consistency, trap-risk flags, and evidence quality.",
            "weight": 0.10,
        },
    }

    total_weight = sum(dim["weight"] for dim in scoring_dimensions.values())
    print("Total weight:", total_weight)

    assert round(total_weight, 5) == 1.0

    rubric_rows = []

    for dimension_name, details in scoring_dimensions.items():
        rubric_rows.append({
            "dimension": dimension_name,
            "weight": details["weight"],
            "description": details["description"],
        })

    rubric_df = pd.DataFrame(rubric_rows)
    print(rubric_df)

    jd_compass = {
        "jd_summary": jd_compass,
        "skill_groups": skill_groups,
        "scoring_dimensions": scoring_dimensions,
    }

    save_json(
        jd_compass,
        JD_COMPASS_JSON
    )

    print("Saved:", JD_COMPASS_JSON)

    files_to_check = [
        JOB_DESCRIPTION_CLEANED.name,
        JD_COMPASS_JSON.name,
    ]

    for file_name in files_to_check:
        file_path = ARTIFACTS_DIR / file_name
        print(f"{file_name:<35} {'SAVED' if file_path.exists() else 'MISSING'}")

    with open(
        JD_COMPASS_JSON,
        "r",
        encoding="utf-8"
    ) as f:

        loaded_jd_compass = json.load(f)

    print("Top-level sections:", list(loaded_jd_compass.keys()))
    print("Role:", loaded_jd_compass["jd_summary"]["role_title"])
    print("Saved file exists:", JD_COMPASS_JSON.exists())

if __name__ == "__main__":
    main()