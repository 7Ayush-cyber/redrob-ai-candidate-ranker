import json
import numpy as np
import pandas as pd

from tqdm.auto import tqdm

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

import faiss

from src.config import *
from src.utils import *
from src.load_data import *



def main():
        
    print("="*70)
    print("HYBRID RETRIEVAL")
    print("="*70)

    check_required_files()

    print("Features:", FEATURES_PARQUET.exists())
    print("JD:", JOB_DESCRIPTION_CLEANED.exists())
    print("JD compass file exists:", JD_COMPASS_JSON.exists())

    features_df = load_parquet(FEATURES_PARQUET)
    # features_df.head()

    jd_text = JOB_DESCRIPTION_CLEANED.read_text(encoding="utf-8")
    # jd_text[:1000]

    jd_compass = load_json(JD_COMPASS_JSON)
    print("JD role:", jd_compass["jd_summary"]["role_title"])


    features_df["retrieval_text"] = (
        "Current title: " + features_df["current_title"].fillna("") + ". "
        + "Headline: " + features_df["headline"].fillna("") + ". "
        + "Summary: " + features_df["summary"].fillna("") + ". "
        + "Skills: " + features_df["skill_text"].fillna("") + ". "
        + "Career history: " + features_df["profile_text"].fillna("")
    )

    retrieval_texts = features_df["retrieval_text"].tolist()

    print("Total retrieval documents:", len(retrieval_texts))
    print("\nExample retrieval document:\n")
    print(retrieval_texts[0][:1500])

    retrieval_query = """
    Senior AI Engineer with production experience in machine learning, embeddings,
    semantic search, information retrieval, hybrid search, ranking, recommendation
    systems, vector databases, FAISS, Elasticsearch, OpenSearch, Pinecone, Qdrant,
    evaluation frameworks, NDCG, MRR, MAP, A/B testing, Python, deployment,
    MLOps, APIs, Docker, cloud infrastructure, LLMs, retrieval systems,
    search systems, matching systems, and real product engineering.
    """

    print(retrieval_query)

    # BM25 preprocessing
    def tokenize(text):
        return clean_text(text).lower().split()

    tokenized_corpus = []

    for text in tqdm(retrieval_texts):
        tokens = tokenize(text)
        tokenized_corpus.append(tokens)
    tokenized_query = tokenize(retrieval_query)

    print("Example tokenized candidate text:")
    print(tokenized_corpus[0][:40])

    bm25 = BM25Okapi(tokenized_corpus)

    bm25_scores = bm25.get_scores(tokenized_query)

    features_df["bm25_score"] = bm25_scores

    bm25_top_k = min(BM25_TOP_K, len(features_df))

    bm25_top_indices = np.argsort(bm25_scores)[::-1][:bm25_top_k]

    bm25_results = features_df.iloc[bm25_top_indices][
        [
            "candidate_id",
            "current_title",
            "current_company",
            "years_of_experience",
            "bm25_score",
            "role_evidence_score",
            "total_risk_score",
        ]
    ].copy()
    

  
    embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    candidate_embeddings = embedding_model.encode(
        retrieval_texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    candidate_embeddings = np.asarray(candidate_embeddings, dtype="float32")  # Because FAISS requires float32 objects only
    print("Candidate embedding shape:", candidate_embeddings.shape)

    np.save(
       CANDIDATE_EMBEDDINGS,
       candidate_embeddings
       )

    print("Saved:", CANDIDATE_EMBEDDINGS)

    embedding_dimension = candidate_embeddings.shape[1]

    faiss_index = faiss.IndexFlatIP(embedding_dimension)

    faiss_index.add(candidate_embeddings)

    print("Vectors in FAISS index:", faiss_index.ntotal)

    faiss.write_index( faiss_index, str(FAISS_INDEX))

    print("Saved:", FAISS_INDEX)

    query_embedding = embedding_model.encode(
        [retrieval_query],
        normalize_embeddings=True
    )

    query_embedding = np.asarray(query_embedding, dtype="float32")

    dense_top_k = min(DENSE_TOP_K, len(features_df))

    dense_scores, dense_indices = faiss_index.search(
        query_embedding,
        dense_top_k
    )

    dense_scores = dense_scores[0]
    dense_indices = dense_indices[0]

    dense_results = features_df.iloc[dense_indices][
        [
            "candidate_id",
            "current_title",
            "current_company",
            "years_of_experience",
            "role_evidence_score",
            "total_risk_score",
        ]
    ].copy()

    dense_results["dense_score"] = dense_scores

    bm25_ranked_ids = features_df.iloc[bm25_top_indices]["candidate_id"].tolist()
    dense_ranked_ids = features_df.iloc[dense_indices]["candidate_id"].tolist()

    bm25_ids = set(bm25_ranked_ids)
    dense_ids = set(dense_ranked_ids)

    overlap_ids = bm25_ids.intersection(dense_ids)

    print("BM25 top candidates:", len(bm25_ids))
    print("Dense top candidates:", len(dense_ids))
    print("Overlap:", len(overlap_ids))
    
    denom = max(1, min(len(bm25_ranked_ids), len(dense_ranked_ids)))
    print("Overlap percentage:", round(len(overlap_ids) / denom * 100, 2), "%")

    def reciprocal_rank_fusion(rankings, k=60):
        fused_scores = {}

        for ranking in rankings:
            for rank, candidate_id in enumerate(ranking, start=1):
                fused_scores[candidate_id] = fused_scores.get(candidate_id, 0) + 1 / (k + rank)

        return fused_scores

    rrf_scores = reciprocal_rank_fusion(
        [bm25_ranked_ids, dense_ranked_ids],
        k=60
    )

    rrf_df = pd.DataFrame(
        list(rrf_scores.items()),
        columns=["candidate_id", "rrf_score"]
    )

    retrieval_df = features_df.merge(
        rrf_df,
        on="candidate_id",
        how="inner"
    )

    retrieval_df = retrieval_df.sort_values(
        "rrf_score",
        ascending=False
    ).reset_index(drop=True)

    print("Candidates in fused retrieval pool:", retrieval_df.shape[0])

    print(
        retrieval_df[
            [
                "candidate_id",
                "current_title",
                "current_company",
                "years_of_experience",
                "bm25_score",
                "role_evidence_score",
                "total_risk_score",
                "rrf_score",
            ]
        ].head(30)
    )

    retrieval_df["retrieval_guardrail_score"] = (
        retrieval_df["role_evidence_score"] * 0.25
        + retrieval_df["availability_strength"] * 0.10
        - retrieval_df["total_risk_score"] * 0.15
    )

    retrieval_df["retrieval_stage_score"] = (
        retrieval_df["rrf_score"] * 100
        + retrieval_df["retrieval_guardrail_score"]
    )

    retrieval_df = retrieval_df.sort_values(
        "retrieval_stage_score",
        ascending=False
    ).reset_index(drop=True)

    print(
        retrieval_df[
            [
                "candidate_id",
                "current_title",
                "years_of_experience",
                "rrf_score",
                "role_evidence_score",
                "availability_strength",
                "total_risk_score",
                "retrieval_stage_score",
            ]
        ].head(30)
    )

    SHORTLIST_SIZE = min(FINAL_SHORTLIST_SIZE, len(retrieval_df))

    shortlist_df = retrieval_df.head(SHORTLIST_SIZE).copy()

    print("Final shortlist size:", shortlist_df.shape[0])

    print(
        shortlist_df[
            [
                "candidate_id",
                "current_title",
                "current_company",
                "years_of_experience",
                "retrieval_stage_score",
                "total_risk_score",
            ]
        ].head(20)
    )

    RETRIEVAL_SHORTLIST_PARQUET = ARTIFACTS_DIR / "candidate_retrieval_shortlist.parquet"
    RETRIEVAL_SHORTLIST_CSV = ARTIFACTS_DIR / "candidate_retrieval_shortlist.csv"
    RETRIEVAL_METADATA_JSON = ARTIFACTS_DIR / "retrieval_metadata.json"

    save_parquet( shortlist_df, RETRIEVAL_SHORTLIST_PARQUET)
    save_csv( shortlist_df, RETRIEVAL_SHORTLIST_CSV)

    retrieval_metadata = {
        "total_candidates": int(features_df.shape[0]),
        "bm25_top_k": int(bm25_top_k),
        "dense_top_k": int(dense_top_k),
        "fused_pool_size": int(retrieval_df.shape[0]),
        "final_shortlist_size": int(shortlist_df.shape[0]),
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "fusion_method": "Reciprocal Rank Fusion",
    }

    save_json(  retrieval_metadata, RETRIEVAL_METADATA_JSON )

    print(retrieval_metadata)

    print(
        shortlist_df[
            [
                "candidate_id",
                "current_title",
                "current_company",
                "years_of_experience",
                "retrieval_stage_score",
                "total_risk_score"
            ]
        ].head(20)
    )

if __name__ == "__main__":
    main()