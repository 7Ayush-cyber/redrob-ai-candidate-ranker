import shutil
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import *
from src.utils import *
from src import audit
from src import candidate_features
from src import retrieval
from src import reranker
from src import reasoning

# -------------------------------------------------------
# Streamlit Configuration
# -------------------------------------------------------
Path("data").mkdir(parents=True, exist_ok=True)
Path("artifacts").mkdir(parents=True, exist_ok=True)
Path("outputs").mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="Redrob AI Candidate Ranker",
    page_icon="🤖",
    layout="wide",
)

# -------------------------------------------------------
# Session State
# -------------------------------------------------------
if "pipeline_completed" not in st.session_state:
    st.session_state.pipeline_completed = False

if "uploaded_file_path" not in st.session_state:
    st.session_state.uploaded_file_path = None

if "submission_df" not in st.session_state:
    st.session_state.submission_df = None

if "last_uploaded_filename" not in st.session_state:
    st.session_state.last_uploaded_filename = None 
# -------------------------------------------------------
# Header
# -------------------------------------------------------

st.title("Redrob AI Candidate Ranking System")

st.markdown(
"""
This demo showcases our complete AI hiring pipeline.

Pipeline:

**Audit**
→ **Candidate Feature Engineering**
→ **Hybrid Retrieval**
→ **Cross Encoder Reranking**
→ **Reasoning Generation**
→ **Submission CSV**
"""
)

st.divider()

# -------------------------------------------------------
# Sidebar
# -------------------------------------------------------

st.sidebar.title("Pipeline")

st.sidebar.success("Audit")
st.sidebar.success("Candidate Features")
st.sidebar.success("Hybrid Retrieval")
st.sidebar.success("Cross Encoder")
st.sidebar.success("Reasoning")

st.sidebar.divider()

st.sidebar.write("Embedding Model")
st.sidebar.code(EMBEDDING_MODEL_NAME)
st.sidebar.write("Cross Encoder")
st.sidebar.code(CROSS_ENCODER_MODEL_NAME)

st.sidebar.divider()


# -------------------------------------------------------
# Upload Section
# -------------------------------------------------------
st.header("Upload Candidate Dataset")

st.info(
    """
Demo mode:
- Upload any `candidates.jsonl` file for the fastest experience.
- Smaller datasets are recommended for quick interactive testing.
- Larger datasets are supported, but first-time processing may take longer because embeddings are recomputed.
"""
)

uploaded_file = st.file_uploader(
    "Upload candidates.jsonl (recommended for demo)",
    type=["jsonl"],
)

# -------------------------------------------------------
# Save Uploaded File
# -------------------------------------------------------
if uploaded_file is not None:

    if uploaded_file.name != st.session_state.last_uploaded_filename:
        temp_dir = Path(tempfile.mkdtemp())
        uploaded_path = temp_dir / uploaded_file.name

        with open(uploaded_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.session_state.uploaded_file_path = uploaded_path
        st.session_state.last_uploaded_filename = uploaded_file.name

        st.session_state.pipeline_completed = False
        st.session_state.submission_df = None

        if FINAL_SUBMISSION_CSV.exists():
            FINAL_SUBMISSION_CSV.unlink()

        st.success("Candidate file uploaded successfully.")
    

# -------------------------------------------------------
# Run Pipeline Button
# -------------------------------------------------------

st.divider()

run_pipeline = st.button(
    "🚀 Run AI Ranking Pipeline",
    use_container_width=True,
)

# -------------------------------------------------------
# Progress Placeholder
# -------------------------------------------------------

progress_bar = st.progress(0)
status_box = st.empty()

# -------------------------------------------------------
# Placeholder
# -------------------------------------------------------
if run_pipeline:

    if st.session_state.uploaded_file_path is None:
        st.error("Please upload a dataset first.")
    else:
        with st.spinner("Running AI Ranking Pipeline..."):

            uploaded_path = Path(st.session_state.uploaded_file_path)

            if uploaded_path != CANDIDATES_JSONL:
                shutil.copy2(uploaded_path, CANDIDATES_JSONL)

            status_box.info("Candidate dataset copied.")
            progress_bar.progress(10)

            status_box.info("Running Audit...")
            audit.main()
            progress_bar.progress(20)

            status_box.info("Generating Candidate Features...")
            candidate_features.main()
            progress_bar.progress(40)

            status_box.info("Running Hybrid Retrieval...")
            retrieval.main()
            progress_bar.progress(70)

            status_box.info("Running Cross Encoder...")
            reranker.main()
            progress_bar.progress(90)

            status_box.info("Generating Final Submission...")
            reasoning.main(strict_submission=False)
            progress_bar.progress(100)

            st.balloons()

        status_box.success("Pipeline finished successfully. Submission generated.")
        st.session_state.pipeline_completed = True


if st.session_state.pipeline_completed and FINAL_SUBMISSION_CSV.exists():

    with open(FINAL_SUBMISSION_CSV, "rb") as f:
        st.download_button(
            label="📥 Download Submission CSV",
            data=f,
            file_name=FINAL_SUBMISSION_CSV.name,
            mime="text/csv",
            use_container_width=True,
        )

    submission_df = pd.read_csv(FINAL_SUBMISSION_CSV)
    st.success(f"Generated {len(submission_df)} ranked candidates.")
    st.session_state.submission_df = submission_df


if st.session_state.pipeline_completed and st.session_state.submission_df is not None:

    st.divider()
    st.header("🏆 Final Leaderboard")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Candidates Ranked", len(st.session_state.submission_df))

    with col2:
        st.metric("Top Rank", 1)

    st.dataframe(
        st.session_state.submission_df,
        use_container_width=True,
    )
if (
    st.session_state.pipeline_completed
    and st.session_state.submission_df is not None
):
    
    st.divider()

    st.header("🔍 Search Candidate")

    candidate_id = st.text_input("Enter Candidate ID")

    if candidate_id and st.session_state.submission_df is not None:

        candidate_id = candidate_id.strip().upper()

        result = st.session_state.submission_df[
            st.session_state.submission_df["candidate_id"].str.upper() == candidate_id
        ]

        if not result.empty:
            st.success("Candidate Found")
            st.dataframe(result, use_container_width=True)
        else:
            st.warning("Candidate not found.")




if (
    st.session_state.pipeline_completed
    and st.session_state.submission_df is not None
):
    st.divider()

    st.header("🥇 Top 10 Candidates")

    st.dataframe(
        st.session_state.submission_df.head(10),
        use_container_width=True
    )




if (
    st.session_state.pipeline_completed
    and st.session_state.submission_df is not None
):
    st.divider()
    st.header("📈 Score Distribution")



    chart = (
        st.session_state.submission_df
        .set_index("rank")["score"]
    )

    st.line_chart(chart)    


st.divider()

st.header("🧠 Models Used")

st.info(
    f"""
Embedding Model

{EMBEDDING_MODEL_NAME}

---

Cross Encoder

{CROSS_ENCODER_MODEL_NAME}
"""
)

st.divider()
st.header("🏗 AI Ranking Pipeline")
st.markdown("""
1. Audit uploaded candidate dataset
2. Engineer candidate features
3. BM25 Retrieval
4. Dense Semantic Retrieval (BGE)
5. Reciprocal Rank Fusion
6. Cross Encoder Reranking
7. Trap Filtering
8. Reasoning Generation
9. Final Submission CSV
""")

st.divider()
st.header("📊 Pipeline Statistics")
if (
    st.session_state.pipeline_completed
    and st.session_state.submission_df is not None
):

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Candidates Ranked",
            len(st.session_state.submission_df)
        )

    with col2:
        st.metric(
            "Top Score",
            round(
                st.session_state.submission_df["score"].max(),
                4
            )
        )

    with col3:
        st.metric(
            "Lowest Score",
            round(
                st.session_state.submission_df["score"].min(),
                4
            )
        )

    with col4:
        st.metric(
            "Unique IDs",
            st.session_state.submission_df["candidate_id"].nunique()
        )


st.divider()

st.header("ℹ️ About This Project")

st.markdown("""
### Redrob AI Candidate Ranking Challenge

This system performs end-to-end candidate ranking using a hybrid retrieval pipeline.

Pipeline Overview

- Candidate Audit
- Feature Engineering
- BM25 Lexical Retrieval
- Dense Semantic Retrieval (BGE Embeddings)
- Reciprocal Rank Fusion (RRF)
- Cross Encoder Reranking
- Trap Filtering
- Reasoning Generation
- Final Submission CSV

Models

- BAAI/bge-small-en-v1.5
- cross-encoder/ms-marco-MiniLM-L-6-v2

Outputs

- Ranked Candidates
- Human-readable Reasoning
- Submission CSV
""")


st.divider()

st.caption(
    "Built for the Redrob AI Candidate Ranking Challenge | IIT Guwahati"
)
