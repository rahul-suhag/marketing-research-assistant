"""
app.py — Streamlit UI for the Marketing Research Multi-Agent Assistant.

MKTG 323 · Mays Business School · Texas A&M University
Professor Rahul Suhag
"""

import streamlit as st
import pandas as pd
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from agents import build_graph, AgentState, EMBEDDING_MODEL, RateLimitError

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="MKTG 323 · Research Assistant",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

FAISS_INDEX_DIR = Path("faiss_index")

# ── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Typography ─────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
[data-testid="stAppViewContainer"] { font-family: 'Inter', sans-serif; }

/* ── Feature card styling ───────────────────── */
.feature-card {
    background: white;
    border: 1px solid #e0d6d0;
    border-radius: 12px;
    padding: 1.5rem 1rem;
    text-align: center;
    transition: all 0.3s ease;
    box-shadow: 0 1px 4px rgba(80,0,0,0.06);
    height: 100%;
}
.feature-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(80,0,0,0.12);
    border-color: #500000;
}
.feature-icon { font-size: 2rem; margin-bottom: 0.6rem; }
.feature-title { font-size: 0.95rem; font-weight: 600; color: #333; margin-bottom: 0.3rem; }
.feature-desc { font-size: 0.8rem; color: #666; line-height: 1.5; }

/* ── Chat styling ───────────────────────────── */
[data-testid="stChatMessage"] { border-radius: 12px; }

/* ── Footer ─────────────────────────────────── */
.footer {
    text-align: center;
    padding: 1.5rem 0 0.5rem 0;
    font-size: 0.75rem;
    color: #999;
    border-top: 1px solid #eee;
    margin-top: 2rem;
}
.footer a { color: #500000; text-decoration: none; font-weight: 500; }

/* ── Sidebar ────────────────────────────────── */
[data-testid="stSidebar"] { background: #faf8f6; }

/* ── Hide default Streamlit elements ────────── */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── Session state defaults ───────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "csv_data" not in st.session_state:
    st.session_state.csv_data = None
if "csv_filename" not in st.session_state:
    st.session_state.csv_filename = None
if "processed_csv" not in st.session_state:
    st.session_state.processed_csv = None


# ── API key (from secrets only — never shown in UI) ─────────────────────────

api_key = st.secrets.get("GROQ_API_KEY", "")


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 📂 Data Upload")
    uploaded_csv = st.file_uploader(
        "Upload your Qualtrics CSV",
        type=["csv"],
        help="Upload your Qualtrics survey export (.csv) to clean, recode, or analyze it.",
    )

    if uploaded_csv is not None:
        st.session_state.csv_data = uploaded_csv.getvalue().decode("utf-8")
        st.session_state.csv_filename = uploaded_csv.name
        st.success(f"Loaded: {uploaded_csv.name}")
        try:
            preview_df = pd.read_csv(uploaded_csv)
            st.dataframe(preview_df.head(), use_container_width=True)
        except Exception:
            pass

    if st.session_state.csv_data and st.button("Remove CSV"):
        st.session_state.csv_data = None
        st.session_state.csv_filename = None
        st.session_state.processed_csv = None
        st.rerun()

    if st.session_state.processed_csv:
        st.divider()
        st.download_button(
            label="Download Processed CSV",
            data=st.session_state.processed_csv,
            file_name="processed_data.csv",
            mime="text/csv",
        )

    st.divider()
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("Powered by Llama 3.3 70B · FAISS · LangGraph")


# ── Load FAISS index ─────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading knowledge base …")
def load_vectorstore():
    if not FAISS_INDEX_DIR.exists():
        return None
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return FAISS.load_local(
        str(FAISS_INDEX_DIR), embeddings, allow_dangerous_deserialization=True
    )

vectorstore = load_vectorstore()


# ── Header ───────────────────────────────────────────────────────────────────

# Maroon banner using native Streamlit container
banner = st.container()
with banner:
    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, #500000 0%, #6e2020 50%, #8C2318 100%);
            padding: 2rem 2.5rem;
            border-radius: 16px;
            color: white;
            margin-bottom: 1.5rem;
        ">
            <div style="font-size: 1.8rem; font-weight: 700; letter-spacing: -0.5px; margin-bottom: 0.2rem;">
                🏛️ MKTG 323 · Marketing Research Assistant
            </div>
            <div style="font-size: 1rem; font-weight: 300; opacity: 0.9; margin-bottom: 0.4rem;">
                AI-Powered Study Companion for Data Analysis &amp; Research Methods
            </div>
            <div style="font-size: 0.82rem; opacity: 0.7;">
                Professor Rahul Suhag · Mays Business School · Texas A&amp;M University
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Feature cards (only on empty chat) ───────────────────────────────────────

if not st.session_state.messages:
    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:
        st.markdown(
            '<div class="feature-card">'
            '<div class="feature-icon">📖</div>'
            '<div class="feature-title">Research Concepts</div>'
            '<div class="feature-desc">Scale types, sampling methods, survey design, and MRP analysis guidance</div>'
            "</div>",
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            '<div class="feature-card">'
            '<div class="feature-icon">📊</div>'
            '<div class="feature-title">Excel &amp; ToolPak</div>'
            '<div class="feature-desc">Step-by-step instructions for descriptive stats, t-tests, ANOVA, chi-square &mdash; Mac &amp; Windows</div>'
            "</div>",
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            '<div class="feature-card">'
            '<div class="feature-icon">🧹</div>'
            '<div class="feature-title">Data Cleaning</div>'
            '<div class="feature-desc">Upload your Qualtrics CSV for guided cleaning, recoding, and analysis</div>'
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown("")  # spacer
    st.markdown(
        '<p style="text-align:center; color:#888; font-size:0.85rem;">'
        "Ask a question below or open the sidebar to upload a CSV."
        "</p>",
        unsafe_allow_html=True,
    )


# ── Guard: API key ───────────────────────────────────────────────────────────

if not api_key:
    st.error(
        "The assistant is not configured yet. "
        "Please contact Professor Suhag if you see this message."
    )
    st.stop()

if vectorstore is None:
    st.warning(
        "Knowledge base is loading or unavailable. "
        "You can still upload a CSV for data analysis."
    )

# ── Render chat history ─────────────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input ───────────────────────────────────────────────────────────────

if prompt := st.chat_input("Ask about research methods, scale types, Excel analysis, or describe a data task …"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking …"):
            graph = build_graph(api_key, vectorstore)

            initial_state: AgentState = {
                "question": prompt,
                "chat_history": st.session_state.messages[:-1],
                "route": "",
                "context_docs": [],
                "context_text": "",
                "rag_answer": "",
                "verified_answer": "",
                "csv_data": st.session_state.csv_data,
                "csv_filename": st.session_state.csv_filename,
                "final_answer": "",
            }

            try:
                result = graph.invoke(initial_state)
                answer = result.get("final_answer", "I wasn't able to generate a response.")

                if result.get("route") == "data_analysis" and result.get("csv_data"):
                    st.session_state.processed_csv = result["csv_data"]
                    st.session_state.csv_data = result["csv_data"]

            except RateLimitError as e:
                answer = (
                    f"**Rate limit reached.** {e}\n\n"
                    "Please wait a moment and try again."
                )
            except Exception as e:
                answer = f"An error occurred: {e}"

        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

# ── Footer ───────────────────────────────────────────────────────────────────

st.markdown(
    '<div class="footer">'
    "MKTG 323 · Marketing Research · "
    '<a href="https://mays.tamu.edu" target="_blank">Mays Business School</a>'
    " · Texas A&amp;M University<br>"
    "Built by Professor Rahul Suhag · Powered by open-source AI"
    "</div>",
    unsafe_allow_html=True,
)
