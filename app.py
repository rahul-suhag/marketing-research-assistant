"""
app.py — Streamlit UI for the Marketing Research Multi-Agent Assistant.

Features:
  • Chat-based Q&A grounded in course slides & textbooks (RAG).
  • CSV upload for Qualtrics data cleaning and analysis.
  • Download button for processed data.
  • Full chat history maintained in session state.
"""

import streamlit as st
import pandas as pd
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from agents import build_graph, AgentState, EMBEDDING_MODEL, RateLimitError

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Marketing Research Assistant",
    page_icon="📊",
    layout="wide",
)

FAISS_INDEX_DIR = Path("faiss_index")

# ── Session state defaults ───────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "csv_data" not in st.session_state:
    st.session_state.csv_data = None
if "csv_filename" not in st.session_state:
    st.session_state.csv_filename = None
if "processed_csv" not in st.session_state:
    st.session_state.processed_csv = None


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Settings")
    # Pre-fill from secrets if available (for Streamlit Cloud deployment)
    default_key = st.secrets.get("GROQ_API_KEY", "") if hasattr(st, "secrets") else ""
    api_key = st.text_input(
        "Groq API Key",
        value=default_key,
        type="password",
        help="Get a free key at https://console.groq.com/keys",
    )

    st.divider()
    st.header("Upload Qualtrics CSV")
    uploaded_csv = st.file_uploader(
        "Upload a CSV file for data analysis",
        type=["csv"],
        help="Upload your Qualtrics survey export (.csv) to clean, recode, or analyze it.",
    )

    if uploaded_csv is not None:
        st.session_state.csv_data = uploaded_csv.getvalue().decode("utf-8")
        st.session_state.csv_filename = uploaded_csv.name
        st.success(f"Loaded: {uploaded_csv.name}")

        # Show a quick preview
        try:
            preview_df = pd.read_csv(uploaded_csv)
            st.dataframe(preview_df.head(), use_container_width=True)
        except Exception:
            pass

    if st.session_state.csv_data and st.button("Clear CSV"):
        st.session_state.csv_data = None
        st.session_state.csv_filename = None
        st.session_state.processed_csv = None
        st.rerun()

    # Download processed CSV
    if st.session_state.processed_csv:
        st.divider()
        st.download_button(
            label="Download Processed CSV",
            data=st.session_state.processed_csv,
            file_name="processed_data.csv",
            mime="text/csv",
        )

    st.divider()
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()


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

# ── Main UI ──────────────────────────────────────────────────────────────────

st.title("📊 Marketing Research Assistant")
st.caption(
    "Ask about marketing research concepts, get Excel how-to guidance, "
    "or upload a Qualtrics CSV for data cleaning and analysis."
)

if not api_key:
    st.info(
        "Enter your **Groq API key** in the sidebar to get started. "
        "Get a free key at [console.groq.com/keys](https://console.groq.com/keys)."
    )
    st.stop()

if vectorstore is None:
    st.warning(
        "No FAISS index found. Run `python ingest.py` first to build the knowledge base "
        "from your PDFs in `data/slides/` and `data/marketing_books/`.\n\n"
        "You can still use the **Data Analysis** features by uploading a CSV."
    )

# ── Render chat history ─────────────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input ───────────────────────────────────────────────────────────────

if prompt := st.chat_input("Ask a question or describe a data task …"):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build and invoke graph
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

                # If the data analyst produced new CSV data, save for download
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
