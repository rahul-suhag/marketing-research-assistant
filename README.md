# MKTG 323 · Marketing Research Assistant

**AI-Powered Study Companion for Data Analysis & Research Methods**

Professor Rahul Suhag · Mays Business School · Texas A&M University

Live app: [rahulsuhag-ai.streamlit.app](https://rahulsuhag-ai.streamlit.app)

---

## What It Does

A multi-agent Streamlit application that helps MKTG 323 students with:

- **Research Concepts** — Explains scale types, sampling methods, survey design, qualitative/quantitative research, and MRP analysis guidance using course materials via RAG (Retrieval-Augmented Generation).
- **Excel & ToolPak** — Step-by-step instructions for descriptive statistics, t-tests, ANOVA, chi-square, correlation, and regression using the Data Analysis ToolPak on both Mac and Windows.
- **Data Cleaning & Analysis** — Upload Qualtrics CSV exports for guided cleaning, variable recoding, scale-type classification, and transformation.

## Architecture

| Component | Technology |
|---|---|
| UI | Streamlit |
| Agent Orchestration | LangGraph (StateGraph) |
| LLM | Groq Cloud — Llama 3.3 70B Versatile |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 (CPU) |
| Vector Store | FAISS |
| Knowledge Base | 904 chunks from 23 course documents |

### Agent Workflow

```
User Question
     │
     ▼
  Router (keyword-based, no LLM call)
     │
     ├── RAG path ──► Retriever ──► Generator ──► Verifier ──► Response
     │
     └── Data path ──► Data Analyst (descriptive or code execution) ──► Response
```

- **Router** — Keyword-based routing saves LLM calls. Distinguishes between research concept questions (RAG) and data analysis tasks.
- **RAG Retriever** — Fetches top-k relevant chunks from FAISS index.
- **RAG Generator** — Produces grounded answers from retrieved context.
- **Verifier** — Rejects hallucinated answers not supported by course materials.
- **Data Analyst** — Dual-mode: descriptive (Excel guidance for Mac & Windows) or transform (pandas code execution in a sandboxed environment).

## Project Structure

```
├── app.py                  # Streamlit UI — chat, CSV upload, sidebar
├── agents.py               # LangGraph workflow — router, RAG, verifier, data analyst
├── ingest.py               # FAISS index builder — PDF, PPTX, text extraction
├── requirements.txt        # Python dependencies (unpinned for broad compatibility)
├── data/
│   └── slides/             # 23 source documents (PPTX, PDF, TXT)
├── faiss_index/            # Pre-built FAISS index (904 chunks)
└── .streamlit/
    ├── config.toml         # Texas A&M theme (maroon #500000)
    └── secrets.toml        # GROQ_API_KEY (gitignored)
```

## Knowledge Base

The RAG system indexes 23 documents covering the MKTG 323 curriculum:

- **13 session slide decks** (PPTX) — Sessions 1 through 12
- **6 course PDFs** — Syllabus, focus group handout, presentation rubrics, intro material
- **4 reference guides** (TXT) — Scale types & measurement, Excel ToolPak, Qualtrics data cleaning, MRP analysis

Documents are chunked (1,000 chars, 200 overlap) and embedded with `all-MiniLM-L6-v2` into a FAISS index.

## Setup

### Prerequisites

- Python 3.9+
- A Groq API key ([console.groq.com](https://console.groq.com))

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Add your Groq API key
mkdir -p .streamlit
echo 'GROQ_API_KEY = "your-key-here"' > .streamlit/secrets.toml

# Rebuild the FAISS index (optional — pre-built index included)
python ingest.py

# Run the app
streamlit run app.py
```

### Deployment (Streamlit Community Cloud)

1. Push to a public GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect the repo.
3. Add `GROQ_API_KEY` in the Streamlit Cloud secrets manager.
4. Deploy — the app uses the pre-built `faiss_index/` from the repo.

## Tech Notes

- **Rate limits**: Groq free tier allows 30 requests/min and 14,400 requests/day. The app includes exponential backoff retry logic.
- **Scale-type awareness**: The data analyst classifies uploaded CSV variables by measurement level (nominal, ordinal, interval, ratio) and recommends appropriate statistical tests.
- **Sandboxed execution**: Code transforms on uploaded data run in a restricted `exec()` with 42 whitelisted builtins.
- **PPTX extraction**: Recursive traversal of shapes, tables, and grouped objects for thorough slide text extraction.

## License

For academic use in MKTG 323 at Texas A&M University.
