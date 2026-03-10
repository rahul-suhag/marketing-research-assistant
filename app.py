"""
app.py — Streamlit UI for the MKTG 323 Marketing Research Contest Assistant.

4-tab layout:
  Tab 1: Ask About Class Content (RAG)
  Tab 2: Build My Survey
  Tab 3: Clean & Code My CSV
  Tab 4: Match MRP to Analysis

MKTG 323 · Mays Business School · Texas A&M University
Professor Rahul Suhag
"""

import streamlit as st
import pandas as pd
import io
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from agents import (
    build_graph,
    AgentState,
    EMBEDDING_MODEL,
    RateLimitError,
    invoke_chain,
    retrieve_context,
    build_csv_profile,
    data_analyst_node,
    get_llm,
    invoke_with_retry,
    SURVEY_BUILDER_PROMPT,
    SURVEY_REVIEW_PROMPT,
    PRETEST_PROMPT,
    SYNTHETIC_DATA_PROMPT,
    MRP_ANALYSIS_PROMPT,
    CODEBOOK_PROMPT,
    DESCRIBE_PROMPT,
    CODE_PROMPT,
    SAFE_BUILTINS,
    _build_preview,
    _is_transform_request,
)

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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
[data-testid="stAppViewContainer"] { font-family: 'Inter', sans-serif; }
[data-testid="stChatMessage"] { border-radius: 12px; }
[data-testid="stSidebar"] { background: #faf8f6; }
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.profile-box {
    background: #faf8f6;
    border: 1px solid #e0d6d0;
    border-radius: 10px;
    padding: 1rem;
    font-size: 0.82rem;
    max-height: 400px;
    overflow-y: auto;
}
</style>
""", unsafe_allow_html=True)


# ── Session state defaults ───────────────────────────────────────────────────

_defaults = {
    # Tab 1
    "tab1_messages": [],
    # Tab 2
    "tab2_messages": [],
    "survey_mrps": "",
    "survey_company": "",
    "survey_population": "",
    "survey_personas": "",
    "survey_draft": "",
    "pretest_results": "",
    "synthetic_data": "",
    # Tab 3
    "tab3_messages": [],
    "csv_data": None,
    "csv_filename": None,
    "csv_profile": "",
    "processed_csv": None,
    "codebook": "",
    # Tab 4
    "tab4_messages": [],
    "analysis_mrps": "",
}
for key, default in _defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ── API key ──────────────────────────────────────────────────────────────────

api_key = st.secrets.get("GROQ_API_KEY", "")


# ── Sidebar (minimal) ───────────────────────────────────────────────────────

with st.sidebar:
    if st.session_state.csv_filename:
        st.markdown(f"**CSV loaded:** {st.session_state.csv_filename}")
    st.divider()
    if st.button("Clear All Tabs"):
        for key, default in _defaults.items():
            st.session_state[key] = default
        st.rerun()
    st.caption("MKTG 323 · Llama 3.3 70B · FAISS")


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


# ── Header (compact) ────────────────────────────────────────────────────────

st.markdown(
    """
    <div style="
        background: linear-gradient(135deg, #500000 0%, #6e2020 50%, #8C2318 100%);
        padding: 1.2rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1rem;
    ">
        <div style="font-size: 1.5rem; font-weight: 700; letter-spacing: -0.5px;">
            🏛️ MKTG 323 · Marketing Research Contest Assistant
        </div>
        <div style="font-size: 0.82rem; opacity: 0.75; margin-top: 0.2rem;">
            Professor Rahul Suhag · Mays Business School · Texas A&amp;M University
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ── Guard: API key ──────────────────────────────────────────────────────────

if not api_key:
    st.error(
        "The assistant is not configured yet. "
        "Please contact Professor Suhag if you see this message."
    )
    st.stop()

if vectorstore is None:
    st.warning("Knowledge base is loading or unavailable.")


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs([
    "📖 Ask About Class Content",
    "📝 Build My Survey",
    "🧹 Clean & Code My CSV",
    "📊 Match MRP to Analysis",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ASK ABOUT CLASS CONTENT (RAG)
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.caption("Ask about scale types, sampling, survey design, Excel ToolPak, data cleaning, or any MKTG 323 topic.")

    # Render chat history
    for msg in st.session_state.tab1_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about research methods, scale types, Excel analysis …", key="tab1_chat"):
        st.session_state.tab1_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking …"):
                graph = build_graph(api_key, vectorstore)

                initial_state: AgentState = {
                    "question": prompt,
                    "chat_history": st.session_state.tab1_messages[:-1],
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
                except RateLimitError as e:
                    answer = f"**Rate limit reached.** {e}\n\nPlease wait a moment and try again."
                except Exception as e:
                    answer = f"An error occurred: {e}"

            st.markdown(answer)

        st.session_state.tab1_messages.append({"role": "assistant", "content": answer})


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — BUILD MY SURVEY
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.caption("Design a Qualtrics-ready survey grounded in your MRPs and class best practices.")

    left_col, right_col = st.columns([1, 2], gap="large")

    with left_col:
        st.markdown("#### Survey Inputs")
        st.session_state.survey_mrps = st.text_area(
            "Marketing Research Propositions (MRPs)",
            value=st.session_state.survey_mrps,
            height=120,
            placeholder="MRP1: There is a significant difference in brand satisfaction between Gen Z and Millennials.\nMRP2: ...",
            key="tab2_mrps_input",
        )
        st.session_state.survey_company = st.text_input(
            "Company / Product / Topic",
            value=st.session_state.survey_company,
            placeholder="e.g., Nike athletic shoes",
            key="tab2_company_input",
        )
        st.session_state.survey_population = st.text_input(
            "Target Population",
            value=st.session_state.survey_population,
            placeholder="e.g., College students aged 18-25",
            key="tab2_population_input",
        )
        st.session_state.survey_personas = st.text_area(
            "Target Personas (optional)",
            value=st.session_state.survey_personas,
            height=100,
            placeholder="Persona 1: Budget-conscious sophomore, rarely buys premium brands ...\nPersona 2: ...",
            key="tab2_personas_input",
        )

        st.divider()

        # Action buttons
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            gen_survey = st.button("✏️ Generate Draft", use_container_width=True, key="btn_gen_survey")
            review_survey = st.button("🔍 Review Survey", use_container_width=True, key="btn_review_survey")
        with btn_col2:
            pretest_survey = st.button("🧪 Pretest", use_container_width=True, key="btn_pretest")
            gen_synthetic = st.button("📊 Sample Data", use_container_width=True, key="btn_synthetic")

        # Show current draft
        if st.session_state.survey_draft:
            st.divider()
            st.markdown("##### Current Draft")
            st.text_area(
                "Edit your survey draft:",
                value=st.session_state.survey_draft,
                height=200,
                key="tab2_draft_editor",
                on_change=lambda: setattr(st.session_state, "survey_draft",
                                          st.session_state.tab2_draft_editor),
            )

        # Download synthetic data
        if st.session_state.synthetic_data:
            st.divider()
            st.download_button(
                "⬇️ Download Sample Data (CSV)",
                data=st.session_state.synthetic_data,
                file_name="synthetic_survey_data.csv",
                mime="text/csv",
                use_container_width=True,
            )

    with right_col:
        st.markdown("#### Survey Chat")

        # Render Tab 2 chat history
        chat_container = st.container(height=500)
        with chat_container:
            for msg in st.session_state.tab2_messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        # Handle button actions
        def _run_tab2_action(action_type):
            """Execute a Tab 2 button action and append result to chat."""
            mrps = st.session_state.survey_mrps
            personas = st.session_state.survey_personas or "Not provided"
            draft = st.session_state.survey_draft or "No draft yet"
            company = st.session_state.survey_company or "Not specified"
            population = st.session_state.survey_population or "Not specified"

            # Retrieve course context for survey design
            course_ctx = ""
            if vectorstore:
                query = f"survey design {company} {mrps[:200]}"
                course_ctx = retrieve_context(vectorstore, query, k=6)

            history_str = "\n".join(
                f"{m['role'].upper()}: {m['content'][:200]}"
                for m in st.session_state.tab2_messages[-6:]
            )

            try:
                if action_type == "generate":
                    question = (
                        f"Generate a complete Qualtrics-ready survey for: {company}. "
                        f"Target population: {population}. "
                        f"Make sure the survey includes a screening question, proper blocks, "
                        f"and all required scale types."
                    )
                    result = invoke_chain(api_key, SURVEY_BUILDER_PROMPT, {
                        "course_context": course_ctx,
                        "mrps": mrps or "Not provided — please ask the student",
                        "personas": personas,
                        "survey_draft": draft,
                        "history": history_str,
                        "question": question,
                    })
                    st.session_state.survey_draft = result

                elif action_type == "review":
                    if not draft or draft == "No draft yet":
                        return "Please generate or paste a survey draft first."
                    result = invoke_chain(api_key, SURVEY_REVIEW_PROMPT, {
                        "course_context": course_ctx,
                        "survey_draft": draft,
                    })

                elif action_type == "pretest":
                    if not draft or draft == "No draft yet":
                        return "Please generate or paste a survey draft first."
                    result = invoke_chain(api_key, PRETEST_PROMPT, {
                        "personas": personas,
                        "survey_draft": draft,
                    })
                    st.session_state.pretest_results = result

                elif action_type == "synthetic":
                    if not draft or draft == "No draft yet":
                        return "Please generate or paste a survey draft first."
                    result = invoke_chain(api_key, SYNTHETIC_DATA_PROMPT, {
                        "personas": personas,
                        "survey_draft": draft,
                        "n_rows": "25",
                    })
                    # Clean CSV from any markdown fences
                    cleaned = result.strip()
                    if cleaned.startswith("```"):
                        cleaned = cleaned.split("\n", 1)[-1]
                    if cleaned.endswith("```"):
                        cleaned = cleaned.rsplit("```", 1)[0]
                    st.session_state.synthetic_data = cleaned.strip()
                    result = f"Generated 25 rows of synthetic data. You can download it from the left panel.\n\n**Preview (first 5 rows):**\n```\n{chr(10).join(cleaned.strip().split(chr(10))[:6])}\n```"

                return result

            except RateLimitError as e:
                return f"**Rate limit reached.** {e}\n\nPlease wait a moment and try again."
            except Exception as e:
                return f"An error occurred: {e}"

        # Process button clicks
        action = None
        if gen_survey:
            action = "generate"
        elif review_survey:
            action = "review"
        elif pretest_survey:
            action = "pretest"
        elif gen_synthetic:
            action = "synthetic"

        if action:
            action_labels = {
                "generate": "Generating survey draft …",
                "review": "Reviewing survey …",
                "pretest": "Running pretest with simulated personas …",
                "synthetic": "Generating sample data …",
            }
            user_msg = {
                "generate": "Generate a survey draft based on my MRPs.",
                "review": "Review my current survey draft for quality issues.",
                "pretest": "Pretest my survey with simulated personas.",
                "synthetic": "Generate sample data for my survey.",
            }
            st.session_state.tab2_messages.append({"role": "user", "content": user_msg[action]})

            with st.spinner(action_labels[action]):
                result = _run_tab2_action(action)

            st.session_state.tab2_messages.append({"role": "assistant", "content": result})
            st.rerun()

        # Free-form chat input for Tab 2
        if tab2_prompt := st.chat_input("Ask follow-up questions about your survey …", key="tab2_chat"):
            st.session_state.tab2_messages.append({"role": "user", "content": tab2_prompt})

            course_ctx = ""
            if vectorstore:
                course_ctx = retrieve_context(vectorstore, tab2_prompt, k=6)

            history_str = "\n".join(
                f"{m['role'].upper()}: {m['content'][:200]}"
                for m in st.session_state.tab2_messages[-6:]
            )

            try:
                result = invoke_chain(api_key, SURVEY_BUILDER_PROMPT, {
                    "course_context": course_ctx,
                    "mrps": st.session_state.survey_mrps or "Not provided",
                    "personas": st.session_state.survey_personas or "Not provided",
                    "survey_draft": st.session_state.survey_draft or "No draft yet",
                    "history": history_str,
                    "question": tab2_prompt,
                })
            except RateLimitError as e:
                result = f"**Rate limit reached.** {e}\n\nPlease wait a moment and try again."
            except Exception as e:
                result = f"An error occurred: {e}"

            st.session_state.tab2_messages.append({"role": "assistant", "content": result})
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CLEAN & CODE MY CSV
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.caption("Upload your Qualtrics CSV for guided cleaning, recoding, and analysis.")

    # CSV uploader
    uploaded_csv = st.file_uploader(
        "Upload your Qualtrics CSV",
        type=["csv"],
        help="Upload your Qualtrics survey export (.csv) to clean, recode, or analyze it.",
        key="tab3_csv_uploader",
    )

    if uploaded_csv is not None:
        csv_bytes = uploaded_csv.getvalue()
        csv_str = csv_bytes.decode("utf-8")
        # Only re-process if new file
        if st.session_state.csv_filename != uploaded_csv.name:
            st.session_state.csv_data = csv_str
            st.session_state.csv_filename = uploaded_csv.name
            st.session_state.processed_csv = None
            st.session_state.codebook = ""
            # Auto-profile (0 LLM calls)
            try:
                df = pd.read_csv(io.StringIO(csv_str))
                st.session_state.csv_profile = build_csv_profile(df)
            except Exception as e:
                st.session_state.csv_profile = f"Error profiling CSV: {e}"

    if st.session_state.csv_data:
        st.success(f"Loaded: **{st.session_state.csv_filename}**")

        # Show data profile and preview
        profile_col, preview_col = st.columns([1, 1])

        with profile_col:
            st.markdown("##### Data Profile")
            st.markdown(
                f'<div class="profile-box"><pre>{st.session_state.csv_profile}</pre></div>',
                unsafe_allow_html=True,
            )

        with preview_col:
            st.markdown("##### Data Preview")
            try:
                preview_df = pd.read_csv(io.StringIO(st.session_state.csv_data))
                st.dataframe(preview_df.head(10), use_container_width=True, height=350)
            except Exception:
                st.warning("Could not preview CSV.")

        # Download buttons
        dl_col1, dl_col2, dl_col3, dl_col4 = st.columns(4)
        with dl_col1:
            if st.session_state.processed_csv:
                st.download_button(
                    "⬇️ Cleaned CSV",
                    data=st.session_state.processed_csv,
                    file_name="cleaned_data.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
        with dl_col2:
            gen_codebook = st.button("📋 Generate Codebook", use_container_width=True, key="btn_codebook")
        with dl_col3:
            if st.session_state.codebook:
                st.download_button(
                    "⬇️ Codebook",
                    data=st.session_state.codebook,
                    file_name="codebook.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
        with dl_col4:
            if st.button("🗑️ Remove CSV", use_container_width=True, key="btn_remove_csv"):
                st.session_state.csv_data = None
                st.session_state.csv_filename = None
                st.session_state.csv_profile = ""
                st.session_state.processed_csv = None
                st.session_state.codebook = ""
                st.rerun()

        # Generate codebook
        if gen_codebook:
            with st.spinner("Generating codebook …"):
                try:
                    cb = invoke_chain(api_key, CODEBOOK_PROMPT, {
                        "data_profile": st.session_state.csv_profile,
                    })
                    st.session_state.codebook = cb
                    st.session_state.tab3_messages.append(
                        {"role": "assistant", "content": f"**Codebook generated:**\n\n{cb}"}
                    )
                except RateLimitError as e:
                    st.error(f"Rate limit reached: {e}")
                except Exception as e:
                    st.error(f"Error: {e}")
            st.rerun()

        st.divider()

    elif not st.session_state.csv_data:
        st.info("Upload a CSV file above to get started with data cleaning and coding.")

    # Chat for cleaning/coding
    for msg in st.session_state.tab3_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if tab3_prompt := st.chat_input("Ask about your data — cleaning, recoding, analysis …", key="tab3_chat"):
        st.session_state.tab3_messages.append({"role": "user", "content": tab3_prompt})
        with st.chat_message("user"):
            st.markdown(tab3_prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing …"):
                if not st.session_state.csv_data:
                    answer = "Please upload a CSV file first using the uploader above."
                else:
                    # Reuse data_analyst_node logic
                    try:
                        import numpy as np
                        import re
                        df = pd.read_csv(io.StringIO(st.session_state.csv_data))
                        preview = _build_preview(df)

                        if _is_transform_request(tab3_prompt):
                            # Code execution path
                            result = invoke_chain(api_key, CODE_PROMPT, {
                                "preview": preview, "question": tab3_prompt,
                            })

                            code_match = re.search(r"```python\s*(.*?)```", result, re.DOTALL)
                            if code_match:
                                code = code_match.group(1).strip()
                                explanation = result[code_match.end():].strip()

                                exec_globals = {"__builtins__": SAFE_BUILTINS, "pd": pd, "np": np}
                                exec_locals = {"df": df}
                                try:
                                    exec(code, exec_globals, exec_locals)
                                    result_df = exec_locals.get("df", df)
                                except Exception as e:
                                    answer = (
                                        f"Error executing code:\n```\n{e}\n```\n\n"
                                        f"Code:\n```python\n{code}\n```\n\n"
                                        "Please try rephrasing your request."
                                    )
                                    result_df = None

                                if result_df is not None and isinstance(result_df, pd.DataFrame):
                                    # Update stored CSV
                                    new_csv = result_df.to_csv(index=False)
                                    st.session_state.processed_csv = new_csv
                                    st.session_state.csv_data = new_csv
                                    st.session_state.csv_profile = build_csv_profile(result_df)

                                    answer = (
                                        f"**Code applied:**\n```python\n{code}\n```\n\n"
                                        f"**Result preview** (first 10 rows):\n\n"
                                        f"{result_df.head(10).to_markdown(index=False)}\n\n"
                                        f"**Shape:** {result_df.shape[0]} rows x {result_df.shape[1]} columns"
                                    )
                                    if explanation:
                                        answer += f"\n\n{explanation}"
                                elif result_df is not None:
                                    answer = (
                                        f"**Code applied:**\n```python\n{code}\n```\n\n"
                                        f"**Result:**\n```\n{result_df}\n```"
                                    )
                                    if explanation:
                                        answer += f"\n\n{explanation}"
                            else:
                                answer = result
                        else:
                            # Description / guidance path
                            answer = invoke_chain(api_key, DESCRIBE_PROMPT, {
                                "preview": preview, "question": tab3_prompt,
                            })

                    except RateLimitError as e:
                        answer = f"**Rate limit reached.** {e}\n\nPlease wait a moment and try again."
                    except Exception as e:
                        answer = f"An error occurred: {e}"

            st.markdown(answer)

        st.session_state.tab3_messages.append({"role": "assistant", "content": answer})


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MATCH MRP TO ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.caption("Enter your MRPs and upload your CSV. Get course-grounded analysis recommendations.")

    # MRP input
    st.session_state.analysis_mrps = st.text_area(
        "Enter your Marketing Research Propositions (MRPs)",
        value=st.session_state.analysis_mrps,
        height=140,
        placeholder=(
            "MRP1: There is a significant difference in brand satisfaction between "
            "Gen Z and Millennials.\n"
            "MRP2: There is a positive relationship between social media engagement "
            "and purchase intention.\n"
            "MRP3: ..."
        ),
        key="tab4_mrps_input",
    )

    # Check for CSV
    if not st.session_state.csv_data:
        st.info("Please upload a CSV in the **Clean & Code My CSV** tab first, or upload one here.")
        tab4_csv = st.file_uploader(
            "Or upload CSV here:",
            type=["csv"],
            key="tab4_csv_uploader",
        )
        if tab4_csv is not None:
            csv_str = tab4_csv.getvalue().decode("utf-8")
            st.session_state.csv_data = csv_str
            st.session_state.csv_filename = tab4_csv.name
            try:
                df = pd.read_csv(io.StringIO(csv_str))
                st.session_state.csv_profile = build_csv_profile(df)
            except Exception as e:
                st.session_state.csv_profile = f"Error profiling CSV: {e}"
            st.rerun()

    # Analyze button
    analyze_btn = st.button(
        "🔬 Analyze — Map MRPs to Data & Recommend Tests",
        use_container_width=True,
        disabled=(not st.session_state.analysis_mrps or not st.session_state.csv_data),
        key="btn_analyze_mrps",
    )

    if analyze_btn:
        with st.spinner("Retrieving course methods and analyzing your MRPs …"):
            try:
                # Get course context via FAISS
                course_ctx = ""
                if vectorstore:
                    mrps_text = st.session_state.analysis_mrps
                    # Query FAISS with MRP-relevant terms
                    ctx_parts = []
                    ctx_parts.append(retrieve_context(
                        vectorstore,
                        f"statistical analysis methods {mrps_text[:300]}",
                        k=6,
                    ))
                    ctx_parts.append(retrieve_context(
                        vectorstore,
                        "scale types nominal ordinal interval ratio appropriate analysis",
                        k=4,
                    ))
                    course_ctx = "\n\n---\n\n".join(ctx_parts)

                # Get data profile
                data_profile = st.session_state.csv_profile
                if not data_profile:
                    df = pd.read_csv(io.StringIO(st.session_state.csv_data))
                    data_profile = build_csv_profile(df)

                result = invoke_chain(api_key, MRP_ANALYSIS_PROMPT, {
                    "course_context": course_ctx,
                    "data_profile": data_profile,
                    "mrps": st.session_state.analysis_mrps,
                })

                st.session_state.tab4_messages.append(
                    {"role": "user", "content": "Analyze my MRPs and recommend statistical tests."}
                )
                st.session_state.tab4_messages.append(
                    {"role": "assistant", "content": result}
                )

            except RateLimitError as e:
                st.error(f"Rate limit reached: {e}")
            except Exception as e:
                st.error(f"Error: {e}")

        st.rerun()

    # Render Tab 4 chat history
    for msg in st.session_state.tab4_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Follow-up chat
    if tab4_prompt := st.chat_input("Ask follow-up questions about your analysis …", key="tab4_chat"):
        st.session_state.tab4_messages.append({"role": "user", "content": tab4_prompt})
        with st.chat_message("user"):
            st.markdown(tab4_prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking …"):
                try:
                    course_ctx = ""
                    if vectorstore:
                        course_ctx = retrieve_context(vectorstore, tab4_prompt, k=6)

                    data_profile = st.session_state.csv_profile or "No CSV uploaded"

                    result = invoke_chain(api_key, MRP_ANALYSIS_PROMPT, {
                        "course_context": course_ctx,
                        "data_profile": data_profile,
                        "mrps": st.session_state.analysis_mrps or "Not provided",
                    })
                    answer = result
                except RateLimitError as e:
                    answer = f"**Rate limit reached.** {e}\n\nPlease wait a moment and try again."
                except Exception as e:
                    answer = f"An error occurred: {e}"

            st.markdown(answer)

        st.session_state.tab4_messages.append({"role": "assistant", "content": answer})
