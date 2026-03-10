"""
agents.py — LangGraph multi-agent workflow for MKTG 323 Marketing Research
Contest Assistant.

Tab 1 (Class Content): RAG pipeline — retriever → generator → END
Tabs 2-4: Direct chain invocations via invoke_chain()
"""

import io
import re
import time
from typing import TypedDict, Optional

import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, END

# ── Shared constants ─────────────────────────────────────────────────────────

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REJECTION_MSG = (
    "I'm sorry, but that information is not covered in our MKTG 323 course "
    "materials or slides. Please reach out to Professor Rahul Suhag directly "
    "for further help."
)

# Safe builtins for the code execution sandbox
SAFE_BUILTINS = {
    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
    "enumerate": enumerate, "filter": filter, "float": float, "format": format,
    "frozenset": frozenset, "hasattr": hasattr, "int": int, "isinstance": isinstance,
    "issubclass": issubclass, "iter": iter, "len": len, "list": list,
    "map": map, "max": max, "min": min, "next": next, "print": print,
    "range": range, "repr": repr, "reversed": reversed, "round": round,
    "set": set, "slice": slice, "sorted": sorted, "str": str, "sum": sum,
    "tuple": tuple, "type": type, "zip": zip, "None": None, "True": True,
    "False": False, "KeyError": KeyError, "ValueError": ValueError,
    "TypeError": TypeError, "IndexError": IndexError, "AttributeError": AttributeError,
}


# ── Graph State (Tab 1 RAG) ─────────────────────────────────────────────────

class AgentState(TypedDict):
    question: str
    chat_history: list
    route: str
    context_docs: list
    context_text: str
    rag_answer: str
    verified_answer: str
    csv_data: Optional[str]
    csv_filename: Optional[str]
    final_answer: str


# ── LLM factory ──────────────────────────────────────────────────────────────

def get_llm(api_key: str) -> ChatGroq:
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=api_key,
        temperature=0.2,
    )


class RateLimitError(Exception):
    """Raised when Groq API rate limit is exhausted after retries."""
    pass


def invoke_with_retry(chain, inputs, max_retries=4):
    """Invoke a LangChain chain with exponential backoff for rate limits."""
    for attempt in range(max_retries):
        try:
            return chain.invoke(inputs)
        except Exception as e:
            error_str = str(e).lower()
            is_rate_limit = any(kw in error_str for kw in [
                "429", "rate", "quota", "resource_exhausted",
            ])
            if is_rate_limit and attempt < max_retries - 1:
                wait_time = (2 ** attempt) * 8
                time.sleep(wait_time)
                continue
            if is_rate_limit:
                raise RateLimitError(
                    "Rate limit reached. Please wait ~60 seconds and try again. "
                    "The free tier allows 30 requests/minute and 14,400 requests/day."
                )
            raise
    return chain.invoke(inputs)


# ── Direct chain helper (Tabs 2-4) ──────────────────────────────────────────

def invoke_chain(api_key: str, prompt_template: ChatPromptTemplate,
                 inputs: dict) -> str:
    """Invoke a single LLM chain with retry. Returns the response text."""
    llm = get_llm(api_key)
    chain = prompt_template | llm
    return invoke_with_retry(chain, inputs).content


# ── RAG retrieval helper ─────────────────────────────────────────────────────

def retrieve_context(vectorstore: FAISS, query: str, k: int = 8) -> str:
    """Retrieve top-k chunks from FAISS and return joined text."""
    docs = vectorstore.similarity_search(query, k=k)
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


# ── CSV profiling (pure pandas, 0 LLM calls) ────────────────────────────────

def build_csv_profile(df: pd.DataFrame) -> str:
    """Generate a comprehensive data profile using only pandas."""
    buf = io.StringIO()
    buf.write(f"DATASET OVERVIEW\n")
    buf.write(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n\n")

    buf.write("VARIABLE SUMMARY:\n")
    for col in df.columns:
        dtype = df[col].dtype
        n_missing = int(df[col].isnull().sum())
        n_unique = int(df[col].nunique())
        pct_missing = f"{n_missing / len(df) * 100:.0f}%" if len(df) > 0 else "N/A"

        buf.write(f"\n  {col}:\n")
        buf.write(f"    dtype={dtype}, missing={n_missing} ({pct_missing}), unique={n_unique}\n")

        if pd.api.types.is_numeric_dtype(df[col]) and n_unique > 0:
            desc = df[col].describe()
            buf.write(f"    mean={desc.get('mean', 'N/A'):.2f}, "
                      f"std={desc.get('std', 'N/A'):.2f}, "
                      f"min={desc.get('min', 'N/A')}, "
                      f"max={desc.get('max', 'N/A')}\n")
        elif n_unique > 0:
            top_vals = df[col].value_counts().head(5)
            items = [f"{v!r}: {c}" for v, c in top_vals.items()]
            buf.write(f"    top values: {', '.join(items)}\n")

    buf.write(f"\nFIRST 3 ROWS:\n{df.head(3).to_string()}\n")
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CLASS CONTENT (RAG)
# ══════════════════════════════════════════════════════════════════════════════

# 1a. Router (keyword-based — no LLM call)

TRANSFORM_KEYWORDS = [
    "clean", "preprocess", "recode", "binary", "dummy", "encode",
    "drop", "remove", "delete", "rename", "merge",
    "normalize", "standardize", "convert", "replace", "map",
    "create column", "add column", "new variable", "compute",
]

DESCRIBE_KEYWORDS = [
    "summarize", "summary", "describe", "what does", "what is in",
    "overview", "contain", "look like", "structure", "explore",
    "excel", "toolpak", "how to", "how do i", "step by step",
    "analyze in excel", "frequency", "crosstab", "pivot",
    "histogram", "chart", "mean", "median", "statistics",
    "descriptive", "count", "distribution", "missing",
    "filter", "sort", "groupby", "average",
]


def router_node(state: AgentState) -> dict:
    question_lower = state["question"].lower()
    has_csv = bool(state.get("csv_data"))
    if has_csv:
        all_data_keywords = TRANSFORM_KEYWORDS + DESCRIBE_KEYWORDS
        if any(kw in question_lower for kw in all_data_keywords):
            return {"route": "data_analysis"}
    return {"route": "rag"}


# 1b. RAG Retriever

def rag_retriever_node(state: AgentState, *, vectorstore: FAISS) -> dict:
    docs = vectorstore.similarity_search(state["question"], k=8)
    context_text = "\n\n---\n\n".join(doc.page_content for doc in docs)
    return {"context_docs": docs, "context_text": context_text}


# 1c. RAG Generator

RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are the MKTG 323 Marketing Research Teaching Assistant at "
        "Texas A&M University, Mays Business School (Professor Rahul Suhag's class). "
        "You help students learn marketing research concepts using the course "
        "materials provided as context below.\n\n"
        "YOUR CAPABILITIES:\n"
        "1. **Explain concepts** — scales (nominal, ordinal, interval, ratio), "
        "   comparative vs. non-comparative scales, sampling methods, research "
        "   designs, qualitative vs. quantitative methods, constructs, survey "
        "   design, and all topics covered in the course slides.\n"
        "2. **Excel guidance** — Step-by-step instructions for Data Analysis "
        "   ToolPak, descriptive statistics, t-tests, ANOVA, chi-square, "
        "   correlation, regression. Always include BOTH Windows and Mac paths.\n"
        "3. **Data cleaning** — Guide students on preprocessing Qualtrics survey "
        "   exports: removing metadata rows, handling missing data, recoding "
        "   variables, binary coding, reverse coding.\n"
        "4. **Survey question feedback** — Help students write, improve, or "
        "   evaluate survey questions. Identify issues like double-barreled, "
        "   leading, double-negative, or loaded questions. Suggest proper scale "
        "   types and response options based on course materials.\n\n"
        "RULES:\n"
        "1. Use the context below as your primary source. Synthesize and explain "
        "   the information in a clear, student-friendly way.\n"
        "2. Cite sources using the format: (Session X — Slide Y) or (Source: filename).\n"
        "3. Be thorough and pedagogical — treat the student as a beginner.\n"
        "4. When discussing statistical tests, specify which scale types they "
        "   are appropriate for.\n"
        "5. If the context contains relevant information, ALWAYS provide a "
        "   substantive answer by synthesizing across all context passages.\n"
        "6. If you provide practical guidance beyond what the slides explicitly "
        "   state, clearly prefix that section with: **Beyond slide coverage:**\n"
        "7. ONLY if the context truly contains NO relevant information at all, "
        "   respond with EXACTLY this message and nothing else:\n"
        '   "{rejection}"\n'
        "8. NEVER fabricate facts not supported by the context.\n\n"
        "Context:\n{context}\n",
    ),
    ("human", "{question}"),
])


def rag_generator_node(state: AgentState, *, llm) -> dict:
    chain = RAG_PROMPT | llm
    result = invoke_with_retry(chain, {
        "question": state["question"],
        "context": state["context_text"],
        "rejection": REJECTION_MSG,
    })
    return {"rag_answer": result.content, "final_answer": result.content}


# 1d. Data Analyst (CSV description + code execution)

def _is_transform_request(question: str) -> bool:
    q = question.lower()
    GUIDANCE_PHRASES = [
        "how to", "how do i", "how should", "how can i",
        "step by step", "steps to", "steps for", "steps i should",
        "guide", "guidance", "instructions", "teach", "explain",
        "what should", "what do i", "what are the steps", "walk me through",
        "show me how", "help me understand",
    ]
    if any(phrase in q for phrase in GUIDANCE_PHRASES):
        return False
    return any(kw in q for kw in TRANSFORM_KEYWORDS)


DESCRIBE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a MKTG 323 Data Analysis Teaching Assistant. "
        "Help the student understand their Qualtrics survey data and guide them "
        "through analysis using Microsoft Excel and the Data Analysis ToolPak.\n\n"
        "Given a data preview, do the following:\n"
        "1. Describe the dataset clearly.\n"
        "2. Classify each variable by scale type (nominal, ordinal, interval, ratio).\n"
        "3. Provide Excel instructions for BOTH Windows and Mac.\n"
        "4. Recommend analyses appropriate to the scale types.\n"
        "5. Suggest next steps.\n\n"
        "Be thorough, pedagogical, and use actual column names from the data.\n"
        "NEVER fabricate information. Only describe what you see in the data preview.\n",
    ),
    ("human", "Data preview:\n{preview}\n\nStudent's question: {question}"),
])


CODE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a Data Analysis Agent for cleaning Qualtrics survey data.\n\n"
        "Your job:\n"
        "1. Write SHORT Python/pandas code to fulfill the request.\n"
        "2. The DataFrame is pre-loaded as `df`. The last expression must be `df`.\n"
        "3. Wrap code in ```python ... ```.\n"
        "4. Only use pandas (`pd`) and numpy (`np`) — pre-imported.\n"
        "5. Do NOT use print() or import anything.\n\n"
        "After the code, provide:\n"
        "a) Brief explanation.\n"
        "b) Equivalent Excel steps (Windows and Mac).\n"
        "c) Scale type of affected variables and appropriate analyses.\n",
    ),
    ("human", "Data preview:\n{preview}\n\nRequest: {question}"),
])


def _build_preview(df: pd.DataFrame) -> str:
    """Build a compact text preview of a DataFrame."""
    buf = io.StringIO()
    buf.write(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n\n")
    buf.write(f"Columns & dtypes:\n{df.dtypes.to_string()}\n\n")
    buf.write(f"First 5 rows:\n{df.head().to_string()}\n\n")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        buf.write(f"Missing values:\n{missing.to_string()}\n\n")
    else:
        buf.write("Missing values: None\n\n")
    obj_cols = df.select_dtypes(include=["object"]).columns[:10]
    if len(obj_cols) > 0:
        buf.write("Sample unique values (first 10 object columns):\n")
        for col in obj_cols:
            uniques = df[col].dropna().unique()[:5]
            buf.write(f"  {col}: {list(uniques)}\n")
    return buf.getvalue()


def data_analyst_node(state: AgentState, *, llm) -> dict:
    csv_data = state.get("csv_data")
    if not csv_data:
        return {"final_answer": "No CSV file uploaded. Please upload a CSV in the 'Clean & Code My CSV' tab."}

    try:
        df = pd.read_csv(io.StringIO(csv_data))
    except Exception as e:
        return {"final_answer": f"Error reading CSV: {e}"}

    preview = _build_preview(df)
    question = state["question"]

    if not _is_transform_request(question):
        chain = DESCRIBE_PROMPT | llm
        result = invoke_with_retry(chain, {"preview": preview, "question": question})
        return {"final_answer": result.content}

    chain = CODE_PROMPT | llm
    result = invoke_with_retry(chain, {"preview": preview, "question": question})

    code_match = re.search(r"```python\s*(.*?)```", result.content, re.DOTALL)
    if not code_match:
        return {"final_answer": result.content}

    code = code_match.group(1).strip()
    explanation = result.content[code_match.end():].strip()

    import numpy as np
    exec_globals = {"__builtins__": SAFE_BUILTINS, "pd": pd, "np": np}
    exec_locals = {"df": df}
    try:
        exec(code, exec_globals, exec_locals)
        result_df = exec_locals.get("df", df)
    except Exception as e:
        return {
            "final_answer": (
                f"Error executing generated code:\n```\n{e}\n```\n\n"
                f"Generated code was:\n```python\n{code}\n```\n\n"
                "Please try rephrasing your request."
            )
        }

    if not isinstance(result_df, pd.DataFrame):
        return {
            "final_answer": (
                f"**Code applied:**\n```python\n{code}\n```\n\n"
                f"**Result:**\n```\n{result_df}\n```\n\n"
                + (f"\n{explanation}" if explanation else "")
            )
        }

    answer_parts = [
        f"**Code applied:**\n```python\n{code}\n```\n",
        f"**Result preview** (first 10 rows):\n\n{result_df.head(10).to_markdown(index=False)}\n",
        f"**Shape:** {result_df.shape[0]} rows x {result_df.shape[1]} columns",
    ]
    if explanation:
        answer_parts.append(f"\n{explanation}")

    return {
        "final_answer": "\n".join(answer_parts),
        "csv_data": result_df.to_csv(index=False),
    }


# ── Graph builder (Tab 1 only) ──────────────────────────────────────────────

def build_graph(api_key: str, vectorstore: FAISS) -> StateGraph:
    """Compile and return the LangGraph workflow for Tab 1 RAG."""
    llm = get_llm(api_key)

    def _router(state):
        return router_node(state)

    def _retriever(state):
        return rag_retriever_node(state, vectorstore=vectorstore)

    def _generator(state):
        return rag_generator_node(state, llm=llm)

    def _analyst(state):
        return data_analyst_node(state, llm=llm)

    workflow = StateGraph(AgentState)
    workflow.add_node("router", _router)
    workflow.add_node("rag_retriever", _retriever)
    workflow.add_node("rag_generator", _generator)
    workflow.add_node("data_analyst", _analyst)
    workflow.set_entry_point("router")
    workflow.add_conditional_edges(
        "router",
        lambda state: state["route"],
        {"rag": "rag_retriever", "data_analysis": "data_analyst"},
    )
    workflow.add_edge("rag_retriever", "rag_generator")
    workflow.add_edge("rag_generator", END)
    workflow.add_edge("data_analyst", END)
    return workflow.compile()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SURVEY BUILDER PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

SURVEY_BUILDER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a Marketing Research Survey Design Consultant for MKTG 323 at "
        "Texas A&M University (Professor Rahul Suhag's class).\n\n"
        "TASK: Help the student build a Qualtrics-ready survey grounded in their "
        "Marketing Research Propositions (MRPs) and class best practices.\n\n"
        "COURSE CONTEXT (from class materials):\n{course_context}\n\n"
        "SURVEY DESIGN RULES (grounded in class):\n"
        "- Start from research objectives / MRPs.\n"
        "- Include a proper opening statement.\n"
        "- Put the screening question FIRST.\n"
        "- Include clear instructions for each section.\n"
        "- Organize logically: Block 1 = general brand/company perceptions, "
        "  Block 2 = research-topic questions tied to MRPs, Block 3 = demographics.\n"
        "- Use simple, unambiguous language.\n"
        "- Avoid leading, double-barreled, loaded, and double-negative questions.\n"
        "- Make response tasks easy.\n"
        "- Ensure response options are mutually exclusive and collectively exhaustive.\n"
        "- Include all required scale types from the project: nominal, ordinal, "
        "  interval, ratio, and specific formats (Likert, Semantic Differential, "
        "  Behavioral Intention, Comparative Rating) where they naturally fit.\n\n"
        "FOR EACH QUESTION, PROVIDE:\n"
        "- Question text\n"
        "- Question type / format (e.g., multiple choice, Likert, semantic diff)\n"
        "- Scale type (nominal / ordinal / interval / ratio)\n"
        "- Which MRP it maps to\n"
        "- Why it is included\n"
        "- Any skip logic suggestions\n\n"
        "WHAT TO ASK THE STUDENT IF MISSING:\n"
        "If the student has NOT provided all of the following, ask for them:\n"
        "1. Company / product / service / topic\n"
        "2. Target population\n"
        "3. Research problem or opportunity\n"
        "4. MRP(s) / research questions\n"
        "5. Any secondary research findings\n"
        "6. Any focus group findings\n"
        "7. Target personas (ask for: age, relationship to brand, usage frequency, "
        "   attitudes, spending sensitivity, experience level, behaviors)\n\n"
        "STUDENT'S MRPs:\n{mrps}\n\n"
        "TARGET PERSONAS:\n{personas}\n\n"
        "CURRENT SURVEY DRAFT (if any):\n{survey_draft}\n\n"
        "CONVERSATION HISTORY:\n{history}\n",
    ),
    ("human", "{question}"),
])


SURVEY_REVIEW_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a Survey Quality Reviewer for MKTG 323 at Texas A&M University.\n\n"
        "COURSE CONTEXT:\n{course_context}\n\n"
        "Review the survey below and flag ALL issues. For each issue provide:\n"
        "1. **Issue type** (ambiguous / leading / double-barreled / double-negative "
        "   / loaded / missing time frame / non-exclusive options / non-exhaustive "
        "   / difficult recall / confusing instructions / repetitive / weak screening "
        "   / weak skip logic / social desirability risk / order effect)\n"
        "2. **Exact problematic question**\n"
        "3. **Why it is a problem**\n"
        "4. **Suggested revision**\n"
        "5. **Grounded in class?** — whether the fix is directly from class materials "
        "   or practical help beyond direct slide coverage.\n\n"
        "Also check:\n"
        "- Are all required scale types present (nominal, ordinal, interval, ratio)?\n"
        "- Are Likert, Semantic Differential, Behavioral Intention, and Comparative "
        "  Rating formats used where appropriate?\n"
        "- If a required scale/format is missing, suggest where to add it naturally.\n\n"
        "SURVEY TO REVIEW:\n{survey_draft}\n",
    ),
    ("human", "Review this survey for quality issues and provide specific revision suggestions."),
])


PRETEST_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a Survey Pretesting Tool for MKTG 323 at Texas A&M University.\n\n"
        "Given a survey draft and target personas, simulate respondents taking "
        "the survey. Create at least 3 persona respondents PLUS 1 'bad' respondent "
        "(distracted/satisficing).\n\n"
        "FOR EACH GOOD PERSONA:\n"
        "1. Persona summary (name, age, background, attitude toward topic)\n"
        "2. Simulated answers to each question\n"
        "3. Questions they found confusing\n"
        "4. Missing response options they needed\n"
        "5. Questions that felt biased or leading\n"
        "6. Any question that would make them stop the survey\n"
        "7. Logic/flow issues encountered\n"
        "8. One suggestion to improve the experience\n\n"
        "FOR THE BAD PERSONA (distracted/satisficing respondent):\n"
        "1. Confusing instructions they would skip\n"
        "2. Repetitive questions they would speed through\n"
        "3. Easy misinterpretations\n"
        "4. Points where they would click through mindlessly\n"
        "5. Wording that creates socially desirable answers\n"
        "6. Top 5 risks to data quality from this survey\n\n"
        "TARGET PERSONAS:\n{personas}\n\n"
        "SURVEY DRAFT:\n{survey_draft}\n",
    ),
    ("human", "Pretest this survey with simulated respondents and provide a structured report."),
])


SYNTHETIC_DATA_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Generate synthetic survey response data as a CSV based on the survey "
        "structure below. This is for PRETESTING and DEBUGGING only — not valid "
        "for real project inference.\n\n"
        "RULES:\n"
        "1. Include a header row with question codes (Q1, Q2, etc.).\n"
        "2. Include columns: ResponseId, StartDate, EndDate, Duration__in_seconds_, "
        "   Finished, Progress.\n"
        "3. Create realistic, varied responses matching the survey's scale types.\n"
        "4. Include some missing values (~5-10%% randomly).\n"
        "5. Include 2-3 speeders (very short Duration).\n"
        "6. Include 1-2 incomplete responses (Finished=0, Progress<100).\n"
        "7. Use the personas to create realistic variation.\n"
        "8. Output ONLY valid CSV content — no explanation, no markdown fences.\n\n"
        "PERSONAS:\n{personas}\n\n"
        "SURVEY STRUCTURE:\n{survey_draft}\n",
    ),
    ("human", "Generate {n_rows} rows of synthetic survey data as CSV."),
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MRP ANALYSIS PROMPT
# ══════════════════════════════════════════════════════════════════════════════

MRP_ANALYSIS_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an MRP Analysis Advisor for MKTG 323 at Texas A&M University "
        "(Professor Rahul Suhag's class).\n\n"
        "TASK: Map the student's Marketing Research Propositions (MRPs) to their "
        "actual data variables and recommend appropriate statistical analyses.\n\n"
        "COURSE METHODS (from class materials — use ONLY these unless noted):\n"
        "{course_context}\n\n"
        "STUDENT'S DATA PROFILE:\n{data_profile}\n\n"
        "FOR EACH MRP, PROVIDE:\n"
        "1. **MRP / Research Question**: (restate it)\n"
        "2. **Question type**: (describe vs. compare groups vs. test association "
        "   vs. test difference in means vs. examine relationship vs. predict/explain)\n"
        "3. **Likely relevant variables**: (from the actual dataset columns)\n"
        "4. **Suspected scale types**: (nominal/ordinal/interval/ratio for each variable)\n"
        "5. **Recommended analysis**: (from course methods above)\n"
        "6. **Why this analysis fits**: (connect scale types + question type to method)\n"
        "7. **What it means in plain English**: (what insight the student gets)\n"
        "8. **Excel steps**: (exact menu path for BOTH Windows and Mac)\n"
        "9. **How to report results**: (APA-style template)\n"
        "10. **Required preparation**: (cleaning, coding, dummy variables needed)\n"
        "11. **Cautions / limitations**\n"
        "12. **Course grounded?**: state whether this recommendation is directly "
        "    from class materials or **Beyond slide coverage**.\n\n"
        "IMPORTANT RULES:\n"
        "- Do NOT recommend every possible test. Only recommend the BEST FIT from "
        "  the course methods for this specific MRP and these specific variables.\n"
        "- If the data cannot properly answer an MRP, say so clearly and explain "
        "  what variable/data is missing.\n"
        "- If uncertain about a variable's scale type, say so and ask the student "
        "  to confirm.\n"
        "- Never give generic scale-type-to-test mappings. Always personalize to "
        "  the student's actual MRPs and actual column names.\n",
    ),
    ("human", "My MRPs:\n{mrps}\n\nMap these to my data and recommend analyses."),
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CODEBOOK PROMPT
# ══════════════════════════════════════════════════════════════════════════════

CODEBOOK_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Generate a data codebook / data dictionary for this Qualtrics survey dataset.\n\n"
        "For each variable include:\n"
        "- Variable name\n"
        "- Description (infer from column name and sample values)\n"
        "- Scale type (nominal / ordinal / interval / ratio)\n"
        "- Valid values or range\n"
        "- Notes (e.g., 'Qualtrics metadata — exclude from analysis', "
        "  'Needs recoding', 'Likert 5-point')\n\n"
        "Format as a clean markdown table.\n"
        "Flag Qualtrics metadata columns (ResponseId, StartDate, EndDate, Status, "
        "IPAddress, Progress, Duration, Finished, RecordedDate, etc.) separately.\n\n"
        "DATA PROFILE:\n{data_profile}\n",
    ),
    ("human", "Generate a codebook for this dataset."),
])
