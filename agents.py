"""
agents.py — LangGraph multi-agent workflow for Marketing Research Assistant.

Nodes:
  1. router        – keyword-based intent classifier (no LLM call)
  2. rag_retriever – fetch relevant FAISS chunks
  3. rag_generator – answer from retrieved context, with source citations
  4. data_analyst  – CSV analysis: description + Excel guidance OR code execution
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


# ── Graph State ──────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    question: str            # user's latest message
    chat_history: list       # prior turns for conversational context
    route: str               # "rag" | "data_analysis"
    context_docs: list       # retrieved Document objects
    context_text: str        # joined text for prompt injection
    rag_answer: str          # draft answer from RAG generator
    verified_answer: str     # final answer after verification
    csv_data: Optional[str]     # raw CSV string (if uploaded)
    csv_filename: Optional[str] # name of uploaded file
    final_answer: str        # output shown to the user


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
            is_rate_limit = any(kw in error_str for kw in ["429", "rate", "quota", "resource_exhausted"])
            if is_rate_limit and attempt < max_retries - 1:
                wait_time = (2 ** attempt) * 8  # 8s, 16s, 32s, ...
                time.sleep(wait_time)
                continue
            if is_rate_limit:
                raise RateLimitError(
                    "Rate limit reached. Please wait ~60 seconds and try again. "
                    "The free tier allows 30 requests/minute and 14,400 requests/day."
                )
            raise
    return chain.invoke(inputs)


# ── Node functions ───────────────────────────────────────────────────────────

# 1. ROUTER (keyword-based — no LLM call needed) ─────────────────────────────

# Keywords that indicate the user wants data TRANSFORMATION (code execution)
TRANSFORM_KEYWORDS = [
    "clean", "preprocess", "recode", "binary", "dummy", "encode",
    "drop", "remove", "delete", "rename", "merge", "recode",
    "normalize", "standardize", "convert", "replace", "map",
    "create column", "add column", "new variable", "compute",
]

# Keywords that indicate the user wants data DESCRIPTION or Excel guidance
# (these also trigger the data route, but use the descriptive agent)
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
    """Route based on whether CSV is present and question type."""
    question_lower = state["question"].lower()
    has_csv = bool(state.get("csv_data"))

    if has_csv:
        all_data_keywords = TRANSFORM_KEYWORDS + DESCRIBE_KEYWORDS
        if any(kw in question_lower for kw in all_data_keywords):
            return {"route": "data_analysis"}
    return {"route": "rag"}


# 2. RAG RETRIEVER ─────────────────────────────────────────────────────────────

def rag_retriever_node(state: AgentState, *, vectorstore: FAISS) -> dict:
    docs = vectorstore.similarity_search(state["question"], k=8)
    context_text = "\n\n---\n\n".join(doc.page_content for doc in docs)
    return {"context_docs": docs, "context_text": context_text}


# 3. RAG GENERATOR ────────────────────────────────────────────────────────────

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
        "2. Cite sources when possible (e.g., Session 7 — Slide 9).\n"
        "3. Be thorough and pedagogical — treat the student as a beginner.\n"
        "4. When discussing statistical tests, specify which scale types they "
        "   are appropriate for.\n"
        "5. If the context contains relevant information, ALWAYS provide a "
        "   substantive answer by synthesizing across all context passages.\n"
        "6. ONLY if the context truly contains NO relevant information at all, "
        "   respond with EXACTLY this message and nothing else:\n"
        '   "{rejection}"\n'
        "7. NEVER fabricate facts not supported by the context.\n\n"
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
    answer = result.content
    return {"rag_answer": answer, "final_answer": answer}


# 4. DATA ANALYSIS AGENT ─────────────────────────────────────────────────────

def _is_transform_request(question: str) -> bool:
    """Determine if the user wants to transform/modify data (vs. describe it).

    If the question contains guidance-seeking phrases (e.g. "how to", "steps",
    "step by step", "guide", "what should"), treat it as a DESCRIPTIVE request
    even if it also mentions transform keywords like "clean".
    """
    q = question.lower()
    # Guidance phrases override transform keywords — the student wants
    # instructions, not code execution.
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


# -- 5a. Descriptive / Excel-guidance prompt (no code execution) ──────────────

DESCRIBE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert Marketing Research Data Analysis Teaching Assistant "
        "who helps students understand their survey data and guides them through "
        "analysis using Microsoft Excel and the Data Analysis ToolPak.\n\n"
        "You are given a preview of a Qualtrics CSV dataset. Your job is to:\n\n"
        "1. **Describe the dataset**: Explain what the data contains, what each "
        "   column/variable represents, the number of responses, and any data "
        "   quality issues (missing values, incomplete responses, metadata rows).\n\n"
        "2. **Classify each variable by scale type**:\n"
        "   - **Nominal**: Categories with no order (e.g., Gender, Brand, Yes/No)\n"
        "   - **Ordinal**: Ordered categories, unequal intervals (e.g., Likert scales, satisfaction ratings)\n"
        "   - **Interval**: Equal intervals, no true zero (e.g., composite Likert scores, temperature)\n"
        "   - **Ratio**: Equal intervals with true zero (e.g., Age, Income, Duration, Count)\n"
        "   For each variable, state its scale type and what analyses are appropriate.\n\n"
        "3. **Provide Excel step-by-step instructions** for BOTH Windows and Mac:\n"
        "   - Exact menu paths (e.g., Data > Data Analysis > Descriptive Statistics)\n"
        "   - Note Mac differences where applicable (e.g., Tools > Excel Add-ins for ToolPak on Mac)\n"
        "   - Cell references and formula examples using actual column names\n"
        "   - How to install the Data Analysis ToolPak if needed\n"
        "   - How to use IF() for binary recoding, COUNTIF for frequencies, etc.\n\n"
        "4. **Recommend analyses based on scale types**: For each variable or "
        "   pair of variables, suggest the appropriate statistical test:\n"
        "   - Nominal: Frequency tables, chi-square, cross-tabs\n"
        "   - Ordinal: Median, Mann-Whitney, Kruskal-Wallis\n"
        "   - Interval/Ratio: Mean, t-test, ANOVA, correlation, regression\n\n"
        "5. **Suggest specific MRP next steps** for this dataset.\n\n"
        "Format with clear markdown headings, numbered steps, and concrete "
        "examples using actual column names from the data.\n"
        "Be thorough and pedagogical — treat the student as a beginner.\n"
        "NEVER fabricate information. Only describe what you can see in the data preview.\n",
    ),
    ("human", "Data preview:\n{preview}\n\nStudent's question: {question}"),
])


# -- 5b. Code execution prompt (for data transformations) ─────────────────────

CODE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a Data Analysis Agent specialized in cleaning Qualtrics survey "
        "data. You will be given a preview of the CSV data and the user's request.\n\n"
        "Your job:\n"
        "1. Write SHORT, focused Python/pandas code to fulfill the request.\n"
        "2. The DataFrame is pre-loaded as `df`. Modify `df` in-place or reassign "
        "   it. The last expression must be `df`.\n"
        "3. Wrap the code in ```python ... ```.\n"
        "4. Only use pandas (`pd`) and numpy (`np`) — both are pre-imported.\n"
        "5. Do NOT use print(). Do NOT import anything.\n"
        "6. Keep the code simple and direct — avoid unnecessary complexity.\n"
        "7. Use only basic Python builtins (len, str, int, float, list, dict, "
        "   range, enumerate, zip, map, filter, sorted, min, max, sum, etc.).\n\n"
        "After the code block, provide:\n"
        "a) A brief explanation of what the code does.\n"
        "b) The equivalent Excel step-by-step instructions (for BOTH Windows "
        "   and Mac where they differ) so the student can do it manually.\n"
        "c) Which scale type each affected variable is (nominal, ordinal, "
        "   interval, ratio) and what that means for appropriate analyses.\n",
    ),
    ("human", "Data preview:\n{preview}\n\nRequest: {question}"),
])


def _build_preview(df: pd.DataFrame) -> str:
    """Build a compact text preview of a DataFrame."""
    buf = io.StringIO()
    buf.write(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n\n")
    buf.write(f"Columns & dtypes:\n{df.dtypes.to_string()}\n\n")
    buf.write(f"First 5 rows:\n{df.head().to_string()}\n\n")

    # Add missing value info
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        buf.write(f"Missing values:\n{missing.to_string()}\n\n")
    else:
        buf.write("Missing values: None\n\n")

    # Add unique value counts for object columns (first 10 only)
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
        return {"final_answer": "No CSV file uploaded. Please upload a Qualtrics CSV to analyze."}

    try:
        df = pd.read_csv(io.StringIO(csv_data))
    except Exception as e:
        return {"final_answer": f"Error reading CSV: {e}"}

    preview = _build_preview(df)
    question = state["question"]

    # ── Descriptive / Excel guidance path ────────────────────────────────
    if not _is_transform_request(question):
        chain = DESCRIBE_PROMPT | llm
        result = invoke_with_retry(chain, {"preview": preview, "question": question})
        return {"final_answer": result.content}

    # ── Code execution path ──────────────────────────────────────────────
    chain = CODE_PROMPT | llm
    result = invoke_with_retry(chain, {"preview": preview, "question": question})

    # Extract the code block
    code_match = re.search(r"```python\s*(.*?)```", result.content, re.DOTALL)
    if not code_match:
        # No code block — return the full text response (may be guidance-only)
        return {"final_answer": result.content}

    code = code_match.group(1).strip()

    # Extract any explanation text after the code block
    explanation = result.content[code_match.end():].strip()

    # Execute in a sandbox with safe builtins
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
                "Please try rephrasing your request with more specific instructions."
            )
        }

    # Ensure result_df is a DataFrame
    if not isinstance(result_df, pd.DataFrame):
        return {
            "final_answer": (
                f"**Code applied:**\n```python\n{code}\n```\n\n"
                f"**Result:**\n```\n{result_df}\n```\n\n"
                + (f"\n{explanation}" if explanation else "")
            )
        }

    # Build a response
    answer_parts = [
        f"**Code applied:**\n```python\n{code}\n```\n",
        f"**Result preview** (first 10 rows):\n\n{result_df.head(10).to_markdown(index=False)}\n",
        f"**Shape:** {result_df.shape[0]} rows x {result_df.shape[1]} columns",
    ]
    if explanation:
        answer_parts.append(f"\n{explanation}")

    # Store the processed df as CSV so the app can offer a download
    return {
        "final_answer": "\n".join(answer_parts),
        "csv_data": result_df.to_csv(index=False),
    }


# ── Graph builder ────────────────────────────────────────────────────────────

def build_graph(api_key: str, vectorstore: FAISS) -> StateGraph:
    """Compile and return the LangGraph workflow."""
    llm = get_llm(api_key)

    # Bind dependencies via closures
    def _router(state):
        return router_node(state)

    def _retriever(state):
        return rag_retriever_node(state, vectorstore=vectorstore)

    def _generator(state):
        return rag_generator_node(state, llm=llm)

    def _analyst(state):
        return data_analyst_node(state, llm=llm)

    # Define the graph
    workflow = StateGraph(AgentState)

    workflow.add_node("router", _router)
    workflow.add_node("rag_retriever", _retriever)
    workflow.add_node("rag_generator", _generator)
    workflow.add_node("data_analyst", _analyst)

    # Entry point
    workflow.set_entry_point("router")

    # Conditional edge from router
    workflow.add_conditional_edges(
        "router",
        lambda state: state["route"],
        {
            "rag": "rag_retriever",
            "data_analysis": "data_analyst",
        },
    )

    # RAG pipeline: retriever → generator → END (no verifier — saves LLM calls)
    workflow.add_edge("rag_retriever", "rag_generator")
    workflow.add_edge("rag_generator", END)

    # Data analysis → END
    workflow.add_edge("data_analyst", END)

    return workflow.compile()
