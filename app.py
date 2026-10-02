"""
Text-to-SQL Assistant
=====================
Ask questions in plain English -> get SQL, an explanation and the query result.

Stack : Streamlit (UI) + LangChain (prompt | model | parser) + Groq (LLM) + SQLite

Highlights
- Pydantic structured output (sql_query + explanation)
- Read-only SQL guard (SELECT-only, single statement, read-only DB connection)
- Auto-repair: if the generated SQL fails, the error is sent back to the LLM once
- Use the built-in sample DB or upload your own CSV
- Query history, CSV download, quick chart
"""
from __future__ import annotations

import hashlib
import io
import os
import re
import sqlite3
import tempfile
import time
from contextlib import closing
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from setup_db import DB_PATH, create_db

load_dotenv()  # reads GROQ_API_KEY from a local .env file (if present)

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
# Groq changes its model list often. We ask the Groq API which models YOUR key can use.
# This list is only the preferred order + fallback if the API call fails.
PREFERRED_MODELS = [
    "llama-3.1-8b-instant",  # fastest, no reasoning step -> default
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "llama-3.3-70b-versatile",
]
NON_CHAT_HINTS = ("whisper", "guard", "tts", "orpheus", "compound", "embed")
MAX_ROWS = 1000
MAX_UPLOAD_MB = 10
SAMPLE_QUESTIONS = [
    "Show employees with salary greater than 50000",
    "Average salary by department",
    "Who is the oldest employee?",
    "How many employees are in each department?",
]
BANNED = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|vacuum|reindex|truncate)\b",
    re.IGNORECASE,
)

st.set_page_config(
    page_title="Text to SQL Assistant",
    page_icon="🗄️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; max-width: 1200px;}
    .hero {padding: 1.5rem 1.8rem; border-radius: 16px; margin-bottom: 1.2rem;
           background: linear-gradient(135deg, #4f46e5 0%, #06b6d4 100%); color: #fff;}
    .hero .title {font-size: 2rem; font-weight: 700; margin: 0;}
    .hero .sub {margin: .3rem 0 0; opacity: .92; font-size: 1.02rem;}
    div[data-testid="stMetric"] {background: rgba(128,128,128,.08); padding: .8rem 1rem;
           border-radius: 12px; border: 1px solid rgba(128,128,128,.18);}
    .stButton > button {border-radius: 10px;}
    footer {visibility: hidden;}
    </style>
    <div class="hero">
        <div class="title">🗄️ Text to SQL Assistant</div>
        <div class="sub">Ask in plain English. Get SQL, a clear explanation and live results.</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Structured output
# --------------------------------------------------------------------------- #
class SQLResponse(BaseModel):
    sql_query: str = Field(description="The SQL query generated for the user's question")
    explanation: str = Field(description="A simple explanation of the SQL query")


class QueryError(Exception):
    """Raised when SQL cannot be validated/executed even after auto-repair."""

    def __init__(self, message: str, sql: str):
        super().__init__(message)
        self.sql = sql


# --------------------------------------------------------------------------- #
# Groq model discovery
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_groq_models(api_key: str) -> list[str]:
    resp = requests.get(
        "https://api.groq.com/openai/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=10,
    )
    resp.raise_for_status()  # errors are not cached
    ids = [m["id"] for m in resp.json().get("data", []) if m.get("active", True)]
    ids = [i for i in ids if not any(h in i.lower() for h in NON_CHAT_HINTS)]
    first = [m for m in PREFERRED_MODELS if m in ids]
    rest = sorted(i for i in ids if i not in PREFERRED_MODELS)
    return first + rest


def get_available_models(api_key: str) -> tuple[list[str], bool]:
    """Return (models, from_api). Falls back to PREFERRED_MODELS if the call fails."""
    try:
        models = _fetch_groq_models(api_key)
        if models:
            return models, True
    except Exception:
        pass
    return PREFERRED_MODELS, False


# --------------------------------------------------------------------------- #
# LangChain: model | prompt | parser
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def build_chains(api_key: str, model_name: str):
    """Build (generate_chain, repair_chain). Cached per key + model."""
    extra = {"reasoning_effort": "low"} if "gpt-oss" in model_name else {}  # faster, enough for SQL
    llm = ChatGroq(
        model=model_name,
        temperature=0,
        api_key=api_key,
        max_retries=1,
        timeout=60,
        **extra,
    )
    parser = PydanticOutputParser(pydantic_object=SQLResponse)
    fmt = parser.get_format_instructions()

    generate_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert SQL developer.

Convert the user's natural language question into a SQLite query.

Rules:
- Use only the tables and columns in the database schema.
- Write exactly ONE read-only SELECT statement (CTEs with WITH are fine).
- For text matching prefer LOWER(col) = LOWER('value') or LIKE.
- Give computed columns clear aliases.
- Keep the explanation short and simple.

{format_instructions}""",
            ),
            ("human", "Database Schema:\n{schema}\n\nUser Question:\n{question}"),
        ]
    ).partial(format_instructions=fmt)

    repair_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert SQL developer. A SQLite query failed.
Fix it so it answers the question. Use only the given schema.
Write exactly ONE read-only SELECT statement.

{format_instructions}""",
            ),
            (
                "human",
                "Database Schema:\n{schema}\n\nUser Question:\n{question}\n\n"
                "Failed SQL:\n{failed_sql}\n\nError:\n{error}",
            ),
        ]
    ).partial(format_instructions=fmt)

    return generate_prompt | llm | parser, repair_prompt | llm | parser


# --------------------------------------------------------------------------- #
# Database helpers
# --------------------------------------------------------------------------- #
def _ro_connect(db_path: str) -> sqlite3.Connection:
    """Read-only SQLite connection (defence in depth on top of validate_sql)."""
    return sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True)


def _sanitize(name: str) -> str:
    name = re.sub(r"\W+", "_", name.strip()).strip("_").lower()
    return f"t_{name}" if (not name or name[0].isdigit()) else name


def load_csv_to_db(file_bytes: bytes, filename: str) -> tuple[str, str]:
    """Store an uploaded CSV in a temp SQLite file. Returns (db_path, table_name)."""
    digest = hashlib.md5(file_bytes).hexdigest()[:10]
    db_path = Path(tempfile.gettempdir()) / f"t2s_{digest}.db"
    table = _sanitize(Path(filename).stem)
    if not db_path.exists():
        df = pd.read_csv(io.BytesIO(file_bytes))
        df.columns = [_sanitize(str(c)) for c in df.columns]
        with closing(sqlite3.connect(db_path)) as conn:
            df.to_sql(table, conn, index=False, if_exists="replace")
    return str(db_path), table


def get_schema(db_path: str) -> str:
    with closing(_ro_connect(db_path)) as conn:
        rows = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
    return "\n\n".join(r[0] for r in rows if r[0])


def list_tables(db_path: str) -> list[str]:
    with closing(_ro_connect(db_path)) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
    return [r[0] for r in rows]


def validate_sql(sql: str) -> tuple[bool, str]:
    """Return (ok, cleaned_sql_or_error_message)."""
    s = sql.strip().rstrip(";").strip()
    if not s:
        return False, "Empty query."
    if ";" in s:
        return False, "Multiple statements are not allowed."
    if not re.match(r"^(select|with)\b", s, re.IGNORECASE):
        return False, "Only SELECT queries are allowed."
    if BANNED.search(s):
        return False, "Query contains a forbidden keyword."
    return True, s


def run_query(db_path: str, sql: str) -> pd.DataFrame:
    with closing(_ro_connect(db_path)) as conn:
        return pd.read_sql_query(sql, conn)


# --------------------------------------------------------------------------- #
# Core pipeline
# --------------------------------------------------------------------------- #
def answer_question(question: str, schema: str, db_path: str, gen_chain, fix_chain) -> dict:
    start = time.perf_counter()
    out = gen_chain.invoke({"schema": schema, "question": question})
    sql, explanation, repaired = out.sql_query, out.explanation, False

    for attempt in range(2):
        ok, cleaned = validate_sql(sql)
        error = None if ok else cleaned
        if ok:
            try:
                df = run_query(db_path, cleaned)
                return {
                    "question": question,
                    "sql": cleaned,
                    "explanation": explanation,
                    "df": df.head(MAX_ROWS),
                    "total_rows": len(df),
                    "seconds": time.perf_counter() - start,
                    "repaired": repaired,
                    "time": datetime.now().strftime("%H:%M:%S"),
                }
            except Exception as exc:  # SQL runtime error
                error = str(exc)

        if attempt == 0:  # one auto-repair attempt
            fixed = fix_chain.invoke(
                {"schema": schema, "question": question, "failed_sql": sql, "error": error}
            )
            sql, explanation, repaired = fixed.sql_query, fixed.explanation, True
        else:
            raise QueryError(error or "Unknown error", sql)

    raise QueryError("Unknown error", sql)  # pragma: no cover


# --------------------------------------------------------------------------- #
# UI helpers
# --------------------------------------------------------------------------- #
def get_api_key() -> str | None:
    try:
        key = st.secrets["GROQ_API_KEY"]
    except Exception:
        key = None
    return key or os.getenv("GROQ_API_KEY")


def set_question(text: str) -> None:
    st.session_state["question"] = text


def render_result(res: dict) -> None:
    df: pd.DataFrame = res["df"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{res['total_rows']:,}")
    c2.metric("Columns", df.shape[1])
    c3.metric("Time", f"{res['seconds']:.1f}s")
    c4.metric("Auto-fixed", "Yes" if res["repaired"] else "No")

    st.subheader("Generated SQL")
    st.code(res["sql"], language="sql")
    st.subheader("Explanation")
    st.info(res["explanation"])

    st.subheader("Result")
    if df.empty:
        st.warning("Query ran successfully but returned no rows.")
        return
    if res["total_rows"] > MAX_ROWS:
        st.caption(f"Showing first {MAX_ROWS:,} of {res['total_rows']:,} rows.")
    st.dataframe(df, hide_index=True)

    left, _ = st.columns([1, 3])
    left.download_button(
        "⬇️ Download CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name="query_result.csv",
        mime="text/csv",
    )

    if df.shape[1] == 2 and pd.api.types.is_numeric_dtype(df.iloc[:, 1]) and 1 < len(df) <= 50:
        with st.expander("📊 Quick chart", expanded=True):
            st.bar_chart(df.set_index(df.columns[0]))


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
def main() -> None:
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("last_result", None)
    st.session_state.setdefault("question", "")

    # ---------------- Sidebar ----------------
    with st.sidebar:
        st.header("⚙️ Settings")

        api_key = get_api_key()
        if api_key:
            st.success("Groq API key loaded", icon="🔑")
        else:
            api_key = st.text_input(
                "Groq API key", type="password", help="Free key at console.groq.com"
            )

        models, from_api = get_available_models(api_key) if api_key else (PREFERRED_MODELS, False)
        model_name = st.selectbox(
            "Model",
            models,
            help="Models your Groq key can use." if from_api else "Could not load your model list; showing defaults.",
        )

        st.divider()
        st.subheader("🗃️ Data source")
        source = st.radio(
            "Choose data", ["Sample employees DB", "Upload my CSV"], label_visibility="collapsed"
        )

        create_db()
        db_path, source_note = str(DB_PATH), "Sample DB: `employees` table"
        if source == "Upload my CSV":
            up = st.file_uploader("CSV file", type="csv")
            if up is not None:
                if up.size > MAX_UPLOAD_MB * 1024 * 1024:
                    st.error(f"File too large (max {MAX_UPLOAD_MB} MB).")
                else:
                    try:
                        db_path, table = load_csv_to_db(up.getvalue(), up.name)
                        source_note = f"Uploaded: `{table}` table"
                    except Exception as exc:
                        st.error(f"Could not read CSV: {exc}")
            else:
                st.caption("Upload a CSV to query it. Using sample DB until then.")
        st.caption(source_note)

        st.divider()
        if st.button("🧹 Clear history"):
            st.session_state["history"] = []
            st.session_state["last_result"] = None
            st.rerun()
        st.caption("Read-only mode: only SELECT queries are executed.")

    if not api_key:
        st.info("👈 Add your Groq API key in the sidebar to get started.")
        st.stop()

    schema = get_schema(db_path)
    gen_chain, fix_chain = build_chains(api_key, model_name)

    # ---------------- Tabs ----------------
    tab_ask, tab_history, tab_schema = st.tabs(["💬 Ask", "🕘 History", "📚 Schema & Data"])

    with tab_ask:
        st.caption("Try an example:")
        cols = st.columns(len(SAMPLE_QUESTIONS))
        for col, q in zip(cols, SAMPLE_QUESTIONS):
            col.button(q, key=f"ex_{q}", on_click=set_question, args=(q,))

        with st.form("ask_form", border=False):
            question = st.text_input(
                "Your question",
                key="question",
                placeholder="e.g. Show employees with salary greater than 50000",
            )
            submitted = st.form_submit_button("✨ Generate & Run", type="primary")

        if submitted:
            if not question.strip():
                st.warning("Please type a question first.")
            else:
                with st.spinner("Writing SQL and running it..."):
                    try:
                        res = answer_question(question.strip(), schema, db_path, gen_chain, fix_chain)
                        st.session_state["last_result"] = res
                        st.session_state["history"].append(res)
                    except QueryError as exc:
                        st.session_state["last_result"] = None
                        st.error(f"Could not produce a working query: {exc}")
                        st.code(exc.sql, language="sql")
                    except Exception as exc:
                        st.session_state["last_result"] = None
                        st.error(
                            f"LLM request failed: {exc}\n\n"
                            "Check your API key, model name and internet connection."
                        )

        if st.session_state["last_result"]:
            render_result(st.session_state["last_result"])

    with tab_history:
        history = st.session_state["history"]
        if not history:
            st.info("No queries yet. Ask something in the Ask tab.")
        for i, item in enumerate(reversed(history)):
            with st.expander(f"{item['time']}  •  {item['question']}"):
                st.code(item["sql"], language="sql")
                st.caption(f"{item['total_rows']:,} rows • {item['seconds']:.1f}s")
                st.button("↩️ Ask again", key=f"again_{i}", on_click=set_question, args=(item["question"],))

    with tab_schema:
        st.subheader("Database schema")
        st.code(schema, language="sql")
        for table in list_tables(db_path):
            with st.expander(f"Preview: {table}"):
                st.dataframe(run_query(db_path, f'SELECT * FROM "{table}" LIMIT 5'), hide_index=True)


main()