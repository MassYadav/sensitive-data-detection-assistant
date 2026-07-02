"""
app.py
------
Main Streamlit application for the Sensitive Data Detection & Compliance
Assistant.  Features:

  - Multi-document upload & processing
  - Tabbed UI: Dashboard · Redaction Lab · Q&A Chat · Audit Logs
  - RAG-powered chat with FAISS
  - Data masking & redacted document download
  - Audit logging

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from utils.file_parsers import parse_uploaded_file
from utils.detectors import scan_text, classify_risk, redact_text, ScanResult
from utils.ai_chains import (
    generate_report,
    ask_question,
    build_faiss_index,
    PROVIDERS,
)
from utils.logger import (
    log_document_upload,
    log_scan_complete,
    log_report_generated,
    log_qa_query,
    log_redaction,
    log_rag_index_built,
    log_error,
    read_recent_logs,
)

load_dotenv()

# ── Page Config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Sensitive Data Detection Assistant",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .risk-high   { background: linear-gradient(135deg, #ff4b4b, #c0392b); color: white;
                   padding: 0.6rem 1.2rem; border-radius: 8px; font-weight: 700;
                   font-size: 1.15rem; text-align: center; }
    .risk-medium { background: linear-gradient(135deg, #f39c12, #e67e22); color: white;
                   padding: 0.6rem 1.2rem; border-radius: 8px; font-weight: 700;
                   font-size: 1.15rem; text-align: center; }
    .risk-low    { background: linear-gradient(135deg, #27ae60, #2ecc71); color: white;
                   padding: 0.6rem 1.2rem; border-radius: 8px; font-weight: 700;
                   font-size: 1.15rem; text-align: center; }

    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 0.8rem;
    }

    .stChatMessage { border-radius: 12px; }
    section[data-testid="stSidebar"] > div:first-child { padding-top: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Session State ───────────────────────────────────────────────────────────

def _init_state() -> None:
    defaults = {
        "documents": {},          # filename → extracted text
        "scan_results": {},       # filename → ScanResult
        "risk_levels": {},        # filename → risk string
        "ai_reports": {},         # filename → report markdown
        "chat_history": [],       # list of (user, ai) tuples
        "vector_store": None,     # FAISS index
        "rag_built": False,
        "processed_names": set(), # set of already-processed filenames
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

_init_state()


# ── Helpers ─────────────────────────────────────────────────────────────────

def _mask(value: str) -> str:
    """Partially mask a sensitive value for safe display."""
    v = value.strip()
    if len(v) <= 6:
        return v[:2] + "•" * (len(v) - 2)
    return v[:3] + "•" * (len(v) - 6) + v[-3:]


def _risk_css(risk: str) -> str:
    return {
        "🔴 High Risk": "risk-high",
        "🟡 Medium Risk": "risk-medium",
        "🟢 Low Risk": "risk-low",
    }.get(risk, "risk-low")


def _aggregate_counts() -> dict[str, int]:
    """Sum detection counts across all scanned documents."""
    totals: dict[str, int] = {}
    for result in st.session_state.scan_results.values():
        for cat, cnt in result.counts().items():
            totals[cat] = totals.get(cat, 0) + cnt
    return totals


def _worst_risk() -> str:
    """Return the highest risk level across all documents."""
    levels = list(st.session_state.risk_levels.values())
    if not levels:
        return "🟢 Low Risk"
    if any("High" in l for l in levels):
        return "🔴 High Risk"
    if any("Medium" in l for l in levels):
        return "🟡 Medium Risk"
    return "🟢 Low Risk"


def _combined_text() -> str:
    """Concatenate all document texts."""
    return "\n\n".join(st.session_state.documents.values())


def _combined_scan_summary() -> str:
    """Build a formatted scan summary across all documents."""
    totals = _aggregate_counts()
    return "\n".join(f"- {cat}: {cnt}" for cat, cnt in totals.items())


# ── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🛡️ Compliance Assistant")
    st.markdown("---")

    uploaded_files = st.file_uploader(
        "📂 Upload Document(s)",
        type=["pdf", "txt", "csv"],
        accept_multiple_files=True,
        help="Supported: PDF, TXT, CSV — upload multiple files at once",
    )

    # Process newly uploaded files
    if uploaded_files:
        for uf in uploaded_files:
            if uf.name not in st.session_state.processed_names:
                with st.spinner(f"Parsing {uf.name}…"):
                    try:
                        text = parse_uploaded_file(uf)
                        st.session_state.documents[uf.name] = text
                        log_document_upload(uf.name, len(text) / 1024)

                        # Scan
                        result = scan_text(text)
                        risk = classify_risk(result)
                        st.session_state.scan_results[uf.name] = result
                        st.session_state.risk_levels[uf.name] = risk
                        log_scan_complete(uf.name, result.total_findings, risk)

                        st.session_state.processed_names.add(uf.name)

                        # Reset RAG & reports on new docs
                        st.session_state.vector_store = None
                        st.session_state.rag_built = False
                        st.session_state.chat_history = []

                        st.success(f"✅ {uf.name}")
                    except ValueError as exc:
                        log_error("parse", str(exc))
                        st.error(f"⚠️ {uf.name}: {exc}")

    st.markdown("---")

    # LLM Provider Selector
    provider_keys = list(PROVIDERS.keys())
    provider_labels = [PROVIDERS[k]["label"] for k in provider_keys]
    selected_idx = st.selectbox(
        "🤖 LLM Provider",
        range(len(provider_keys)),
        format_func=lambda i: provider_labels[i],
        help="Pick the AI model for reports & chat. Groq and OpenRouter are free.",
    )
    selected_provider = provider_keys[selected_idx]

    st.markdown("---")
    st.caption("Built with Streamlit · LangChain · FAISS · Multi-LLM")


# ── Header ──────────────────────────────────────────────────────────────────

st.markdown("# 🛡️ Sensitive Data Detection & Compliance Assistant")
st.markdown(
    "Upload documents → detect PII → classify risk → get AI compliance guidance."
)
st.markdown("---")

# Guard
if not st.session_state.documents:
    st.info("👈 Upload one or more **PDF**, **TXT**, or **CSV** files from the sidebar.")
    st.stop()


# ── Tabs ────────────────────────────────────────────────────────────────────

tab_dash, tab_redact, tab_chat, tab_logs = st.tabs(
    ["📊 Dashboard", "🔒 Redaction Lab", "💬 Q&A Chat", "📝 Audit Logs"]
)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

with tab_dash:
    # ── Summary Metrics ─────────────────────────────────────────────────
    agg_counts = _aggregate_counts()
    total_findings = sum(agg_counts.values())
    overall_risk = _worst_risk()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("📁 Files Processed", len(st.session_state.documents))
    with c2:
        st.metric("🔍 Total Sensitive Hits", total_findings)
    with c3:
        st.markdown(
            f'<div class="{_risk_css(overall_risk)}">{overall_risk}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("### 📊 Detection Breakdown")

    # Bar chart
    if agg_counts and total_findings > 0:
        chart_df = pd.DataFrame(
            {"Category": list(agg_counts.keys()), "Count": list(agg_counts.values())}
        ).set_index("Category")
        st.bar_chart(chart_df)
    else:
        st.info("No sensitive data detected across the uploaded documents.")

    # Per-document detail
    st.markdown("### 📄 Per-Document Results")
    for fname, result in st.session_state.scan_results.items():
        risk = st.session_state.risk_levels[fname]
        with st.expander(f"**{fname}**  —  {risk}  ({result.total_findings} findings)"):
            cols = st.columns(4)
            for idx, (cat, cnt) in enumerate(result.counts().items()):
                with cols[idx % 4]:
                    st.metric(cat, cnt)

            # Matched snippets
            for cat, items in [
                ("Aadhaar Numbers", result.aadhaar),
                ("PAN Numbers", result.pan),
                ("Email Addresses", result.emails),
                ("Phone Numbers", result.phones),
                ("Credit Card Numbers", result.credit_cards),
                ("API Keys / Tokens", result.api_keys),
                ("Passwords", result.passwords),
            ]:
                if items:
                    st.markdown(f"**{cat}** ({len(items)})")
                    st.code("\n".join(_mask(s) for s in items), language="text")

    # ── AI Compliance Report ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🤖 AI Compliance Report")

    # Show existing reports
    for fname, report in st.session_state.ai_reports.items():
        with st.expander(f"Report: {fname}", expanded=True):
            st.markdown(report)

    # Generate button
    unreported = [
        f for f in st.session_state.documents if f not in st.session_state.ai_reports
    ]
    if unreported:
        if st.button("🚀 Generate AI Report(s)", type="primary", use_container_width=True):
            has_error = False
            for fname in unreported:
                text = st.session_state.documents[fname]
                result = st.session_state.scan_results[fname]
                risk = st.session_state.risk_levels[fname]
                summary = "\n".join(f"- {c}: {n}" for c, n in result.counts().items())

                with st.spinner(f"Generating report for {fname}…"):
                    try:
                        report = generate_report(
                            document_text=text,
                            scan_summary=summary,
                            risk_level=risk,
                            provider=selected_provider,
                        )
                        st.session_state.ai_reports[fname] = report
                        log_report_generated(fname, selected_provider)
                    except Exception as exc:
                        log_error("report_generation", str(exc))
                        st.error(f"⚠️ {fname}: {exc}")
                        has_error = True
            
            if not has_error:
                st.rerun()
    else:
        if st.session_state.ai_reports:
            if st.button("🔄 Regenerate All Reports"):
                st.session_state.ai_reports = {}
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — REDACTION LAB
# ══════════════════════════════════════════════════════════════════════════════

with tab_redact:
    st.markdown("### 🔒 Data Masking & Redaction")
    st.markdown(
        "Generate a redacted version of your documents with all sensitive data "
        "replaced by `[REDACTED - TYPE]` tags."
    )

    for fname, text in st.session_state.documents.items():
        with st.expander(f"📄 {fname}", expanded=True):
            result = st.session_state.scan_results.get(fname)

            if result and result.total_findings > 0:
                redacted, count = redact_text(text, result)
                log_redaction(fname, count)

                st.success(f"✅ {count} sensitive items redacted.")

                # Preview
                st.text_area(
                    "Redacted Preview",
                    redacted[:3000] + ("\n\n[… truncated for preview …]" if len(redacted) > 3000 else ""),
                    height=300,
                    key=f"redact_preview_{fname}",
                )

                # Download
                st.download_button(
                    label=f"⬇️ Download Redacted — {fname.rsplit('.', 1)[0]}.txt",
                    data=redacted,
                    file_name=f"redacted_{fname.rsplit('.', 1)[0]}.txt",
                    mime="text/plain",
                    key=f"download_{fname}",
                )
            else:
                st.info("No sensitive data detected — no redaction needed.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Q&A CHAT (RAG)
# ══════════════════════════════════════════════════════════════════════════════

with tab_chat:
    st.markdown("### 💬 Ask Questions About Your Documents")
    st.markdown("*Powered by RAG — your documents are chunked and indexed for precise answers.*")

    # Build FAISS index on demand
    if not st.session_state.rag_built:
        if st.button("🔗 Build RAG Index", type="primary"):
            combined = _combined_text()
            with st.spinner("Chunking & embedding documents for RAG…"):
                try:
                    vs, num_chunks = build_faiss_index(combined, provider=selected_provider)
                    st.session_state.vector_store = vs
                    st.session_state.rag_built = True
                    log_rag_index_built(num_chunks)
                    st.success(f"✅ FAISS index built with {num_chunks} chunks.")
                    st.rerun()
                except Exception as exc:
                    log_error("rag_build", str(exc))
                    st.error(f"⚠️ Failed to build RAG index: {exc}")
    else:
        st.success("✅ RAG index is ready. Ask away!")

        # Display chat history
        for user_msg, ai_msg in st.session_state.chat_history:
            with st.chat_message("user"):
                st.markdown(user_msg)
            with st.chat_message("assistant"):
                st.markdown(ai_msg)

        # Chat input
        if prompt := st.chat_input("Ask a question about your documents…"):
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    try:
                        log_qa_query(prompt, selected_provider)
                        answer = ask_question(
                            question=prompt,
                            document_text=_combined_text(),
                            scan_summary=_combined_scan_summary(),
                            risk_level=_worst_risk(),
                            provider=selected_provider,
                            chat_history=st.session_state.chat_history,
                            vector_store=st.session_state.vector_store,
                        )
                        st.markdown(answer)
                        st.session_state.chat_history.append((prompt, answer))
                    except Exception as exc:
                        log_error("qa_query", str(exc))
                        st.error(f"⚠️ {exc}")

        if st.session_state.chat_history:
            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — AUDIT LOGS
# ══════════════════════════════════════════════════════════════════════════════

with tab_logs:
    st.markdown("### 📝 Audit Log")
    st.markdown("All actions are logged to `audit.log`. Sensitive data values are **never** recorded.")

    logs = read_recent_logs(max_lines=200)
    st.code(logs, language="log")

    if st.button("🔄 Refresh Logs"):
        st.rerun()
