"""
utils/logger.py
---------------
Audit logging for the Sensitive Data Detection & Compliance Assistant.

Logs every major action (document uploads, scans, risk classifications,
AI queries) to a local ``audit.log`` file.  Sensitive data values are
**never** logged — only metadata and action descriptions.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime

# ── Logger Setup ────────────────────────────────────────────────────────────

_LOG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOG_FILE = os.path.join(_LOG_DIR, "audit.log")

# Create a dedicated logger (not the root logger)
audit_logger = logging.getLogger("compliance_audit")
audit_logger.setLevel(logging.INFO)

# Prevent duplicate handlers on Streamlit reruns
if not audit_logger.handlers:
    # File handler — appends to audit.log
    file_handler = logging.FileHandler(_LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.INFO)

    # Formatter with timestamp, level, and message
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    audit_logger.addHandler(file_handler)


# ── Public API ──────────────────────────────────────────────────────────────

def log_document_upload(filename: str, file_size_kb: float) -> None:
    """Log a document upload event."""
    audit_logger.info(
        "DOCUMENT_UPLOAD | file=%s | size=%.1f KB", filename, file_size_kb
    )


def log_scan_complete(filename: str, total_findings: int, risk_level: str) -> None:
    """Log the completion of a sensitive-data scan."""
    audit_logger.info(
        "SCAN_COMPLETE | file=%s | findings=%d | risk=%s",
        filename, total_findings, risk_level,
    )


def log_report_generated(filename: str, provider: str) -> None:
    """Log AI compliance report generation."""
    audit_logger.info(
        "REPORT_GENERATED | file=%s | provider=%s", filename, provider
    )


def log_qa_query(question_preview: str, provider: str) -> None:
    """Log a Q&A chat query (first 80 chars only, no sensitive data)."""
    safe_preview = question_preview[:80].replace("\n", " ")
    audit_logger.info(
        "QA_QUERY | provider=%s | question=%s", provider, safe_preview
    )


def log_redaction(filename: str, redacted_count: int) -> None:
    """Log a document redaction action."""
    audit_logger.info(
        "REDACTION | file=%s | items_redacted=%d", filename, redacted_count
    )


def log_rag_index_built(num_chunks: int) -> None:
    """Log FAISS index creation."""
    audit_logger.info(
        "RAG_INDEX_BUILT | chunks=%d", num_chunks
    )


def log_error(action: str, error_msg: str) -> None:
    """Log an error event."""
    audit_logger.error(
        "ERROR | action=%s | error=%s", action, error_msg[:200]
    )


def get_log_path() -> str:
    """Return the absolute path to the audit log file."""
    return _LOG_FILE


def read_recent_logs(max_lines: int = 100) -> str:
    """Read the most recent log entries.

    Args:
        max_lines: Maximum number of lines to return from the end of the file.

    Returns:
        The last *max_lines* of the audit log, or a message if the log is empty.
    """
    if not os.path.exists(_LOG_FILE):
        return "No audit logs yet."
    try:
        with open(_LOG_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if not lines:
            return "No audit logs yet."
        return "".join(lines[-max_lines:])
    except Exception:
        return "Could not read audit log."
