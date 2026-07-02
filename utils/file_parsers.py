"""
utils/file_parsers.py
---------------------
Functions to extract raw text from uploaded PDF, TXT, and CSV files.
Uses PyMuPDF (fitz) for PDF parsing, pandas for CSV, and falls back
to OCR (pytesseract + pdf2image) for scanned / image-only PDFs.
"""

from __future__ import annotations

import io
from typing import Optional

import fitz  # PyMuPDF
import pandas as pd
import streamlit as st


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract all text from a PDF file.

    Tries native text extraction first.  If the PDF contains no extractable
    text (e.g. scanned documents), falls back to OCR via pytesseract.

    Args:
        file_bytes: Raw bytes of the uploaded PDF.

    Returns:
        The concatenated text of every page, separated by newlines.

    Raises:
        ValueError: If the PDF cannot be opened or text cannot be extracted.
    """
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as exc:
        raise ValueError(f"Could not open the PDF file: {exc}") from exc

    pages: list[str] = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        if text.strip():
            pages.append(f"--- Page {page_num} ---\n{text}")

    doc.close()

    # If native extraction found text, return it
    if pages:
        return "\n\n".join(pages)

    # ── OCR Fallback ────────────────────────────────────────────────────
    return _ocr_pdf(file_bytes)


def _ocr_pdf(file_bytes: bytes) -> str:
    """Extract text from a scanned PDF using OCR.

    Requires ``pytesseract`` and ``pdf2image`` (plus system-level
    ``tesseract-ocr`` and ``poppler-utils``).

    Args:
        file_bytes: Raw bytes of the PDF.

    Returns:
        OCR-extracted text.

    Raises:
        ValueError: If OCR dependencies are missing or extraction fails.
    """
    try:
        from pdf2image import convert_from_bytes
        import pytesseract
    except ImportError:
        raise ValueError(
            "The PDF appears to be scanned / image-only but OCR dependencies "
            "are not installed.  Install 'pytesseract' and 'pdf2image', and "
            "ensure 'tesseract-ocr' and 'poppler-utils' are available on "
            "your system."
        )

    try:
        images = convert_from_bytes(file_bytes, dpi=200)
    except Exception as exc:
        raise ValueError(
            f"Could not convert PDF pages to images for OCR: {exc}"
        ) from exc

    pages: list[str] = []
    for page_num, img in enumerate(images, start=1):
        text = pytesseract.image_to_string(img)
        if text.strip():
            pages.append(f"--- Page {page_num} (OCR) ---\n{text}")

    if not pages:
        raise ValueError(
            "OCR could not extract any text from this PDF. "
            "The document may be empty or heavily degraded."
        )
    return "\n\n".join(pages)


def extract_text_from_txt(file_bytes: bytes) -> str:
    """Decode and return the contents of a plain-text file.

    Tries UTF-8 first, then falls back to latin-1.
    """
    for encoding in ("utf-8", "latin-1"):
        try:
            return file_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("Unable to decode the text file with UTF-8 or Latin-1 encoding.")


def extract_text_from_csv(file_bytes: bytes) -> str:
    """Convert a CSV file to a text representation.

    Each row is rendered as a single line with columns joined by `` | ``.
    """
    try:
        df = pd.read_csv(io.BytesIO(file_bytes))
    except Exception as exc:
        raise ValueError(f"Could not parse the CSV file: {exc}") from exc

    if df.empty:
        raise ValueError("The CSV file is empty.")

    header = " | ".join(str(c) for c in df.columns)
    rows = [" | ".join(str(v) for v in row) for row in df.values]
    return header + "\n" + "\n".join(rows)


def parse_uploaded_file(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> str:
    """Dispatch to the correct parser based on file extension.

    Args:
        uploaded_file: The Streamlit UploadedFile object.

    Returns:
        Extracted text content.

    Raises:
        ValueError: On unsupported file types or parsing errors.
    """
    name: str = uploaded_file.name.lower()
    raw: bytes = uploaded_file.read()

    if name.endswith(".pdf"):
        return extract_text_from_pdf(raw)
    elif name.endswith(".txt"):
        return extract_text_from_txt(raw)
    elif name.endswith(".csv"):
        return extract_text_from_csv(raw)
    else:
        raise ValueError(
            f"Unsupported file type: '{name.split('.')[-1]}'. "
            "Please upload a PDF, TXT, or CSV file."
        )
