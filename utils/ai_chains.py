"""
utils/ai_chains.py
------------------
LangChain-powered AI chains for:
  1. Generating a structured compliance report from scan results.
  2. RAG-powered Q&A — chunks documents, builds a FAISS index, and retrieves
     the top-k most relevant chunks to answer user questions.

Supports four LLM backends selectable at runtime:
  - Google Gemini  (gemini-2.0-flash-lite)
  - OpenAI         (gpt-4o-mini)
  - Groq           (llama-3.1-8b-instant — free)
  - OpenRouter     (llama-3.1-8b-instruct:free — free)
"""

from __future__ import annotations

import os
from typing import List, Tuple, Optional

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.language_models import BaseChatModel
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


# ── Provider Definitions ────────────────────────────────────────────────────

PROVIDERS = {
    "google": {
        "label": "Google Gemini  (gemini-2.0-flash-lite)",
        "model": "gemini-2.0-flash-lite",
        "key_env": "GOOGLE_API_KEY",
    },
    "openai": {
        "label": "OpenAI  (gpt-4o-mini)",
        "model": "gpt-4o-mini",
        "key_env": "OPENAI_API_KEY",
    },
    "groq": {
        "label": "Groq  (llama-3.1-8b-instant)  — FREE",
        "model": "llama-3.1-8b-instant",
        "key_env": "GROQ_API_KEY",
    },
    "openrouter": {
        "label": "OpenRouter  (llama-3.1-8b  free)  — FREE",
        "model": "meta-llama/llama-3.1-8b-instruct:free",
        "key_env": "OPENROUTER_API_KEY",
    },
}


# ── LLM Factory ─────────────────────────────────────────────────────────────

def _get_api_key(provider: str) -> str:
    """Retrieve and validate the API key for a provider.

    Checks ``os.environ`` first (for local ``.env``), then falls back to
    ``st.secrets`` (for Streamlit Community Cloud deployment).
    """
    import streamlit as st

    info = PROVIDERS.get(provider)
    if info is None:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            f"Choose from: {', '.join(PROVIDERS)}"
        )

    key_name = info["key_env"]

    # 1. Try os.environ (loaded from .env locally)
    api_key = os.getenv(key_name, "")

    # 2. Fallback to st.secrets (Streamlit Cloud)
    if not api_key or api_key.startswith("your-"):
        try:
            api_key = st.secrets.get(key_name, "")
        except Exception:
            api_key = ""

    if not api_key or api_key.startswith("your-"):
        raise ValueError(
            f"{key_name} is not set. "
            f"Add it to your .env file or Streamlit Cloud secrets."
        )
    return api_key


def get_llm(provider: str = "google") -> BaseChatModel:
    """Instantiate the configured LLM backend.

    Args:
        provider: One of ``"google"``, ``"openai"``, ``"groq"``, ``"openrouter"``.

    Returns:
        A LangChain chat model instance.
    """
    provider = provider.strip().lower()
    api_key = _get_api_key(provider)
    model_name = PROVIDERS[provider]["model"]

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.3,
            convert_system_message_to_human=True,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model_name, api_key=api_key, temperature=0.3)

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=model_name, groq_api_key=api_key, temperature=0.3)

    if provider == "openrouter":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0.3,
            default_headers={
                "HTTP-Referer": "http://localhost:8501",
                "X-Title": "Sensitive Data Detection Assistant",
            },
        )

    raise ValueError(f"Unhandled provider: {provider}")


# ── RAG: FAISS Vector Store ─────────────────────────────────────────────────

def _get_embeddings(provider: str):
    """Return a LangChain embeddings instance for the given provider.

    Falls back to a simple approach for providers that don't have native
    embeddings (Groq, OpenRouter) by using HuggingFace sentence-transformers
    via FAISS-compatible fake embeddings, or Google/OpenAI embeddings.
    """
    api_key_val = _get_api_key(provider)

    if provider == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=api_key_val,
        )

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(api_key=api_key_val, model="text-embedding-3-small")

    # For Groq and OpenRouter, use Google embeddings if available, else OpenAI
    import streamlit as st

    try:
        google_key = _get_api_key("google")
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=google_key,
        )
    except Exception:
        pass

    try:
        openai_key = _get_api_key("openai")
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(api_key=openai_key, model="text-embedding-3-small")
    except Exception:
        pass

    raise ValueError(
        "Groq/OpenRouter don't provide embeddings natively here. "
        "Please set GOOGLE_API_KEY or OPENAI_API_KEY in .env (or Streamlit secrets) for RAG embeddings."
    )


def build_faiss_index(text: str, provider: str = "google"):
    """Chunk the text and build a FAISS vector store.

    Args:
        text: The full document text.
        provider: LLM provider (used to pick the embeddings model).

    Returns:
        A FAISS vector store instance.
    """
    from langchain_community.vectorstores import FAISS

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = splitter.create_documents([text])

    embeddings = _get_embeddings(provider)
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store, len(chunks)


def retrieve_context(vector_store, query: str, top_k: int = 3) -> str:
    """Retrieve the top-k most relevant chunks for a query.

    Args:
        vector_store: The FAISS vector store.
        query: The user's question.
        top_k: Number of chunks to retrieve.

    Returns:
        A formatted string with the retrieved context chunks.
    """
    docs = vector_store.similarity_search(query, k=top_k)
    context_parts = []
    for i, doc in enumerate(docs, 1):
        context_parts.append(f"[Chunk {i}]\n{doc.page_content}")
    return "\n\n".join(context_parts)


# ── Compliance Report Chain ─────────────────────────────────────────────────

_REPORT_SYSTEM = """\
You are a senior data-privacy and compliance analyst. You have been given:
1. The extracted text of a document.
2. A structured scan report listing every category of sensitive data found
   and the number of occurrences.
3. The overall risk classification.

Produce a **Compliance Analysis Report** in Markdown with exactly the
following three sections:

### 📋 Compliance Observations
Summarize what types of sensitive data were found, which regulations they
relate to (e.g. IT Act 2000, GDPR, PCI-DSS, DPDPA 2023), and whether the
document appears to handle them appropriately.

### ⚠️ Security Risks
Describe the concrete security risks posed by the data found (e.g., identity
theft, financial fraud, credential leakage).

### ✅ Remediation Steps
Give numbered, actionable steps to mitigate the identified risks (e.g.,
redact PII before sharing, rotate exposed API keys, encrypt at rest).

Keep the report concise but thorough (300-500 words).
"""

_REPORT_HUMAN = """\
**Document text (truncated to first 8 000 characters):**
```
{document_text}
```

**Scan results:**
{scan_summary}

**Risk classification:** {risk_level}
"""


def generate_report(
    document_text: str,
    scan_summary: str,
    risk_level: str,
    provider: str = "google",
) -> str:
    """Generate a structured compliance report via the LLM."""
    llm = get_llm(provider)
    prompt = ChatPromptTemplate.from_messages([
        ("system", _REPORT_SYSTEM),
        ("human", _REPORT_HUMAN),
    ])
    chain = prompt | llm
    truncated = document_text[:8_000]
    response = chain.invoke({
        "document_text": truncated,
        "scan_summary": scan_summary,
        "risk_level": risk_level,
    })
    return response.content


# ── RAG Q&A Chat Chain ──────────────────────────────────────────────────────

_QA_SYSTEM = """\
You are a helpful compliance assistant. You have access to retrieved context
chunks from the user's document and the sensitive-data scan results.

Answer the user's questions accurately and concisely using the provided context.
If the answer is not in the context or scan results, say so honestly.
Always cite which chunk(s) informed your answer when relevant.

**Retrieved Context:**
{rag_context}

**Scan results:**
{scan_summary}

**Risk classification:** {risk_level}
"""


def ask_question(
    question: str,
    document_text: str,
    scan_summary: str,
    risk_level: str,
    provider: str = "google",
    chat_history: List[Tuple[str, str]] | None = None,
    vector_store=None,
) -> str:
    """Answer a user question using RAG if a vector store is available,
    otherwise fall back to truncated document context.

    Args:
        question: The user's question.
        document_text: Full extracted document text.
        scan_summary: Formatted detection counts.
        risk_level: Risk classification string.
        provider: LLM provider key.
        chat_history: List of (user_msg, ai_msg) tuples.
        vector_store: Optional FAISS vector store for RAG retrieval.

    Returns:
        The AI's answer as a string.
    """
    llm = get_llm(provider)

    # Use RAG retrieval if available, else truncated text
    if vector_store is not None:
        rag_context = retrieve_context(vector_store, question, top_k=3)
    else:
        rag_context = document_text[:8_000]

    prompt = ChatPromptTemplate.from_messages([
        ("system", _QA_SYSTEM),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])
    chain = prompt | llm

    history_msgs = []
    if chat_history:
        for human_text, ai_text in chat_history:
            history_msgs.append(HumanMessage(content=human_text))
            history_msgs.append(AIMessage(content=ai_text))

    response = chain.invoke({
        "rag_context": rag_context,
        "scan_summary": scan_summary,
        "risk_level": risk_level,
        "history": history_msgs,
        "question": question,
    })
    return response.content
