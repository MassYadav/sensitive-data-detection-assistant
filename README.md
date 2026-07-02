# 🛡️ Sensitive Data Detection & Compliance Assistant

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://massyadav-sensitive-data-detection-assistant-app-rubqyy.streamlit.app)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/🦜🔗-LangChain-gray.svg)](https://langchain.com/)

An AI-powered document compliance engine built to automatically scan, redact, and summarize sensitive Personal Identifiable Information (PII) across documents, ensuring enterprise data security.

> **Live Prototype URL:** [Access the App Here](https://massyadav-sensitive-data-detection-assistant-app-rubqyy.streamlit.app)

---

## 🛠️ Tech Stack
*   **Frontend/UI:** Streamlit
*   **AI/LLM:** LangChain, Groq API (Llama 3), Google Gemini API
*   **Vector Store:** FAISS (Facebook AI Similarity Search)
*   **Data Processing:** PyMuPDF (PDFs), Pandas (CSVs)
*   **Detection Engine:** Python `re` (Regex)

---

## 1. Architecture Overview
The system is built on a highly modular architecture designed for security and scalability:

*   **Frontend Controller (`app.py`):** A Streamlit-based multi-tab dashboard offering intuitive interfaces for Document Scanning, a Redaction Lab, RAG-powered Q&A, and secure Audit Logs.
*   **Hybrid Detection Engine (`utils/detectors.py`):** The core scanner combining deterministic Regular Expressions for hard PII extraction and entity masking.
*   **RAG Pipeline (`utils/ai_chains.py`):** A LangChain and FAISS-powered semantic search backend that chunks uploaded documents, stores their embeddings in-memory, and retrieves highly relevant context to ground the LLM responses.
*   **Parsers & Utilities (`utils/file_parsers.py`, `utils/logger.py`):** Modular handlers for multi-format text extraction (PDF, CSV, TXT) with OCR fallback, alongside a metadata-only audit logger to track compliance without persisting sensitive data.

---

## 2. AI/ML Approach Used
We implemented a **Hybrid AI Approach** to balance speed, cost, and reliability:

*   **Regex (Deterministic):** Used for fast, deterministic extraction of structured PII (PAN, Aadhaar, Emails, Credit Cards, API Keys). This completely eliminates the risk of AI "hallucinations" on critical identifiers and avoids high token costs for basic scanning.
*   **Generative AI (LLMs):** Leveraged for high-level cognitive tasks, such as synthesizing compliance observations, understanding the document's context, and suggesting actionable remediation steps.
*   **RAG (Retrieval-Augmented Generation):** By chunking large documents and using FAISS for vector similarity search, we overcome the context-window limitations of LLMs. This allows users to ask exact, complex questions about their documents, with the AI providing answers strictly grounded in the retrieved text.

---

## 3. Setup Instructions
To run this project locally, follow these steps:

**1. Clone the repository:**
```bash
git clone https://github.com/MassYadav/sensitive-data-detection-assistant.git
cd sensitive-data-detection-assistant
```

**2. Create and activate a virtual environment:**
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

**4. Set up Environment Variables:**
Create a `.env` file in the root directory and add your API keys:
```env
GROQ_API_KEY=your_groq_api_key_here
# OR
GOOGLE_API_KEY=your_google_api_key_here
```

**5. Run the application:**
```bash
streamlit run app.py
```

---

## 4. Challenges Faced
During cloud deployment on Streamlit Community Cloud, the `tesseract-ocr` and `poppler-utils` Linux system dependencies caused OS-level package conflicts (specifically with the `time_t` transition in Debian libraries) which blocked the build process. 

**Solution:** We implemented a graceful error-handling fallback in the PDF parser (`utils/file_parsers.py`). If a scanned PDF or image is uploaded and the heavy OCR system dependencies are missing in the cloud container, the application catches the `ImportError`/`TesseractNotFoundError`. It then safely alerts the user that advanced OCR is disabled for cloud optimization, gracefully falling back to standard text extraction, entirely preventing a fatal application crash.

---

## 5. Future Improvements
*   **Persistent Vector Storage:** Migrate from in-memory FAISS to a scalable, persistent cloud vector database like Pinecone, Milvus, or Qdrant for cross-session document memory.
*   **Expanded File Support:** Add robust parsing capabilities for `.docx` and `.xlsx` files to cover standard enterprise formats.
*   **Enterprise Security Integrations:** Implement RBAC (Role-Based Access Control) and SSO (Single Sign-On) so different users have distinct permissions (e.g., Auditors vs. Standard Employees).
*   **Custom NLP Models (NER):** Train and deploy a lightweight spaCy or HuggingFace Named Entity Recognition (NER) model to detect unstructured, non-regex PII (like proper names, unique medical conditions, or regional addresses).