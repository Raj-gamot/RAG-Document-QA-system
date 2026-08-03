# 📄 RAG Document Q&A System

An AI-powered Retrieval-Augmented Generation (RAG) application that allows users to upload PDF/DOCX documents and ask natural language questions about them.

## 🚀 Live Demo
Frontend:https://rag-document-ui.onrender.com/

Backend API:https://rag-document-qa-system-r5nv.onrender.com/docs#/default/upload_file_upload_post

## Features

- Upload documents: **PDF, DOCX, PPTX**
- Optional **image OCR** (PNG/JPG) on Python ≤ 3.12 when Tesseract is installed
- Automatic text extraction and chunking
- Embeddings via **Mistral AI**
- Persistent vector storage with **ChromaDB**
- Semantic search + context-aware answers from **Mistral LLM**
- REST API (FastAPI) + interactive UI (Streamlit)

## Architecture

```text
User
 ├── Uploads Document
 │
 ├── FastAPI Backend
 │     ├── Text Extraction
 │     ├── Chunking
 │     ├── Embedding (Mistral)
 │     └── Storage (ChromaDB)
 │
 ├── Query
 │
 └── Retrieval + Answer Generation (Mistral LLM)
```

## Project Structure

```text
mistral-rag-system/
├── app.py              # Streamlit frontend
├── main.py             # FastAPI backend
├── utils.py            # RAG utilities (embedding, storage, retrieval)
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
├── uploads/            # Temporary uploaded files
└── chroma_db/          # Persistent ChromaDB storage
```

## Python Version Notes

| Version | Support |
|---------|---------|
| **3.12 or lower** | Full support including optional image OCR |
| **3.13** | PDF, DOCX, PPTX supported; image OCR disabled by default |

## Tech Stack

- **FastAPI** — Backend API
- **Streamlit** — Frontend UI
- **ChromaDB** — Vector database
- **LangChain** — RAG orchestration
- **Mistral AI** — Embeddings + LLM

## Quick Start

### 1. Install Dependencies

```bash
cd mistral-rag-system-main
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure Environment Variables

```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

Edit `.env` and set your key:

```env
MISTRAL_API_KEY=your_mistral_api_key_here
FASTAPI_URL=http://localhost:8000
```

Get an API key from: https://console.mistral.ai

### 3. Run the Backend (FastAPI)

```bash
uvicorn main:app --reload
```

API: http://localhost:8000  
Docs: http://localhost:8000/docs

### 4. Run the Frontend (Streamlit)

In a second terminal (with the same venv activated):

```bash
streamlit run app.py
```

UI: http://localhost:8501

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/supported-formats` | List uploadable file types |
| `POST` | `/upload` | Upload and index a document |
| `POST` | `/query` | Ask a question (`{"question": "..."}`) |
| `DELETE` | `/documents` | Clear the knowledge base |

## Usage

1. Open the Streamlit UI and confirm **API Status** shows Connected.
2. Upload a PDF, DOCX, or PPTX and click **Process Document**.
3. Ask questions in the chat; answers use only retrieved document context.
