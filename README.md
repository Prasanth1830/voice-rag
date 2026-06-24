# 🧠 RAG Voice Boilerplate

> A production-ready Python + React boilerplate for building **Retrieval-Augmented Generation (RAG)** applications with **cross-encoder re-ranking**, **voice query support**, and a stunning **Admin Dashboard**.

[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=flat-square&logo=react)](https://react.dev)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📚 **RAG Pipeline** | Full retrieve-then-generate pipeline powered by OpenAI embeddings + GPT |
| ⚡ **Cross-Encoder Re-ranking** | Boosts retrieval quality by reranking top-K vector results using a cross-encoder model before passing them to the LLM |
| 🎤 **Voice Queries** | Upload audio and get RAG answers via OpenAI Whisper transcription |
| 🗄️ **Dual Vector Store** | ChromaDB (local, no API key) or Pinecone (cloud) — switchable via env var |
| 🖥️ **Admin Dashboard** | React/Vite UI to upload docs, browse indexed chunks, and interactively test queries |
| 🐋 **Docker Ready** | One-command deployment with Docker Compose for both backend and dashboard |
| 🧪 **Tests Included** | pytest unit tests for the reranker and document processor |

---

## 🏗️ Project Structure

```
voice-rag/
│
├── main.py                        # FastAPI entry point
├── config.py                      # Pydantic-based settings (reads from .env)
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment variable template
│
├── api/
│   └── routes.py                  # All REST API endpoints
│
├── core/
│   ├── rag_engine.py              # Orchestrates the full RAG pipeline
│   ├── reranker.py                # ⚡ Cross-encoder re-ranking (Cohere/local/none)
│   ├── document_processor.py     # PDF/TXT/DOCX parsing + chunking
│   └── voice_processor.py        # Whisper speech-to-text
│
├── database/
│   └── vector_store.py            # ChromaDB & Pinecone abstraction
│
├── utils/
│   └── helpers.py                 # Shared utilities (hashing, text, prompts)
│
├── tests/
│   ├── test_reranker.py           # Reranker unit tests
│   └── test_document_processor.py # Document processor tests
│
├── dashboard/                     # 🖥️ React Admin Dashboard (Vite)
│   ├── src/
│   │   ├── App.jsx                # Main app with tabs + stats
│   │   ├── index.css              # Dark glassmorphism design system
│   │   └── components/
│   │       ├── UploadDocs.jsx     # Drag-and-drop document upload
│   │       ├── ChunksViewer.jsx   # Browse & delete indexed chunks
│   │       └── QueryTester.jsx    # Interactive query + reranking comparison
│   └── vite.config.js
│
└── docker/
    ├── Dockerfile
    └── docker-compose.yaml
```

---

## ⚡ Re-ranking Deep Dive

After a vector similarity search returns the top-K candidates, the raw ranking is based only on **cosine distance** — a relatively shallow metric. A cross-encoder re-ranker reads each `(query, chunk)` pair together and produces a much more accurate **relevance score**.

```
Query ──► Embed ──► Vector Search (top-20) ──► Re-ranker ──► Top-5 ──► LLM
                         ↑                          ↑
                    fast / approximate         slower / precise
```

### Available Re-ranking Modes

| Mode | Description | Requirements |
|------|-------------|-------------|
| `local` | `cross-encoder/ms-marco-MiniLM-L-6-v2` via sentence-transformers | None (auto-downloads ~86 MB) |
| `cohere` | Cohere Rerank API (`rerank-english-v3.0`) | `COHERE_API_KEY` |
| `none` | Skip reranking, use raw vector results | None |

Set via `RERANKER_MODE` in your `.env`.

**Graceful fallback**: If `RERANKER_MODE=cohere` but `COHERE_API_KEY` is missing, the system automatically falls back to `local` mode.

---

## 🖥️ Admin Dashboard

A dark-themed React dashboard running on `localhost:5173`:

| Tab | What it does |
|-----|-------------|
| **📤 Upload Docs** | Drag & drop PDF/TXT/DOCX files. Sequential upload queue with progress bars. |
| **📦 Indexed Chunks** | Browse all stored chunks grouped by source. Search, expand text, delete chunks. |
| **🔍 Query Tester** | Type a query and see: raw vector results vs. reranked results side-by-side, plus the final LLM answer with token usage. |

The dashboard proxies all `/api/*` calls to the FastAPI backend, so no CORS issues in development.

---

## 🚀 Quick Start

### 1. Clone & configure

```bash
git clone https://github.com/Prasanth1830/rag-voice-boilerplate.git
cd rag-voice-boilerplate
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY at minimum
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the API

```bash
uvicorn main:app --reload
# → http://localhost:8000
# → http://localhost:8000/docs  (interactive Swagger UI)
```

### 4. Start the Dashboard

```bash
cd dashboard
npm install
npm run dev
# → http://localhost:5173
```

### 5. (Optional) Docker Compose

```bash
cd docker
docker-compose up -d
# API:       http://localhost:8000
# Dashboard: http://localhost:5173
```

---

## 🌐 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Service health + configuration |
| `GET` | `/api/stats` | Index statistics (chunk count, models) |
| `POST` | `/api/upload` | Upload a document (multipart/form-data) |
| `GET` | `/api/chunks` | List all chunks (`?limit=50&offset=0`) |
| `DELETE` | `/api/chunks/{id}` | Delete a chunk by ID |
| `DELETE` | `/api/documents/{source}` | Delete all chunks for a document |
| `POST` | `/api/query` | Run a RAG text query |
| `POST` | `/api/voice-query` | Transcribe audio + run RAG query |

Full interactive docs: **`http://localhost:8000/docs`**

### Example: Text Query

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the main findings?"}'
```

Response includes:
- `answer` — LLM-generated answer
- `vector_results` — raw top-K from vector search
- `reranked_results` — top-N after cross-encoder reranking (with both `vector_score` and `rerank_score`)
- `reranker_mode`, `tokens_used`, `model`

### Example: Document Upload

```bash
curl -X POST http://localhost:8000/api/upload \
  -F "file=@report.pdf"
```

---

## ⚙️ Configuration

All settings are driven by environment variables (see `.env.example`):

```env
# Required
OPENAI_API_KEY=sk-...

# Vector store: "chroma" (local) or "pinecone" (cloud)
VECTOR_STORE=chroma

# Re-ranker: "local" | "cohere" | "none"
RERANKER_MODE=local

# Retrieval tuning
RETRIEVAL_TOP_K=20    # chunks fetched from vector store
RERANKER_TOP_N=5      # chunks passed to LLM after reranking

# Chunking
CHUNK_SIZE=512
CHUNK_OVERLAP=50
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a Pull Request.

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.