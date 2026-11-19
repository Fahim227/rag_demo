# rag_demo

Ask questions about your PDFs. Upload a PDF, it gets chunked, embedded with Gemini and stored in Qdrant; questions are answered by Gemini using the most relevant chunks. The pipeline runs as [Inngest](https://www.inngest.com/) durable functions served by FastAPI, with a Streamlit UI on top.

## Architecture

```
Streamlit UI ──send event──▶ Inngest dev server (:8288) ──▶ FastAPI /api/inngest (:8000)
     ▲                                                          │
     └──── polls run output ◀───────────────────────────────────┤
                                                                ├─ Gemini (embeddings + answers)
                                                                └─ Qdrant (:6333)
```

| Event | Function | Steps |
|---|---|---|
| `rag/ingest_pdf` | `rag_ingest_pdf` | `load-and-chunk` → `embed-and-upsert` |
| `rag/query_pdf_ai` | `rag_query_answer` | `embed-and-search` → `llm-answer` |

Each step is retried and memoized independently by Inngest, so a failed LLM call doesn't redo the embedding.

## Project layout

| File | Purpose |
|---|---|
| `main.py` | Inngest client, the two functions, FastAPI app |
| `data_loader.py` | PDF loading/chunking (llama-index), Gemini embeddings (`gemini-embedding-001`, 3072 dims) |
| `vector_db.py` | `QdrantStorage`: collection setup, upsert, search |
| `custom_types.py` | Pydantic models passed between steps |
| `streamlit_app.py` | UI for uploading PDFs and asking questions |
| `uploads/` | PDFs saved by the UI |
| `qdrant_storage/` | Qdrant data volume |

## Requirements

- Python 3.14+ and [uv](https://docs.astral.sh/uv/)
- Node.js (for the Inngest dev server via `npx`)
- Docker (for Qdrant)
- A Gemini API key

## Setup

```bash
uv sync
echo "GEMINI_API_KEY=your-key" > .env
```

## Running

Start each in its own terminal:

```bash
# 1. Qdrant
docker run -p 6333:6333 -v "$(pwd)/qdrant_storage:/qdrant/storage" qdrant/qdrant

# 2. FastAPI app (serves /api/inngest)
uv run uvicorn main:app --reload

# 3. Inngest dev server
npx inngest-cli@latest dev -u http://127.0.0.1:8000/api/inngest

# 4. Streamlit UI
uv run streamlit run streamlit_app.py
```

Open http://localhost:8501 to upload a PDF and ask questions. The Inngest dashboard at http://localhost:8288 shows every run and its steps.

### Without the UI

Send events from the Inngest dashboard:

```json
// rag/ingest_pdf
{ "pdf_path": "/absolute/path/to/file.pdf", "source_id": "file.pdf" }

// rag/query_pdf_ai
{ "question": "What is this document about?", "top_k": 5 }
```

The query run's output is `{"answer": ..., "sources": [...], "num_contexts": ...}`.

## Configuration

| Variable | Default | Used by |
|---|---|---|
| `GEMINI_API_KEY` | — (required) | `data_loader.py` |
| `INNGEST_API_BASE` | `http://127.0.0.1:8288/v1` | `streamlit_app.py` |

Qdrant URL (`http://localhost:6333`), collection (`docs`) and the answer model (`gemini-3.5-flash-lite`) are set in code.

## Troubleshooting

- **`function_config_invalid: no functions found`**: the function isn't in the list passed to `inngest.fast_api.serve(...)` in `main.py`.
- **UI loads but answers never arrive**: one of FastAPI, the Inngest dev server or Qdrant isn't running.
- **Qdrant dimension errors**: the `docs` collection was created with a different vector size. Delete it if you change the embedding model or dimension.
