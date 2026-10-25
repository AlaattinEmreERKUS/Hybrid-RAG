# Hybrid GraphRAG Engine

Hybrid retrieval (Vector + Lexical + Graph Traversal) over a Neo4j knowledge graph
and ChromaDB vector store, with a React split-screen UI (chat + live graph canvas).

## Architecture

```
PDF/MD upload ─▶ chunker (512 tok, 64 overlap, structure-aware)
             ├─▶ LLM extraction via Groq (entities/relations/concepts) ─▶ Neo4j (MERGE, idempotent)
             └─▶ local ONNX MiniLM-L6-v2 embeddings (384d) ─▶ ChromaDB (chunk_id = join key)

query ─▶ [vector top-k] + [Neo4j fulltext] + [2-hop graph traversal]
      ─▶ RRF merge ─▶ generator LLM ─▶ answer + provenance (node/edge/chunk ids)
      ─▶ frontend highlights exactly the used subgraph
```

## Run

### 1. Databases

```powershell
docker compose up -d
```

- Neo4j browser: http://localhost:7474 (neo4j / hybridrag123)
- Chroma: http://localhost:8001

### 2. Backend

```powershell
cd backend
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
copy .env.example .env   # then put your real Groq key (gsk_...) in .env
.venv\Scripts\uvicorn app.main:app --reload --port 8000
```

### 3. Frontend

```powershell
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 — Vite proxies `/api` to the backend, no CORS pain.

### 4. Ingest + query

UI: "+ Upload doc" button, then ask questions in chat.
Or via curl:

```powershell
curl -F "file=@mydoc.pdf" http://localhost:8000/api/ingest
curl -X POST http://localhost:8000/api/query -H "Content-Type: application/json" -d "{\"question\": \"...\"}"
```

## Project layout

```
backend/app/
  config.py            pydantic-settings (GRAG_* env vars)
  models/schemas.py    extraction models + camelCase API contract
  services/chunker.py  structure-aware sliding window
  services/extractor.py Groq LLM extraction (JSON mode + Pydantic validation + retry)
  services/embedder.py  local ONNX MiniLM-L6-v2 embeddings (no API needed)
  services/graph_store.py Neo4j: schema, MERGE writes, fulltext, 2-hop traversal
  services/vector_store.py Chroma async client
  pipeline/ingest.py    orchestrator (parse -> chunk -> extract -> store)
  retrieval/retriever.py hybrid legs + RRF + grounded generation
frontend/src/
  components/GraphCanvas.tsx  react-force-graph-2d, highlight retrieved subgraph
  components/ChatPanel.tsx    chat + upload; click answer = highlight sources
  store/useGraphStore.ts      zustand: graph + highlight state
```

## LLM provider

Groq (OpenAI-compatible endpoint). Default model: `llama-3.3-70b-versatile` for both
extraction and generation (`GRAG_EXTRACTION_MODEL` / `GRAG_GENERATION_MODEL` to override).
Groq has no embeddings API — embeddings run locally (ONNX MiniLM, downloads ~80MB on
first use). Free tier is ~30 requests/min: `GRAG_EXTRACTION_CONCURRENCY` defaults to 4;
large PDFs ingest slowly but won't hit 429s.

## Upgrade path

- Cross-encoder rerank: insert `BAAI/bge-reranker-v2-m3` between RRF and generation
  (`sentence-transformers`), rerank top 25 -> 8.
- Entity alias resolution: periodic job, embed entity name+description, merge pairs
  with cosine > 0.92 via `apoc.refactor.mergeNodes` (APOC already enabled in compose).
- Scale vector store: swap Chroma for Pgvector + asyncpg pool when single-node Chroma
  becomes the bottleneck.
