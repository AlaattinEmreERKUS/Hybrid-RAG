# Hybrid RAG

Question answering over a Neo4j knowledge graph and a ChromaDB vector store. Each query runs three retrievers in parallel (vector, fulltext, 2-hop graph traversal), fuses them with Reciprocal Rank Fusion, and returns a cited answer plus the exact subgraph used, which the UI highlights.

## Does hybrid help? Ablation results

`backend/eval/ablation.py` scores each retriever on a labelled question set (hit@3 shown below; the script also prints recall@k and MRR). Corpus: one chapter of *Designing Data-Intensive Applications* (copyrighted, not included; place your copy in `backend/eval/corpus/`) plus two unrelated lecture PDFs as noise. Easy set = 20 questions (keyword, paraphrase, relational); hard set = 14 questions (no lexical overlap, multi-hop).

| Retriever | hit@3 easy | hit@3 hard |
|---|---|---|
| vector | 0.65 | 0.79 |
| fulltext | 0.95 | 0.71 |
| graph | 0.65 | 0.14 |
| vector + fulltext | 0.80 | 0.86 |
| hybrid (all 3, RRF) | 0.90 | 0.79 |

Takeaway: no single retriever wins on both sets; fusion removes the need to pick one. On the hard set the graph leg alone is weak and adding it did not beat vector + fulltext, so the script also tests a down-weighted graph vote (`hybrid g=0.5`, `g=0.25`).

> Measured with `llama-3.3-70b-versatile` for extraction. Numbers are from the author's measurement notes; the repo has no saved results file. Reproduce them with the commands below.

Reproduce (databases up, document ingested, from `backend/`):

```powershell
.venv\Scripts\python -m eval.ablation eval/questions.json
.venv\Scripts\python -m eval.ablation eval/questions_hard.json
```

## How it works

```
PDF/MD upload -> chunker (512 tok, 64 overlap, structure-aware)
             |-> LLM extraction via Groq (entities/relations/concepts) -> Neo4j (MERGE, idempotent)
             '-> local ONNX MiniLM-L6-v2 embeddings (384d) -> ChromaDB (chunk_id = join key)

query -> [vector top-15] + [Neo4j fulltext top-15] + [2-hop graph traversal from query entities]
      -> RRF merge (k=60) -> top 8 chunks -> generator LLM -> answer + provenance (node/edge/chunk ids)
      -> frontend highlights exactly the used subgraph
```

Stack: FastAPI, Neo4j 5 (with APOC), ChromaDB, Groq (OpenAI-compatible API, default model `llama-3.3-70b-versatile`), local ONNX embeddings, React 18 + Vite + zustand + react-force-graph-2d.

## Run

Requires Docker, Python, Node.js and a [Groq API key](https://console.groq.com/keys).

```powershell
# 1. Databases (Neo4j browser :7474, Chroma :8001)
docker compose up -d

# 2. Backend
cd backend
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
copy .env.example .env      # put your Groq key (gsk_...) in .env
.venv\Scripts\uvicorn app.main:app --reload --port 8000

# 3. Frontend (Vite proxies /api to the backend)
cd frontend
npm install
npm run dev                 # http://localhost:5173
```

Ingest and query from the UI ("+ Upload doc", then chat) or the API:

```powershell
curl -F "file=@mydoc.pdf" http://localhost:8000/api/ingest
curl -X POST http://localhost:8000/api/query -H "Content-Type: application/json" -d "{\"question\": \"...\"}"
```

Other endpoints: `GET /api/graph`, `GET /api/health`.

## Project layout

```
backend/app/
  config.py                pydantic-settings (GRAG_* env vars)
  models/schemas.py        extraction models + camelCase API contract
  services/chunker.py      structure-aware sliding window
  services/extractor.py    Groq LLM extraction (JSON mode + Pydantic validation + retry)
  services/embedder.py     local ONNX MiniLM-L6-v2 embeddings
  services/graph_store.py  Neo4j: schema, MERGE writes, fulltext, 2-hop traversal
  services/vector_store.py Chroma async client
  pipeline/ingest.py       parse -> chunk -> extract -> store
  retrieval/retriever.py   three retrieval legs + RRF + grounded generation
backend/eval/              ablation.py, labelled questions, test_metrics.py, corpus/
frontend/src/              ChatPanel, GraphCanvas (highlights retrieved subgraph), zustand store
```

## Notes

- Groq free tier is about 30 requests/min; `GRAG_EXTRACTION_CONCURRENCY` defaults to 4, so large PDFs ingest slowly but avoid 429s. Embeddings run locally (~80 MB model download on first use).
- Eval labels are evidence phrases, not chunk ids, so they survive re-ingestion. Query entities are cached in `eval/entity_cache.json` so reruns are free.
- Not yet implemented: cross-encoder reranking, entity alias merging, a production vector store.
