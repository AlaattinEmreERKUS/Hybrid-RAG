import asyncio
import logging

from openai import AsyncOpenAI

from app.config import get_settings
from app.models.schemas import Passage, QueryResponse, RetrievalInfo
from app.services.embedder import Embedder
from app.services.extractor import Extractor
from app.services.graph_store import GraphContext, GraphStore
from app.services.vector_store import VectorStore

log = logging.getLogger(__name__)

ANSWER_PROMPT = """You are a precise question-answering assistant.
Answer ONLY from the knowledge-graph facts and source passages below.
If the context is insufficient, say so explicitly — do not guess.
Cite passage numbers like [1], [2] for every claim.

## Knowledge Graph Facts
{facts}

## Source Passages
{passages}

## Question
{question}"""


def rrf_merge(result_lists: list[list[str]], k: int = 60) -> list[str]:
    """Reciprocal Rank Fusion. Each inner list: chunk_ids ranked best-first."""
    scores: dict[str, float] = {}
    for ranking in result_lists:
        for rank, chunk_id in enumerate(ranking):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return [cid for cid, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


class QueryService:
    def __init__(self, graph: GraphStore, vectors: VectorStore,
                 embedder: Embedder, extractor: Extractor) -> None:
        s = get_settings()
        self.graph, self.vectors = graph, vectors
        self.embedder, self.extractor = embedder, extractor
        self._client = AsyncOpenAI(api_key=s.groq_api_key, base_url=s.groq_base_url,
                                   max_retries=s.llm_max_retries)
        self._model = s.generation_model
        self._s = s

    async def retrieve(self, question: str) -> tuple[list[str], list[str], GraphContext]:
        """Run the three retrieval legs; each list is chunk_ids ranked best-first."""
        s = self._s

        # leg inputs in parallel: query embedding + query entities
        (q_emb,), q_entities = await asyncio.gather(
            self.embedder.embed([question]),
            self.extractor.extract_query_entities(question),
        )
        log.info("query entities: %s", q_entities)

        # three retrieval legs in parallel
        vec_ids, lex_ids, graph_ctx = await asyncio.gather(
            self.vectors.query(q_emb, k=s.vector_top_k),
            self.graph.fulltext_chunks(question, k=s.lexical_top_k),
            self.graph.neighborhood(q_entities),
        )
        return vec_ids, lex_ids, graph_ctx

    async def answer(self, question: str) -> QueryResponse:
        s = self._s
        vec_ids, lex_ids, graph_ctx = await self.retrieve(question)

        fused = rrf_merge([vec_ids, lex_ids, graph_ctx.evidence_chunk_ids])
        top_ids = fused[:s.rerank_candidates][:s.final_top_k]
        chunks = await self.graph.fetch_chunks(top_ids)

        facts = "\n".join(
            f"({t['source']}) -[{t['predicate']}]-> ({t['target']})"
            for t in graph_ctx.triples[:40]
        ) or "(none)"
        passages = "\n\n".join(
            f"[{i + 1}] ({c['doc_title']}, \"{c['section']}\")\n{c['text']}"
            for i, c in enumerate(chunks)
        ) or "(none)"

        rsp = await self._client.chat.completions.create(
            model=self._model,
            messages=[{
                "role": "user",
                "content": ANSWER_PROMPT.format(
                    facts=facts, passages=passages, question=question),
            }],
            temperature=0.1,
        )
        answer = rsp.choices[0].message.content or ""

        return QueryResponse(
            answer=answer,
            retrieval=RetrievalInfo(
                chunk_ids=[c["id"] for c in chunks],
                node_ids=graph_ctx.node_ids,
                edge_ids=graph_ctx.edge_ids,
                passages=[Passage(id=c["id"], doc_title=c["doc_title"],
                                  section=c["section"] or "", text=c["text"])
                          for c in chunks],
            ),
            subgraph=graph_ctx.subgraph(),
        )
