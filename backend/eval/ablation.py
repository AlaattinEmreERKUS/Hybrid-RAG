"""Retrieval ablation: which leg finds the right chunks?

Usage (from backend/, databases up, document already ingested):
    .venv\\Scripts\\python -m eval.ablation eval/questions.json

questions.json: [{"question": "...", "evidence": ["phrase that appears in the answer chunk", ...]}]
A chunk is relevant if its text contains any evidence phrase (case/whitespace-insensitive),
so labels survive re-ingestion even though chunk ids change.
"""
import asyncio
import json
import re
import sys

from app.config import get_settings
from app.retrieval.retriever import QueryService, rrf_merge
from app.services.embedder import Embedder
from app.services.extractor import Extractor
from app.services.graph_store import GraphStore
from app.services.vector_store import VectorStore


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def relevant_ids(chunks: dict[str, str], evidence: list[str]) -> set[str]:
    phrases = [norm(e) for e in evidence]
    return {cid for cid, text in chunks.items() if any(p in norm(text) for p in phrases)}


def score(ranking: list[str], relevant: set[str], k: int) -> tuple[float, float, float]:
    """(hit@k, recall@k, reciprocal rank) for one question."""
    top = ranking[:k]
    found = [cid for cid in top if cid in relevant]
    rr = next((1.0 / (i + 1) for i, cid in enumerate(ranking) if cid in relevant), 0.0)
    return float(bool(found)), len(found) / len(relevant), rr


async def all_chunks(graph: GraphStore) -> dict[str, str]:
    async with graph._driver.session() as session:
        result = await session.run("MATCH (c:Chunk) RETURN c.chunk_id AS id, c.text AS text")
        return {rec["id"]: rec["text"] async for rec in result}


async def main(path: str) -> None:
    s = get_settings()
    k = s.final_top_k
    questions = json.load(open(path, encoding="utf-8"))

    graph, vectors = GraphStore(), VectorStore()
    await vectors.connect()
    service = QueryService(graph, vectors, Embedder(), Extractor())
    chunks = await all_chunks(graph)
    print(f"{len(chunks)} chunks in store, {len(questions)} questions, k={k}\n")

    configs = ["vector", "lexical", "graph", "vector+lexical", "hybrid (all 3)"]
    totals = {c: [0.0, 0.0, 0.0] for c in configs}
    by_type: dict[str, dict[str, list[float]]] = {}  # type -> config -> [hits, count]
    used = 0
    for q in questions:
        rel = relevant_ids(chunks, q["evidence"])
        if not rel:
            print(f"SKIP (no chunk matches evidence): {q['question']}")
            continue
        used += 1
        vec, lex, ctx = await service.retrieve(q["question"])
        gph = ctx.evidence_chunk_ids
        rankings = {
            "vector": vec,
            "lexical": lex,
            "graph": gph,
            "vector+lexical": rrf_merge([vec, lex]),
            "hybrid (all 3)": rrf_merge([vec, lex, gph]),
        }
        marks = []
        for c, ranking in rankings.items():
            result = score(ranking, rel, k)
            for i, v in enumerate(result):
                totals[c][i] += v
            t = by_type.setdefault(q.get("type", "-"), {}).setdefault(c, [0.0, 0])
            t[0] += result[0]
            t[1] += 1
            marks.append("✓" if result[0] else "·")
        print(" ".join(marks), q["question"])

    await graph.close()
    if not used:
        sys.exit("no usable questions")
    print(f"\n| config | hit@{k} | recall@{k} | MRR |\n|---|---|---|---|")
    for c in configs:
        hit, rec, mrr = (v / used for v in totals[c])
        print(f"| {c} | {hit:.2f} | {rec:.2f} | {mrr:.2f} |")
    print(f"\nhit@{k} by question type\n| type | " + " | ".join(configs) + " |")
    print("|---" * (len(configs) + 1) + "|")
    for qtype, row in by_type.items():
        n = next(iter(row.values()))[1]
        print(f"| {qtype} (n={n}) | " + " | ".join(f"{row[c][0] / row[c][1]:.2f}" for c in configs) + " |")
    print(f"\n{used} questions scored; marks per line: {', '.join(configs)}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "eval/questions.json"))
