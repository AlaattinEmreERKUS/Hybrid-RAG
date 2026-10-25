"""Re-run LLM extraction for chunks that ended up with no entities (e.g. after 429s).

Usage (from backend/): .venv\\Scripts\\python -m app.pipeline.reextract "<document title>"
Runs one chunk at a time to stay under the provider's tokens-per-minute limit.
write_document MERGEs, so writing just the repaired chunks is safe.
"""
import asyncio
import sys

from app.models.schemas import Chunk
from app.services.extractor import Extractor
from app.services.graph_store import GraphStore


async def main(title: str) -> None:
    graph, extractor = GraphStore(), Extractor()
    async with graph._driver.session() as session:
        result = await session.run("""
            MATCH (d:Document {title: $title})<-[:PART_OF]-(c:Chunk)
            WHERE NOT ()-[:MENTIONED_IN]->(c)
            RETURN d.doc_id AS doc_id, d.content_hash AS hash, c.chunk_id AS chunk_id,
                   c.index AS index, c.text AS text, c.section AS section,
                   c.token_count AS token_count
            ORDER BY c.index
        """, title=title)
        rows = [dict(r) async for r in result]

    print(f"{len(rows)} chunks without entities")
    fixed = 0
    for r in rows:
        chunk = Chunk(chunk_id=r["chunk_id"], doc_id=r["doc_id"], index=r["index"],
                      text=r["text"], section=r["section"] or "", token_count=r["token_count"])
        ext = await extractor.extract_chunk(chunk)
        if ext is None or not ext.entities:
            print(f"  chunk {chunk.index}: still empty")
            continue
        await graph.write_document(r["doc_id"], title, r["hash"], [chunk], [ext])
        fixed += 1
        print(f"  chunk {chunk.index}: {len(ext.entities)} entities, {len(ext.relations)} relations")
    await graph.close()
    print(f"repaired {fixed}/{len(rows)}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
