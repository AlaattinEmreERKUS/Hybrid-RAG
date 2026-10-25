import re
from dataclasses import dataclass, field

from neo4j import AsyncDriver, AsyncGraphDatabase

from app.config import get_settings
from app.models.schemas import (
    Chunk,
    ChunkExtraction,
    GraphLink,
    GraphNode,
    GraphPayload,
    normalize,
)

SCHEMA_STATEMENTS = [
    "CREATE CONSTRAINT doc_id IF NOT EXISTS FOR (d:Document) REQUIRE d.doc_id IS UNIQUE",
    "CREATE CONSTRAINT doc_hash IF NOT EXISTS FOR (d:Document) REQUIRE d.content_hash IS UNIQUE",
    "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
    "CREATE CONSTRAINT entity_key IF NOT EXISTS FOR (e:Entity) REQUIRE (e.normalized_name, e.type) IS UNIQUE",
    "CREATE CONSTRAINT concept_key IF NOT EXISTS FOR (c:Concept) REQUIRE c.normalized_name IS UNIQUE",
    "CREATE FULLTEXT INDEX chunk_text IF NOT EXISTS FOR (c:Chunk) ON EACH [c.text]",
    "CREATE FULLTEXT INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON EACH [e.name, e.normalized_name]",
]

_LUCENE_SPECIAL = re.compile(r'([+\-!(){}\[\]^"~*?:\\/]|&&|\|\|)')


def lucene_escape(text: str) -> str:
    return _LUCENE_SPECIAL.sub(r"\\\1", text)


@dataclass
class GraphContext:
    """Result of the graph-traversal retrieval leg."""
    triples: list[dict] = field(default_factory=list)
    node_ids: list[str] = field(default_factory=list)
    edge_ids: list[str] = field(default_factory=list)
    evidence_chunk_ids: list[str] = field(default_factory=list)

    def subgraph(self) -> GraphPayload:
        nodes: dict[str, GraphNode] = {}
        links: list[GraphLink] = []
        for t in self.triples:
            for nid, label, etype in (
                (t["source_id"], t["source"], t.get("source_type")),
                (t["target_id"], t["target"], t.get("target_type")),
            ):
                if nid in nodes:
                    nodes[nid].degree += 1
                else:
                    nodes[nid] = GraphNode(id=nid, label=label, entity_type=etype, degree=1)
            links.append(GraphLink(
                id=t["edge_id"], source=t["source_id"], target=t["target_id"],
                predicate=t["predicate"], weight=t.get("weight") or 0.5,
            ))
        return GraphPayload(nodes=list(nodes.values()), links=links)


class GraphStore:
    def __init__(self) -> None:
        s = get_settings()
        self._driver: AsyncDriver = AsyncGraphDatabase.driver(
            s.neo4j_uri,
            auth=(s.neo4j_user, s.neo4j_password),
            max_connection_pool_size=s.neo4j_pool_size,
            connection_acquisition_timeout=30,
        )

    async def init_schema(self) -> None:
        async with self._driver.session() as session:
            for stmt in SCHEMA_STATEMENTS:
                await session.run(stmt)

    async def close(self) -> None:
        await self._driver.close()

    # ─── Ingest path ─────────────────────────────────────────────────────────

    async def document_exists(self, content_hash: str) -> bool:
        async with self._driver.session() as session:
            result = await session.run(
                "MATCH (d:Document {content_hash: $h}) RETURN d.doc_id LIMIT 1",
                h=content_hash,
            )
            return await result.single() is not None

    async def write_document(self, doc_id: str, title: str,
                             content_hash: str, chunks: list[Chunk],
                             extractions: list[ChunkExtraction]) -> None:
        chunk_rows = [c.model_dump() for c in chunks]

        entity_rows, mention_rows, relation_rows, concept_rows = [], [], [], []
        for chunk, ext in zip(chunks, extractions):
            for e in ext.entities:
                entity_rows.append({
                    "name": e.name, "norm": e.normalized_name,
                    "type": e.type.value, "description": e.description,
                })
                mention_rows.append({
                    "norm": e.normalized_name, "type": e.type.value,
                    "chunk_id": chunk.chunk_id,
                })
            for r in ext.relations:
                relation_rows.append({
                    "src": normalize(r.source), "tgt": normalize(r.target),
                    "predicate": r.predicate, "confidence": r.confidence,
                    "chunk_id": chunk.chunk_id,
                })
            for cl in ext.concept_links:
                concept_rows.append({
                    "entity": normalize(cl.entity),
                    "concept": cl.parent_concept,
                    "concept_norm": normalize(cl.parent_concept),
                })

        async with self._driver.session() as session:
            # single transaction per document: all-or-nothing ingest
            async def work(tx):
                await tx.run("""
                    MERGE (d:Document {doc_id: $doc_id})
                    SET d.title = $title, d.content_hash = $hash,
                        d.ingested_at = datetime()
                """, doc_id=doc_id, title=title, hash=content_hash)

                await tx.run("""
                    UNWIND $rows AS row
                    MATCH (d:Document {doc_id: $doc_id})
                    MERGE (c:Chunk {chunk_id: row.chunk_id})
                    SET c.text = row.text, c.index = row.index,
                        c.section = row.section, c.token_count = row.token_count
                    MERGE (c)-[:PART_OF]->(d)
                """, rows=chunk_rows, doc_id=doc_id)

                await tx.run("""
                    MATCH (c:Chunk)-[:PART_OF]->(:Document {doc_id: $doc_id})
                    WITH c ORDER BY c.index
                    WITH collect(c) AS cs
                    UNWIND range(0, size(cs) - 2) AS i
                    WITH cs[i] AS a, cs[i + 1] AS b
                    MERGE (a)-[:NEXT]->(b)
                """, doc_id=doc_id)

                # MERGE on dedup key; coalesce keeps first non-null description
                await tx.run("""
                    UNWIND $rows AS row
                    MERGE (e:Entity {normalized_name: row.norm, type: row.type})
                    ON CREATE SET e.name = row.name, e.description = row.description
                    ON MATCH SET e.description = coalesce(e.description, row.description)
                """, rows=entity_rows)

                await tx.run("""
                    UNWIND $rows AS row
                    MATCH (e:Entity {normalized_name: row.norm, type: row.type})
                    MATCH (c:Chunk {chunk_id: row.chunk_id})
                    MERGE (e)-[m:MENTIONED_IN]->(c)
                    ON CREATE SET m.count = 1
                    ON MATCH SET m.count = m.count + 1
                """, rows=mention_rows)

                # relation MERGE keyed by predicate; re-observation bumps weight,
                # appends evidence chunk
                await tx.run("""
                    UNWIND $rows AS row
                    MATCH (s:Entity {normalized_name: row.src})
                    MATCH (t:Entity {normalized_name: row.tgt})
                    MERGE (s)-[r:RELATED_TO {predicate: row.predicate}]->(t)
                    ON CREATE SET r.weight = row.confidence,
                                  r.chunk_ids = [row.chunk_id]
                    ON MATCH SET r.weight = (r.weight + row.confidence) / 2.0 + 0.1,
                                 r.chunk_ids = CASE
                                   WHEN row.chunk_id IN r.chunk_ids THEN r.chunk_ids
                                   ELSE r.chunk_ids + row.chunk_id END
                """, rows=relation_rows)

                await tx.run("""
                    UNWIND $rows AS row
                    MERGE (con:Concept {normalized_name: row.concept_norm})
                    ON CREATE SET con.name = row.concept
                    WITH con, row
                    MATCH (e:Entity {normalized_name: row.entity})
                    MERGE (e)-[:INSTANCE_OF]->(con)
                """, rows=concept_rows)

            await session.execute_write(work)

    # ─── Retrieval path ──────────────────────────────────────────────────────

    async def fulltext_chunks(self, query: str, k: int = 15) -> list[str]:
        """Lexical leg: chunk_ids ranked by Lucene score."""
        q = lucene_escape(query)
        if not q.strip():
            return []
        async with self._driver.session() as session:
            result = await session.run("""
                CALL db.index.fulltext.queryNodes('chunk_text', $q) YIELD node, score
                RETURN node.chunk_id AS chunk_id
                ORDER BY score DESC LIMIT $k
            """, q=q, k=k)
            return [rec["chunk_id"] async for rec in result]

    async def neighborhood(self, entity_names: list[str],
                           max_seeds: int = 5, max_triples: int = 60) -> GraphContext:
        """Graph leg: resolve query entities via fulltext, expand 2 hops."""
        names = [lucene_escape(n) + "~" for n in entity_names if n.strip()]
        if not names:
            return GraphContext()

        async with self._driver.session() as session:
            result = await session.run("""
                UNWIND $names AS qname
                CALL db.index.fulltext.queryNodes('entity_name', qname)
                YIELD node, score
                WITH node AS seed, max(score) AS score
                ORDER BY score DESC LIMIT $max_seeds
                WITH collect(seed) AS seeds
                UNWIND seeds AS seed
                MATCH (seed)-[rs:RELATED_TO*1..2]-(:Entity)
                UNWIND rs AS rel
                WITH DISTINCT rel, startNode(rel) AS s, endNode(rel) AS t
                ORDER BY rel.weight DESC
                LIMIT $max_triples
                RETURN elementId(rel) AS edge_id, rel.predicate AS predicate,
                       rel.weight AS weight, rel.chunk_ids AS chunk_ids,
                       elementId(s) AS source_id, s.name AS source, s.type AS source_type,
                       elementId(t) AS target_id, t.name AS target, t.type AS target_type
            """, names=names, max_seeds=max_seeds, max_triples=max_triples)
            triples = [dict(rec) async for rec in result]

            # evidence chunks cited by edges, deduped in encounter order
            seen: set[str] = set()
            evidence: list[str] = []
            for t in triples:
                for cid in t.get("chunk_ids") or []:
                    if cid not in seen:
                        seen.add(cid)
                        evidence.append(cid)

            node_ids = list({t["source_id"] for t in triples} |
                            {t["target_id"] for t in triples})
            edge_ids = [t["edge_id"] for t in triples]
            return GraphContext(triples=triples, node_ids=node_ids,
                                edge_ids=edge_ids, evidence_chunk_ids=evidence)

    async def fetch_chunks(self, chunk_ids: list[str]) -> list[dict]:
        """Hydrate chunk texts + provenance, preserving input order."""
        if not chunk_ids:
            return []
        async with self._driver.session() as session:
            result = await session.run("""
                UNWIND $ids AS cid
                MATCH (c:Chunk {chunk_id: cid})-[:PART_OF]->(d:Document)
                RETURN c.chunk_id AS id, c.text AS text,
                       c.section AS section, d.title AS doc_title
            """, ids=chunk_ids)
            rows = {rec["id"]: dict(rec) async for rec in result}
        return [rows[cid] for cid in chunk_ids if cid in rows]

    async def overview(self, limit: int = 300) -> GraphPayload:
        """Global graph view: top-N entities by degree + edges between them."""
        async with self._driver.session() as session:
            result = await session.run("""
                MATCH (e:Entity)
                WITH e, COUNT { (e)--() } AS degree
                ORDER BY degree DESC LIMIT $limit
                WITH collect({n: e, degree: degree}) AS rows
                WITH rows, [r IN rows | r.n] AS ns
                UNWIND rows AS row
                WITH rows, ns, row.n AS n
                OPTIONAL MATCH (n)-[r:RELATED_TO]->(b:Entity) WHERE b IN ns
                WITH rows, collect(DISTINCT r) AS rels
                RETURN
                  [row IN rows | {id: elementId(row.n), label: row.n.name,
                                  entityType: row.n.type, degree: row.degree}] AS nodes,
                  [r IN rels | {id: elementId(r), source: elementId(startNode(r)),
                                target: elementId(endNode(r)), predicate: r.predicate,
                                weight: r.weight}] AS links
            """, limit=limit)
            rec = await result.single()
            if rec is None:
                return GraphPayload()
            nodes = [GraphNode(id=n["id"], label=n["label"],
                               entity_type=n["entityType"], degree=n["degree"])
                     for n in rec["nodes"]]
            links = [GraphLink(id=l["id"], source=l["source"], target=l["target"],
                               predicate=l["predicate"], weight=l["weight"] or 0.5)
                     for l in rec["links"]]
            return GraphPayload(nodes=nodes, links=links)
