import io
import logging

import pymupdf

from app.models.schemas import ChunkExtraction
from app.services.chunker import chunk_document, content_hash
from app.services.embedder import Embedder
from app.services.extractor import Extractor
from app.services.graph_store import GraphStore
from app.services.vector_store import VectorStore

log = logging.getLogger(__name__)


def parse_file(filename: str, data: bytes) -> str:
    if filename.lower().endswith(".pdf"):
        with pymupdf.open(stream=io.BytesIO(data), filetype="pdf") as doc:
            return "\n\n".join(page.get_text("text") for page in doc)
    return data.decode("utf-8", errors="replace")


class IngestPipeline:
    def __init__(self, graph: GraphStore, vectors: VectorStore,
                 extractor: Extractor, embedder: Embedder) -> None:
        self.graph, self.vectors = graph, vectors
        self.extractor, self.embedder = extractor, embedder

    async def run(self, filename: str, data: bytes) -> dict:
        text = parse_file(filename, data)
        chash = content_hash(text)
        if await self.graph.document_exists(chash):
            return {"status": "skipped_duplicate", "doc_id": "", "chunks": 0}

        doc_id, chunks = chunk_document(text)
        log.info("doc %s (%s): %d chunks", doc_id, filename, len(chunks))

        results = await self.extractor.extract_all(chunks)
        failed = sum(r is None for r in results)
        if failed:
            log.warning("doc %s: extraction failed for %d/%d chunks", doc_id, failed, len(chunks))
        extractions = [r or ChunkExtraction() for r in results]
        embeddings = await self.embedder.embed([c.text for c in chunks])

        # graph first; vector upsert idempotent so partial failure recoverable
        await self.graph.write_document(doc_id, filename, chash, chunks, extractions)
        await self.vectors.upsert_chunks(filename, chash, chunks, extractions, embeddings)

        return {"status": "ingested", "doc_id": doc_id, "chunks": len(chunks),
                "failed_extractions": failed}
