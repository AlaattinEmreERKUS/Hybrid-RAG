import chromadb

from app.config import get_settings
from app.models.schemas import Chunk, ChunkExtraction


class VectorStore:
    def __init__(self) -> None:
        self._collection = None

    async def connect(self) -> None:
        s = get_settings()
        client = await chromadb.AsyncHttpClient(host=s.chroma_host, port=s.chroma_port)
        self._collection = await client.get_or_create_collection(
            name=s.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )

    async def upsert_chunks(self, doc_title: str, content_hash: str,
                            chunks: list[Chunk],
                            extractions: list[ChunkExtraction],
                            embeddings: list[list[float]]) -> None:
        await self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=embeddings,
            metadatas=[{
                "doc_id": c.doc_id,
                "doc_title": doc_title,
                "section": c.section,
                "chunk_index": c.index,
                "token_count": c.token_count,
                "entities": "|".join(sorted({e.normalized_name for e in ext.entities})),
                "content_hash": content_hash,
            } for c, ext in zip(chunks, extractions)],
        )

    async def query(self, embedding: list[float], k: int = 15) -> list[str]:
        """Vector leg: chunk_ids ranked by cosine similarity."""
        rsp = await self._collection.query(
            query_embeddings=[embedding], n_results=k, include=["distances"],
        )
        ids = rsp.get("ids") or [[]]
        return ids[0]
