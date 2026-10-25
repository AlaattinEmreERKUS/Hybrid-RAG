import asyncio

from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2


class Embedder:
    """Local embeddings: all-MiniLM-L6-v2 via ONNX runtime (384 dims).

    Groq has no embeddings endpoint, so embedding runs in-process.
    First call downloads the model (~80MB) to ~/.cache/chroma/onnx_models.
    """

    def __init__(self) -> None:
        self._ef = ONNXMiniLM_L6_V2()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self._embed_sync, texts)

    def _embed_sync(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), 64):
            out.extend([float(x) for x in vec] for vec in self._ef(texts[i:i + 64]))
        return out
