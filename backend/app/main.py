import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from app.api.routes import router
from app.pipeline.ingest import IngestPipeline
from app.retrieval.retriever import QueryService
from app.services.embedder import Embedder
from app.services.extractor import Extractor
from app.services.graph_store import GraphStore
from app.services.vector_store import VectorStore

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    graph = GraphStore()
    vectors = VectorStore()
    await graph.init_schema()
    await vectors.connect()

    extractor = Extractor()
    embedder = Embedder()
    app.state.graph = graph
    app.state.vectors = vectors
    app.state.pipeline = IngestPipeline(graph, vectors, extractor, embedder)
    app.state.query_service = QueryService(graph, vectors, embedder, extractor)
    yield
    await graph.close()


app = FastAPI(title="Hybrid GraphRAG Engine", lifespan=lifespan)
app.add_middleware(GZipMiddleware, minimum_size=1024)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router, prefix="/api")
