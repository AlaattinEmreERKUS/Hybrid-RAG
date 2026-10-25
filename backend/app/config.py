from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="GRAG_")

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "hybridrag123"
    neo4j_pool_size: int = 50

    # Chroma
    chroma_host: str = "localhost"
    chroma_port: int = 8001
    chroma_collection: str = "chunks"

    # LLM (Groq, OpenAI-compatible API)
    groq_api_key: str
    groq_base_url: str = "https://api.groq.com/openai/v1"
    extraction_model: str = "llama-3.3-70b-versatile"
    generation_model: str = "llama-3.3-70b-versatile"
    # free tier answers 429 with retry-after ~20s; SDK default (2) gives up mid-ingest
    llm_max_retries: int = 8

    # Embeddings: local ONNX MiniLM-L6-v2 (Groq has no embeddings API)
    embedding_dim: int = 384

    # Chunking / pipeline
    chunk_tokens: int = 512
    chunk_overlap: int = 64
    extraction_concurrency: int = 4  # Groq free tier: ~30 RPM, keep modest

    # Retrieval
    vector_top_k: int = 15
    lexical_top_k: int = 15
    rerank_candidates: int = 25
    final_top_k: int = 8


@lru_cache
def get_settings() -> Settings:
    return Settings()
