import asyncio
import json
import logging
from typing import TypeVar

from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

from app.config import get_settings
from app.models.schemas import Chunk, ChunkExtraction, QueryEntities, normalize

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

SYSTEM_PROMPT = """You are a knowledge-graph extraction engine.
From the given text chunk extract:
1. entities — concrete named things (people, orgs, technologies, products, events).
2. relations — directed triples ONLY between entities you extracted. Predicate is a
   short snake_case verb phrase. Set confidence by how explicitly the text states it.
3. concept_links — map each entity to ONE broader parent concept (category).

Rules:
- Do NOT invent entities not present in the text.
- Resolve pronouns to entity names when unambiguous.
- Prefer fewer, high-precision relations over many weak ones."""

QUERY_ENTITY_PROMPT = """Extract the named entities and key technical terms from the
user's question. Return only terms likely to be nodes in a knowledge graph.
Do not include generic words like 'difference', 'overview', 'system'."""


class Extractor:
    """LLM extraction via Groq (OpenAI-compatible API).

    Groq does not support strict json_schema structured outputs on all models,
    so this uses JSON mode + schema-in-prompt + Pydantic validation with retry.
    """

    def __init__(self) -> None:
        s = get_settings()
        self._client = AsyncOpenAI(api_key=s.groq_api_key, base_url=s.groq_base_url,
                                   max_retries=s.llm_max_retries)
        self._model = s.extraction_model
        self._sem = asyncio.Semaphore(s.extraction_concurrency)

    async def _parse(self, model_cls: type[T], system: str, user: str,
                     retries: int = 2) -> T:
        schema = json.dumps(model_cls.model_json_schema(), indent=None)
        sys_msg = (
            f"{system}\n\n"
            f"Respond ONLY with a single JSON object that validates against this "
            f"JSON schema (no prose, no markdown fences):\n{schema}"
        )
        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user},
        ]
        last_err: Exception | None = None
        for _ in range(retries):
            rsp = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            raw = rsp.choices[0].message.content or "{}"
            try:
                return model_cls.model_validate_json(raw)
            except ValidationError as e:
                last_err = e
                # feed the validation error back for one corrective attempt
                messages = messages + [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content":
                        f"Your JSON failed validation: {e}. "
                        f"Return a corrected JSON object only."},
                ]
        raise last_err  # type: ignore[misc]

    async def extract_chunk(self, chunk: Chunk) -> ChunkExtraction | None:
        """None = extraction failed (distinct from a chunk with no entities)."""
        async with self._sem:
            try:
                result = await self._parse(ChunkExtraction, SYSTEM_PROMPT, chunk.text)
            except Exception:
                log.exception("extraction failed for %s", chunk.chunk_id)
                return None

        # discard relations referencing entities the model hallucinated
        names = {e.normalized_name for e in result.entities}
        result.relations = [
            r for r in result.relations
            if normalize(r.source) in names and normalize(r.target) in names
        ]
        return result

    async def extract_all(self, chunks: list[Chunk]) -> list[ChunkExtraction | None]:
        return await asyncio.gather(*(self.extract_chunk(c) for c in chunks))

    async def extract_query_entities(self, question: str) -> list[str]:
        try:
            parsed = await self._parse(QueryEntities, QUERY_ENTITY_PROMPT, question)
            return [e for e in parsed.entities if e.strip()]
        except Exception:
            log.exception("query entity extraction failed")
            return []
