from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.alias_generators import to_camel


def normalize(name: str) -> str:
    return " ".join(name.lower().strip().split())


# ─── Extraction (LLM structured output) ─────────────────────────────────────

class EntityType(str, Enum):
    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    TECHNOLOGY = "TECHNOLOGY"
    CONCEPT = "CONCEPT"
    LOCATION = "LOCATION"
    EVENT = "EVENT"
    PRODUCT = "PRODUCT"


class ExtractedEntity(BaseModel):
    """Single entity found in a chunk."""
    name: str = Field(description="Canonical surface form, e.g. 'Neo4j'")
    type: EntityType
    description: str = Field(description="One-sentence definition grounded in the chunk")

    @property
    def normalized_name(self) -> str:
        return normalize(self.name)


class ExtractedRelation(BaseModel):
    """Directed triple between two extracted entities."""
    source: str = Field(description="Source entity name, must match an extracted entity")
    target: str = Field(description="Target entity name, must match an extracted entity")
    predicate: str = Field(description="snake_case verb phrase, e.g. 'depends_on'")
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("predicate")
    @classmethod
    def snake(cls, v: str) -> str:
        return v.strip().lower().replace(" ", "_")


class ConceptLink(BaseModel):
    """Hierarchical placement: entity -> parent concept."""
    entity: str
    parent_concept: str = Field(description="Broader category, e.g. 'Graph Databases'")


class ChunkExtraction(BaseModel):
    """Full structured-output schema sent to the LLM."""
    entities: list[ExtractedEntity] = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)
    concept_links: list[ConceptLink] = Field(default_factory=list)


class QueryEntities(BaseModel):
    """Entity names / key technical terms extracted from a user question."""
    entities: list[str] = Field(default_factory=list)


# ─── Internal pipeline models ────────────────────────────────────────────────

class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    index: int
    text: str
    section: str
    token_count: int


# ─── API contract (camelCase on the wire, matches frontend types.ts) ────────

class ApiModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class IngestResponse(ApiModel):
    doc_id: str
    status: str
    chunks: int
    failed_extractions: int = 0


class GraphNode(ApiModel):
    id: str
    label: str
    type: str = "Entity"
    entity_type: str | None = None
    degree: int = 1


class GraphLink(ApiModel):
    id: str
    source: str
    target: str
    predicate: str
    weight: float = 0.5


class GraphPayload(ApiModel):
    nodes: list[GraphNode] = Field(default_factory=list)
    links: list[GraphLink] = Field(default_factory=list)


class Passage(ApiModel):
    id: str
    doc_title: str
    section: str
    text: str


class RetrievalInfo(ApiModel):
    chunk_ids: list[str] = Field(default_factory=list)
    node_ids: list[str] = Field(default_factory=list)
    edge_ids: list[str] = Field(default_factory=list)
    passages: list[Passage] = Field(default_factory=list)


class QueryRequest(ApiModel):
    question: str


class QueryResponse(ApiModel):
    answer: str
    retrieval: RetrievalInfo
    subgraph: GraphPayload
