"""
Pydantic v2 models for Stage 2: Graph Indexing & Neo4j Integration.

Payload contract:
- Input: ExtractionResult (from Stage 1, imported)
- Output: GraphWriteResult (what we return after processing)
- Internal: CanonicalNode, CanonicalRelation (the graph state)
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, field_validator


# ─────────────────────────────────────────────────────────────────────────────
# Import Stage 1 payload for reference
# ─────────────────────────────────────────────────────────────────────────────

from stages.extraction.schemas import (
    ExtractionResult,
    Chunk,
    CandidateEntity,
    CandidateRelation,
    DocumentMetadata,
)


# ─────────────────────────────────────────────────────────────────────────────
# Entity Resolution Models
# ─────────────────────────────────────────────────────────────────────────────


class CanonicalNode(BaseModel):
    """
    A canonical entity node after resolution and deduplication.
    Merges surface forms and sources from multiple candidate entities.
    """

    id: str = Field(
        ...,
        description="Deterministic ID: entity_type:canonical_name (lowercase, normalized)",
    )
    entity_type: str = Field(..., description="Type label (Person, Organization, Location, etc.)")
    canonical_name: str = Field(
        ..., description="Single authoritative name chosen from surface forms"
    )
    surface_forms: list[str] = Field(
        default_factory=list,
        description="All variations seen across documents; guaranteed to include canonical_name",
    )
    description: str = Field(
        default="", description="Description synthesized from candidate descriptions"
    )
    sources: list[str] = Field(
        default_factory=list,
        description="Document IDs (ExtractionResult.metadata.document_id) mentioning this entity",
    )
    first_seen: datetime = Field(
        default_factory=datetime.utcnow, description="When this canonical form was first inferred"
    )
    updated: datetime = Field(
        default_factory=datetime.utcnow, description="Last update timestamp"
    )
    candidate_ids: list[str] = Field(
        default_factory=list,
        description="CandidateEntity IDs that merged into this canonical node; for tracing",
    )

    @field_validator("surface_forms")
    @classmethod
    def surface_forms_includes_canonical(cls, v: list[str], info) -> list[str]:
        """Ensure canonical_name is always in surface_forms."""
        canonical = info.data.get("canonical_name", "")
        if canonical and canonical not in v:
            v = [canonical] + v
        return list(dict.fromkeys(v))  # deduplicate while preserving order

    @field_validator("id")
    @classmethod
    def validate_id_format(cls, v: str) -> str:
        """ID must follow entity_type:name format."""
        if ":" not in v:
            raise ValueError(f"Invalid canonical ID format: {v}. Expected 'entity_type:name'")
        return v


class CanonicalRelation(BaseModel):
    """
    A canonical edge between two entities after resolution and aggregation.
    Merges evidence and supporting chunks from multiple candidate relations.
    """

    source_id: str = Field(..., description="Canonical entity ID (source of edge)")
    target_id: str = Field(..., description="Canonical entity ID (target of edge)")
    relation_type: str = Field(..., description="Edge label type (IS_A, WORKS_AT, LOCATED_IN, etc.)")
    description: str = Field(default="", description="Merged description from candidates")
    evidence: list[str] = Field(
        default_factory=list, description="Verbatim spans supporting this relation"
    )
    supporting_chunks: list[str] = Field(
        default_factory=list, description="Chunk IDs (Chunk.id) providing evidence"
    )
    relation_count: int = Field(
        default=1, description="How many candidate relations merged into this edge"
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Support breadth: ratio of assertions to total entity mentions",
    )
    created: datetime = Field(
        default_factory=datetime.utcnow, description="When this relation was first created"
    )
    updated: datetime = Field(
        default_factory=datetime.utcnow, description="Last update timestamp"
    )


class ResolutionMetadata(BaseModel):
    """Statistics and audit info from the entity resolution process."""

    total_candidates: int = Field(default=0, description="Number of candidate entities input")
    total_canonical: int = Field(default=0, description="Number of canonical nodes output")
    within_doc_merges: int = Field(default=0, description="Candidates merged within the same document")
    exact_match_merges: int = Field(default=0, description="Merges via exact name match")
    fuzzy_match_candidates: int = Field(
        default=0, description="Pairs flagged for fuzzy matching but not auto-merged"
    )
    rule_based_merges: int = Field(default=0, description="Merges from _config/ rules")
    total_surface_forms: int = Field(default=0, description="Total distinct surface forms across all canonical nodes")
    errors: list[dict] = Field(
        default_factory=list, description="Non-fatal resolution errors (e.g. invalid IDs)"
    )


class EntityResolutionResult(BaseModel):
    """Output of the entity resolution phase."""

    canonical_entities: dict[str, CanonicalNode] = Field(
        default_factory=dict, description="Map: candidate_id → CanonicalNode"
    )
    canonical_relations: dict[str, CanonicalRelation] = Field(
        default_factory=dict,
        description="Map: (source_id:target_id:relation_type) → CanonicalRelation",
    )
    candidate_to_canonical: dict[str, str] = Field(
        default_factory=dict, description="Map: candidate_entity_id → canonical_node_id"
    )
    metadata: ResolutionMetadata = Field(default_factory=ResolutionMetadata)


# ─────────────────────────────────────────────────────────────────────────────
# Neo4j Schema & Index Models
# ─────────────────────────────────────────────────────────────────────────────


class ConstraintConfig(BaseModel):
    """Neo4j constraint definition."""

    label: str = Field(..., description="Node label (Entity, Chunk, Document, etc.)")
    property: str = Field(..., description="Property name to constrain")
    constraint_type: str = Field(
        default="UNIQUE",
        description="Constraint type: UNIQUE, NOT_NULL, NODE_KEY, etc.",
    )
    cypher: str = Field(
        ..., description="Full Cypher statement to create this constraint"
    )


class IndexConfig(BaseModel):
    """Neo4j index definition."""

    name: str = Field(..., description="Index name")
    label: str = Field(..., description="Node label to index")
    properties: list[str] = Field(
        ..., description="Property names to index (single or compound)"
    )
    index_type: str = Field(
        default="BTREE", description="Index type: BTREE, TEXT, VECTOR, FULLTEXT, etc."
    )
    cypher: str = Field(
        ..., description="Full Cypher statement to create this index"
    )
    vector_config: Optional[dict] = Field(
        default=None,
        description="For VECTOR indexes: {dimensions, similarity_function, etc.}",
    )


class GraphSchema(BaseModel):
    """Neo4j schema: constraints and indexes."""

    constraints: list[ConstraintConfig] = Field(
        default_factory=list, description="Constraints to enforce"
    )
    indexes: list[IndexConfig] = Field(
        default_factory=list, description="Indexes for query performance"
    )

    @classmethod
    def default(cls) -> "GraphSchema":
        """Return the default schema used by Stage 2."""
        return cls(
            constraints=[
                ConstraintConfig(
                    label="Entity",
                    property="id",
                    constraint_type="UNIQUE",
                    cypher="CREATE CONSTRAINT entity_id_unique FOR (e:Entity) REQUIRE e.id IS UNIQUE",
                ),
                ConstraintConfig(
                    label="Chunk",
                    property="id",
                    constraint_type="UNIQUE",
                    cypher="CREATE CONSTRAINT chunk_id_unique FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
                ),
                ConstraintConfig(
                    label="Document",
                    property="id",
                    constraint_type="UNIQUE",
                    cypher="CREATE CONSTRAINT document_id_unique FOR (d:Document) REQUIRE d.id IS UNIQUE",
                ),
            ],
            indexes=[
                IndexConfig(
                    name="entity_type_idx",
                    label="Entity",
                    properties=["entity_type"],
                    index_type="BTREE",
                    cypher="CREATE INDEX entity_type_idx FOR (e:Entity) ON (e.entity_type)",
                ),
                IndexConfig(
                    name="chunk_document_idx",
                    label="Chunk",
                    properties=["document_id"],
                    index_type="BTREE",
                    cypher="CREATE INDEX chunk_document_idx FOR (c:Chunk) ON (c.document_id)",
                ),
                IndexConfig(
                    name="document_uri_idx",
                    label="Document",
                    properties=["uri"],
                    index_type="BTREE",
                    cypher="CREATE INDEX document_uri_idx FOR (d:Document) ON (d.uri)",
                ),
            ],
        )


# ─────────────────────────────────────────────────────────────────────────────
# Embedding Models
# ─────────────────────────────────────────────────────────────────────────────


class EmbeddingRequest(BaseModel):
    """Request to embed a batch of chunks."""

    chunks: list[str] = Field(..., description="List of chunk texts to embed")
    model: str = Field(..., description="Model ID (e.g., text-embedding-3-large)")
    provider: str = Field(
        ..., description="Provider (anthropic or openai)"
    )


class EmbeddingResult(BaseModel):
    """Result of embedding a batch of chunks."""

    embeddings: list[list[float]] = Field(
        ..., description="List of vectors, one per chunk"
    )
    model: str = Field(..., description="Model ID used")
    dimensions: int = Field(..., description="Vector dimension")
    tokens_used: int = Field(
        default=0, description="Total tokens consumed by the embedding call"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Write & Upsert Results
# ─────────────────────────────────────────────────────────────────────────────


class NodeUpsertResult(BaseModel):
    """Result of upserting entities and chunks to Neo4j."""

    entities_created: int = Field(default=0, description="New entity nodes inserted")
    entities_updated: int = Field(default=0, description="Existing entity nodes updated")
    chunks_created: int = Field(default=0, description="New chunk nodes inserted")
    chunks_updated: int = Field(default=0, description="Existing chunk nodes updated")
    documents_created: int = Field(default=0, description="New document nodes inserted")
    documents_updated: int = Field(default=0, description="Existing document nodes updated")


class RelationUpsertResult(BaseModel):
    """Result of upserting relationships to Neo4j."""

    mention_relations_created: int = Field(
        default=0, description="New MENTIONED_IN edges created"
    )
    mention_relations_updated: int = Field(
        default=0, description="Existing MENTIONED_IN edges updated"
    )
    content_relations_created: int = Field(
        default=0, description="New domain relations (IS_A, WORKS_AT, etc.) created"
    )
    content_relations_updated: int = Field(
        default=0, description="Existing domain relations updated"
    )
    from_relations_created: int = Field(
        default=0, description="New FROM edges (Chunk → Document) created"
    )


class GraphWriteResult(BaseModel):
    """
    Output payload from Stage 2: the result of processing one ExtractionResult.
    This is returned to the caller and passed to Stage 3's graph accessor.
    """

    document_id: str = Field(..., description="Document ID (from ExtractionResult.metadata)")
    document_uri: str = Field(..., description="Document URI")
    extraction_model: str = Field(..., description="Model used for extraction (Stage 1)")
    schema_version: str = Field(..., description="Graph schema version")
    ingested_at: datetime = Field(
        default_factory=datetime.utcnow, description="When this document was processed"
    )
    nodes: NodeUpsertResult = Field(
        default_factory=NodeUpsertResult, description="Node upsert counts"
    )
    relations: RelationUpsertResult = Field(
        default_factory=RelationUpsertResult, description="Relation upsert counts"
    )
    chunks_embedded: int = Field(default=0, description="Chunks that were vectorized")
    vectors_indexed: int = Field(
        default=0, description="Chunks successfully indexed in vector search"
    )
    canonical_entities: dict[str, CanonicalNode] = Field(
        default_factory=dict, description="Resolved entities for tracing (candidate_id → node)"
    )
    canonical_relations: dict[str, CanonicalRelation] = Field(
        default_factory=dict, description="Resolved relations for tracing"
    )
    resolution_metadata: ResolutionMetadata = Field(
        default_factory=ResolutionMetadata, description="Entity resolution statistics"
    )
    errors: list[dict] = Field(
        default_factory=list,
        description="Non-fatal errors during processing (chunk embedding failed, etc.)",
    )

    @property
    def total_nodes_written(self) -> int:
        """Total nodes created or updated."""
        return (
            self.nodes.entities_created
            + self.nodes.entities_updated
            + self.nodes.chunks_created
            + self.nodes.chunks_updated
            + self.nodes.documents_created
            + self.nodes.documents_updated
        )

    @property
    def total_relations_written(self) -> int:
        """Total relations created or updated."""
        return (
            self.relations.mention_relations_created
            + self.relations.mention_relations_updated
            + self.relations.content_relations_created
            + self.relations.content_relations_updated
            + self.relations.from_relations_created
        )


# ─────────────────────────────────────────────────────────────────────────────
# Configuration Models (for reference; actual config loaded from _config/)
# ─────────────────────────────────────────────────────────────────────────────


class Neo4jConfig(BaseModel):
    """Neo4j connection and operation parameters."""

    uri: str = Field(default="bolt://localhost:7687", description="Bolt URI")
    user: str = Field(default="neo4j", description="Username")
    password: str = Field(default="", description="Password (from .env)")
    batch_size: int = Field(default=500, description="Batch size for upsert transactions")
    max_retries: int = Field(default=3, description="Retry count on transient failures")
    timeout: int = Field(default=30, description="Query timeout in seconds")


class EmbeddingConfig(BaseModel):
    """Chunk embedding parameters."""

    model: str = Field(
        default="text-embedding-3-large",
        description="Model ID (OpenAI or Anthropic)",
    )
    provider: str = Field(
        default="openai", description="Provider: openai or anthropic"
    )
    batch_size: int = Field(default=100, description="Chunks per API call")
    dimensions: int = Field(default=1536, description="Expected vector dimension")
    cache_embeddings: bool = Field(
        default=False, description="Cache embeddings to avoid re-embedding"
    )


class ResolutionConfig(BaseModel):
    """Entity resolution parameters."""

    fuzzy_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Edit distance threshold for fuzzy matching",
    )
    rules_file: Optional[str] = Field(
        default=None, description="Path to JSON file with domain-specific rules"
    )
    auto_merge_fuzzy: bool = Field(
        default=False,
        description="Auto-merge fuzzy candidates or flag for review",
    )
    semantic_similarity_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Embedding similarity threshold for semantic merging (future)",
    )


class GraphIndexingConfig(BaseModel):
    """Stage 2 configuration aggregator."""

    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    resolution: ResolutionConfig = Field(default_factory=ResolutionConfig)
    apoc_enabled: bool = Field(default=False, description="Run APOC enrichment")
    apoc_algorithms: list[str] = Field(
        default_factory=lambda: ["lpa"],
        description="APOC algorithms to run: lpa (community detection), pagerank, etc.",
    )
