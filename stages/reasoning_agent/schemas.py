"""
Pydantic v2 models for Stage 3: Reasoning Agent & Query Orchestration.

Payload contract:
- Input: QueryPayload (user query with optional context)
- Output: AnswerPayload (grounded answer with full citations and trace)
- Internal: Agent state, retrieval results, intermediate models
"""

from datetime import datetime
from typing import Optional, Any
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Input Payload
# ─────────────────────────────────────────────────────────────────────────────


class QueryPayload(BaseModel):
    """User query with optional context and session tracking."""

    text: str = Field(..., description="The question or query text")
    context: Optional[str] = Field(
        default=None, description="Optional background context or conversation history"
    )
    user_id: Optional[str] = Field(default=None, description="User identifier for tracking")
    session_id: Optional[str] = Field(
        default=None, description="Session ID for checkpoint resumption"
    )
    max_hops: Optional[int] = Field(
        default=3, description="Override default max hops for traversal (1-5)"
    )
    top_k: Optional[int] = Field(
        default=10, description="Override default top-k for vector search"
    )
    prefer_strategies: Optional[list[str]] = Field(
        default=None,
        description="Preference order for retrieval strategies: graph_traversal, vector_search, cypher_direct",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Graph Element Models (from Stage 2)
# ─────────────────────────────────────────────────────────────────────────────


class GraphEntity(BaseModel):
    """Entity node from the Neo4j graph."""

    id: str = Field(..., description="Canonical entity ID (entity_type:name)")
    entity_type: str = Field(..., description="Entity type (Person, Organization, Location, etc.)")
    canonical_name: str = Field(..., description="Authoritative name")
    surface_forms: list[str] = Field(
        default_factory=list, description="All variations seen"
    )
    description: str = Field(default="", description="Entity description")
    sources: list[str] = Field(
        default_factory=list, description="Document IDs mentioning this entity"
    )


class GraphRelation(BaseModel):
    """Relationship edge from the Neo4j graph."""

    source_id: str = Field(..., description="Source entity ID")
    target_id: str = Field(..., description="Target entity ID")
    relation_type: str = Field(..., description="Edge label (IS_A, WORKS_AT, etc.)")
    description: str = Field(default="", description="Relation description")
    evidence: list[str] = Field(
        default_factory=list, description="Verbatim spans supporting this relation"
    )
    supporting_chunks: list[str] = Field(
        default_factory=list, description="Chunk IDs providing evidence"
    )
    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Support breadth (0-1)"
    )


class GraphChunk(BaseModel):
    """Chunk/passage from the Neo4j graph."""

    id: str = Field(..., description="Chunk ID (document_id:index)")
    text: str = Field(..., description="Chunk text")
    start_char: int = Field(..., description="Character offset in document")
    end_char: int = Field(..., description="End offset in document")
    document_id: str = Field(..., description="Source document ID")
    embedding: Optional[list[float]] = Field(
        default=None, description="Vector embedding (if present)"
    )


class Subgraph(BaseModel):
    """A subgraph result: entities, relations, and supporting chunks."""

    entities: dict[str, GraphEntity] = Field(
        default_factory=dict, description="Entities in the subgraph (id → entity)"
    )
    relations: dict[str, GraphRelation] = Field(
        default_factory=dict,
        description="Relations in the subgraph (source_id:target_id:type → relation)",
    )
    chunks: dict[str, GraphChunk] = Field(
        default_factory=dict, description="Chunks in the subgraph (id → chunk)"
    )

    @property
    def entity_count(self) -> int:
        return len(self.entities)

    @property
    def relation_count(self) -> int:
        return len(self.relations)

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval Results
# ─────────────────────────────────────────────────────────────────────────────


class RetrievalStrategyResult(BaseModel):
    """Result from one retrieval strategy."""

    strategy: str = Field(
        ..., description="Strategy name (graph_traversal, vector_search, cypher_direct)"
    )
    subgraph: Subgraph = Field(default_factory=Subgraph, description="Retrieved subgraph")
    rank: int = Field(default=0, description="Ranking among strategies (0=best)")
    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Overall confidence in this result"
    )
    explanation: str = Field(default="", description="Why this strategy was chosen/executed")
    error: Optional[str] = Field(default=None, description="Error message if strategy failed")


class RetrievalResult(BaseModel):
    """Results from all retrieval strategies, ranked."""

    query: str = Field(..., description="The original query")
    strategies_executed: list[str] = Field(
        default_factory=list, description="Strategies that were run"
    )
    strategy_results: list[RetrievalStrategyResult] = Field(
        default_factory=list, description="Results ranked by confidence"
    )
    merged_subgraph: Subgraph = Field(
        default_factory=Subgraph, description="Merged results from all strategies"
    )
    total_entities: int = Field(default=0, description="Total entities retrieved")
    total_relations: int = Field(default=0, description="Total relations retrieved")
    total_chunks: int = Field(default=0, description="Total chunks retrieved")


# ─────────────────────────────────────────────────────────────────────────────
# Citation & Grounding
# ─────────────────────────────────────────────────────────────────────────────


class Citation(BaseModel):
    """A source for a claim in the answer."""

    claim: str = Field(..., description="The specific text being cited")
    source_type: str = Field(
        ..., description="Source type: node, edge, or chunk"
    )
    source_id: str = Field(..., description="Neo4j entity ID")
    source_text: str = Field(..., description="Content of the cited entity")
    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0, description="How strongly this supports the claim"
    )
    explanation: str = Field(
        default="", description="Why this source supports the claim"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Token & Performance Tracking
# ─────────────────────────────────────────────────────────────────────────────


class TokenUsage(BaseModel):
    """Token usage by model."""

    input_tokens: int = Field(default=0, description="Total input tokens")
    output_tokens: int = Field(default=0, description="Total output tokens")
    total_tokens: int = Field(default=0, description="Sum of input + output")
    by_model: dict[str, dict] = Field(
        default_factory=dict,
        description="Breakdown by model: {model_name: {input, output, total}}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval Trace & Reasoning Steps
# ─────────────────────────────────────────────────────────────────────────────


class ReasoningStep(BaseModel):
    """One step in the agent reasoning process."""

    step_type: str = Field(
        ..., description="Step type: query_classification, retrieval, synthesis, verification"
    )
    agent: str = Field(..., description="Agent name (router, retrieval, etc.)")
    input: dict = Field(default_factory=dict, description="Input to this step")
    output: dict = Field(default_factory=dict, description="Output from this step")
    reasoning: str = Field(default="", description="Why this decision was made")
    duration_ms: int = Field(default=0, description="Execution time in milliseconds")


class RetrievalTrace(BaseModel):
    """Full audit trail of retrieval and reasoning."""

    query_classification: str = Field(
        ...,
        description="Query type: multi_hop, semantic, aggregate, hybrid",
    )
    retrieval_strategies: list[str] = Field(
        default_factory=list, description="Strategies used"
    )
    strategies_skipped: list[str] = Field(
        default_factory=list, description="Strategies not applicable or skipped"
    )
    subgraph_stats: dict = Field(
        default_factory=dict,
        description="Subgraph size: {nodes_touched, edges_touched, chunks_used}",
    )
    reasoning_steps: list[ReasoningStep] = Field(
        default_factory=list, description="Intermediate agent decisions"
    )
    errors: list[dict] = Field(
        default_factory=list, description="Errors or fallbacks during retrieval"
    )
    total_duration_ms: int = Field(
        default=0, description="Total retrieval time in milliseconds"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Output Payload
# ─────────────────────────────────────────────────────────────────────────────


class AnswerPayload(BaseModel):
    """Grounded answer with full provenance."""

    query: str = Field(..., description="The original user query")
    answer_text: str = Field(..., description="The synthesized answer")
    citations: list[Citation] = Field(
        default_factory=list, description="Citation for each claim"
    )
    retrieval_trace: RetrievalTrace = Field(
        default_factory=RetrievalTrace, description="Full reasoning audit trail"
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Overall answer confidence based on citation coverage",
    )
    token_usage: TokenUsage = Field(
        default_factory=TokenUsage, description="Token usage breakdown"
    )
    latency_ms: int = Field(default=0, description="End-to-end execution time")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="When answer was generated"
    )
    user_id: Optional[str] = Field(default=None, description="User ID for tracking")
    session_id: Optional[str] = Field(default=None, description="Session ID for continuity")
    gaps: list[str] = Field(
        default_factory=list,
        description="Known gaps or limitations in the answer (from verification agent)",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Agent State (LangGraph)
# ─────────────────────────────────────────────────────────────────────────────


class AgentState(BaseModel):
    """Shared state for LangGraph agent swarm."""

    query: QueryPayload = Field(..., description="Original user query")
    query_classification: Optional[str] = Field(
        default=None, description="Query type from router"
    )
    preferred_strategies: list[str] = Field(
        default_factory=list, description="Strategies selected by router"
    )
    retrieval_results: Optional[RetrievalResult] = Field(
        default=None, description="Results from retrieval strategies"
    )
    draft_answer: Optional[str] = Field(
        default=None, description="Answer draft from synthesis agent"
    )
    citations: list[Citation] = Field(
        default_factory=list, description="Citations from synthesis"
    )
    verification_feedback: Optional[dict] = Field(
        default=None, description="Feedback from verification agent"
    )
    reasoning_steps: list[ReasoningStep] = Field(
        default_factory=list, description="All reasoning steps for audit trail"
    )
    token_usage: TokenUsage = Field(
        default_factory=TokenUsage, description="Accumulated token usage"
    )
    errors: list[dict] = Field(
        default_factory=list, description="Non-fatal errors during processing"
    )

    # Metadata for timing
    started_at: datetime = Field(default_factory=datetime.utcnow)


class RouterDecision(BaseModel):
    """Decision from the query router agent."""

    query_classification: str = Field(...)
    strategies: list[str] = Field(description="Ordered list of strategies to execute")
    seed_entities: list[str] = Field(
        default_factory=list, description="Named entities extracted from query"
    )
    parameters: dict = Field(
        default_factory=dict, description="Strategy-specific parameters"
    )
    reasoning: str = Field(default="")


class SynthesisOutput(BaseModel):
    """Output from the synthesis agent."""

    answer_text: str = Field(...)
    citations: list[Citation] = Field(default_factory=list)
    cited_entities: list[str] = Field(default_factory=list)
    cited_relations: list[str] = Field(default_factory=list)
    reasoning: str = Field(default="")


class VerificationOutput(BaseModel):
    """Output from the verification agent."""

    valid: bool = Field(default=True, description="Is the answer valid?")
    consistency_issues: list[str] = Field(
        default_factory=list, description="Logical inconsistencies found"
    )
    citation_issues: list[str] = Field(
        default_factory=list, description="Citation problems (missing, invalid, etc.)"
    )
    gaps: list[str] = Field(
        default_factory=list, description="Known gaps or limitations"
    )
    confidence_adjustment: float = Field(
        default=0.0, description="Adjustment to answer confidence (-1 to 1)"
    )
    reasoning: str = Field(default="")


# ─────────────────────────────────────────────────────────────────────────────
# Configuration Models
# ─────────────────────────────────────────────────────────────────────────────


class ReasoningConfig(BaseModel):
    """Stage 3 configuration."""

    retrieval_strategies: list[str] = Field(
        default_factory=lambda: ["graph_traversal", "vector_search"],
        description="Enabled retrieval strategies",
    )
    query_classifier_model: str = Field(
        default="claude-sonnet-5", description="Model for query classification"
    )
    cypher_writer_model: str = Field(
        default="claude-sonnet-5", description="Model for Cypher authoring"
    )
    synthesis_model: str = Field(
        default="claude-sonnet-5", description="Model for answer synthesis"
    )
    max_traversal_hops: int = Field(default=3, ge=1, le=5)
    vector_search_top_k: int = Field(default=10, ge=1, le=100)
    min_citation_confidence: float = Field(default=0.3, ge=0.0, le=1.0)
    enable_checkpointing: bool = Field(default=True)
    checkpoint_backend: str = Field(default="sqlite", description="sqlite or postgres")
    query_timeout_sec: int = Field(default=30, ge=5, le=300)
    enable_caching: bool = Field(default=True)
    cache_ttl_sec: int = Field(default=3600)
