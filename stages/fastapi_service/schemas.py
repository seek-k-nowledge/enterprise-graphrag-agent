"""
Pydantic v2 models for Stage 4: FastAPI Service.

Request/response models for all HTTP endpoints.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Query Endpoints
# ─────────────────────────────────────────────────────────────────────────────


class QueryRequest(BaseModel):
    """User query request."""

    query: str = Field(..., description="The question to answer")
    context: Optional[str] = Field(
        default=None, description="Optional background context"
    )
    user_id: Optional[str] = Field(default=None, description="User identifier")
    prefer_strategies: Optional[list[str]] = Field(
        default=None,
        description="Preferred retrieval strategies (graph_traversal, vector_search, cypher_direct)",
    )
    timeout_sec: Optional[int] = Field(
        default=60, ge=10, le=300, description="Query timeout in seconds"
    )


class CitationModel(BaseModel):
    """Citation for a claim in the answer."""

    claim: str = Field(..., description="The claim being cited")
    source_type: str = Field(..., description="Source type: node, edge, or chunk")
    source_id: str = Field(..., description="Neo4j entity ID")
    source_text: str = Field(..., description="Content of the source")
    confidence: float = Field(ge=0.0, le=1.0, description="Citation confidence")


class TokenUsageModel(BaseModel):
    """Token usage breakdown."""

    input_tokens: int = Field(default=0)
    output_tokens: int = Field(default=0)
    total_tokens: int = Field(default=0)


class RetrievalTraceModel(BaseModel):
    """Audit trail of retrieval and reasoning."""

    query_classification: str = Field(...)
    retrieval_strategies: list[str] = Field(default_factory=list)
    subgraph_stats: dict = Field(
        default_factory=dict,
        description="nodes_touched, edges_touched, chunks_used",
    )
    reasoning_steps: list[dict] = Field(default_factory=list)
    errors: list[dict] = Field(default_factory=list)


class QueryResponse(BaseModel):
    """Complete query response with answer and citations."""

    answer: str = Field(..., description="The synthesized answer")
    citations: list[CitationModel] = Field(
        default_factory=list, description="Citations for each claim"
    )
    retrieval_trace: Optional[RetrievalTraceModel] = Field(
        default=None, description="Audit trail (optional, use ?trace=true)"
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Overall answer confidence"
    )
    token_usage: TokenUsageModel = Field(
        default_factory=TokenUsageModel, description="Token usage"
    )
    latency_ms: int = Field(ge=0, description="End-to-end latency in milliseconds")
    request_id: str = Field(..., description="Unique request identifier")
    gaps: list[str] = Field(
        default_factory=list, description="Known gaps or limitations"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Streaming Events
# ─────────────────────────────────────────────────────────────────────────────


class StreamingEvent(BaseModel):
    """A single event in a streaming response."""

    type: str = Field(
        ...,
        description="Event type: reasoning_step, retrieval, synthesis, verification, complete, error",
    )
    data: dict = Field(default_factory=dict, description="Event-specific data")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ─────────────────────────────────────────────────────────────────────────────
# Ingestion Endpoints
# ─────────────────────────────────────────────────────────────────────────────


class IngestionRequest(BaseModel):
    """Ingestion job request."""

    document_url: Optional[str] = Field(
        default=None, description="URL to fetch document (s3://, http://, etc.)"
    )
    document_text: Optional[str] = Field(
        default=None, description="Raw document text (if not using URL)"
    )
    source_id: str = Field(..., description="Source identifier")
    priority: str = Field(
        default="normal",
        description="Job priority: low, normal, or high",
    )

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "document_url": "s3://bucket/document.pdf",
                    "source_id": "doc_12345",
                    "priority": "normal",
                },
                {
                    "document_text": "The quick brown fox...",
                    "source_id": "doc_12346",
                    "priority": "high",
                },
            ]
        }


class IngestionResponse(BaseModel):
    """Response to ingestion request."""

    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(default="queued", description="Job status")
    message: str = Field(default="Document queued for ingestion")
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ─────────────────────────────────────────────────────────────────────────────
# Job Status Endpoints
# ─────────────────────────────────────────────────────────────────────────────


class JobProgress(BaseModel):
    """Job progress indicator."""

    current: int = Field(ge=0, description="Current progress")
    total: int = Field(ge=0, description="Total steps")

    @property
    def percentage(self) -> float:
        """Return progress as percentage."""
        if self.total == 0:
            return 0.0
        return (self.current / self.total) * 100


class JobStatus(BaseModel):
    """Job status response."""

    job_id: str = Field(..., description="Job identifier")
    status: str = Field(
        ...,
        description="Job status: queued, processing, completed, or failed",
    )
    progress: Optional[JobProgress] = Field(
        default=None, description="Progress indicator"
    )
    result: Optional[dict] = Field(
        default=None, description="Job result (when completed)"
    )
    error: Optional[str] = Field(default=None, description="Error message if failed")
    created_at: datetime = Field(..., description="Job creation timestamp")
    started_at: Optional[datetime] = Field(
        default=None, description="When job started processing"
    )
    completed_at: Optional[datetime] = Field(
        default=None, description="When job completed"
    )

    @property
    def duration_sec(self) -> Optional[float]:
        """Return job duration in seconds."""
        if not self.completed_at or not self.created_at:
            return None
        delta = self.completed_at - self.created_at
        return delta.total_seconds()


# ─────────────────────────────────────────────────────────────────────────────
# Health & Status Endpoints
# ─────────────────────────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(default="healthy", description="Overall status")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class StageStatus(BaseModel):
    """Status of a single pipeline stage."""

    name: str = Field(...)
    status: str = Field(..., description="ok, degraded, or unavailable")
    details: Optional[str] = Field(default=None)


class ReadinessResponse(BaseModel):
    """Readiness probe response."""

    ready: bool = Field(..., description="Is service ready to accept traffic?")
    neo4j: str = Field(..., description="Neo4j connection status")
    stages: dict[str, StageStatus] = Field(
        default_factory=dict, description="Status of each pipeline stage"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ─────────────────────────────────────────────────────────────────────────────
# Error Responses
# ─────────────────────────────────────────────────────────────────────────────


class ErrorDetail(BaseModel):
    """Detailed error information."""

    field: Optional[str] = Field(default=None, description="Field name (if validation error)")
    message: str = Field(..., description="Error message")
    code: Optional[str] = Field(default=None, description="Error code")


class ErrorResponse(BaseModel):
    """Error response."""

    error: str = Field(..., description="Error message")
    details: Optional[list[ErrorDetail]] = Field(
        default=None, description="Detailed error info"
    )
    request_id: Optional[str] = Field(default=None, description="Request ID for logging")
    retry_after: Optional[int] = Field(
        default=None, description="Seconds to wait before retrying (for 429)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Configuration Models
# ─────────────────────────────────────────────────────────────────────────────


class FastAPIConfig(BaseModel):
    """FastAPI configuration."""

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1, le=65535)
    workers: int = Field(default=4, ge=1, le=32)
    log_level: str = Field(default="INFO")
    title: str = Field(default="Enterprise GraphRAG")
    version: str = Field(default="1.0.0")


class AuthConfig(BaseModel):
    """Authentication configuration."""

    scheme: str = Field(default="api_key", description="api_key, jwt, or session")
    jwt_secret: Optional[str] = Field(default=None)
    jwt_algorithm: str = Field(default="HS256")


class RateLimitConfig(BaseModel):
    """Rate limiting configuration."""

    query_per_minute: int = Field(default=100)
    ingest_per_minute: int = Field(default=10)
    global_per_second: int = Field(default=1000)


class JobQueueConfig(BaseModel):
    """Job queue configuration."""

    backend: str = Field(default="memory", description="memory, redis, or celery")
    max_workers: int = Field(default=4, ge=1, le=16)
    max_retries: int = Field(default=3)
    timeout_sec: int = Field(default=3600)


class ServiceConfig(BaseModel):
    """Overall service configuration."""

    fastapi: FastAPIConfig = Field(default_factory=FastAPIConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    ratelimit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    job_queue: JobQueueConfig = Field(default_factory=JobQueueConfig)
