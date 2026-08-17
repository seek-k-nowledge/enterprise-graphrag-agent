# Stage 4 — FastAPI Service & HTTP Interface

**Not yet implemented.** This document establishes the contract. Read the root [`../../CONTEXT.md`](../../CONTEXT.md) first for the pipeline-wide picture.

---

## Mandate

Provide the HTTP surface for the entire system. Accept user queries, trigger ingestions, and return results. Auth, rate limiting, error handling, and async job management all live here.

**This stage owns:** FastAPI application, routers, request/response validation, auth, rate limiting, async job queues, health checks, streaming responses, structured logging.

**This stage does not own:** reasoning, Cypher, Neo4j writes, entity resolution, embedding, or vector search. It is a thin wrapper that validates and delegates. Business logic lives in Stages 1-3.

**The key constraint** (restated from root): no business logic in route handlers. Routes validate, call stages 1-3, and shape responses. If a route handler contains reasoning, resolution logic, or database writes, it belongs in an earlier stage.

---

## API Contract

### Query Endpoint (Synchronous)

```http
POST /api/v1/query
Content-Type: application/json

{
  "query": "How does company X relate to person Y?",
  "context": "Optional background information",
  "user_id": "user@example.com"
}
```

**Response:**
```json
{
  "answer": "answer text here",
  "citations": [
    {
      "claim": "claim text",
      "source_type": "node|edge|chunk",
      "source_id": "entity:id",
      "source_text": "...",
      "confidence": 0.95
    }
  ],
  "retrieval_trace": {
    "query_classification": "multi_hop",
    "retrieval_strategies": ["graph_traversal", "vector_search"],
    "subgraph_stats": {"nodes_touched": 42, "edges_touched": 18, "chunks_used": 5},
    "reasoning_steps": [...],
    "errors": []
  },
  "confidence": 0.85,
  "token_usage": {"input_tokens": 150, "output_tokens": 280, "total_tokens": 430},
  "latency_ms": 2345,
  "request_id": "req_xyz123"
}
```

### Query Endpoint (Streaming)

```http
POST /api/v1/query/stream
```

Returns `text/event-stream` with Server-Sent Events (SSE):
```
data: {"type": "reasoning_step", "step": "Classifying query..."}
data: {"type": "retrieval", "strategy": "graph_traversal", "entities_found": 5}
data: {"type": "synthesis", "partial_answer": "Based on..."}
data: {"type": "verification", "valid": true}
data: {"type": "complete", "answer": {...full response...}}
```

### Ingestion Endpoint (Async Job)

```http
POST /api/v1/ingest
Content-Type: application/json

{
  "document_url": "s3://bucket/document.pdf",
  "document_text": "raw text (if not using URL)",
  "source_id": "doc_12345",
  "priority": "normal|high"
}
```

**Response:**
```json
{
  "job_id": "job_abc123",
  "status": "queued",
  "message": "Document queued for ingestion"
}
```

### Job Status Endpoint

```http
GET /api/v1/jobs/{job_id}
```

**Response:**
```json
{
  "job_id": "job_abc123",
  "status": "processing|completed|failed",
  "progress": {"current": 50, "total": 100},
  "result": {...extraction_result...},
  "error": null,
  "created_at": "2026-03-15T10:00:00Z",
  "completed_at": "2026-03-15T10:05:00Z"
}
```

### Health & Readiness

```http
GET /health
```

**Response:** `{"status": "healthy"}`

```http
GET /ready
```

**Response:** `{"ready": true, "neo4j": "connected", "stages": {"extraction": "ok", ...}}`

---

## Architecture

```
FastAPI app
  ├── Middleware
  │   ├── Auth (API key, JWT, etc.)
  │   ├── Rate limiting (per user, per IP)
  │   ├── Request logging (structured JSON)
  │   └── Error handling & mapping
  │
  ├── Routers
  │   ├── /api/v1/query - sync query endpoint
  │   ├── /api/v1/query/stream - streaming queries
  │   ├── /api/v1/ingest - async ingestion
  │   ├── /api/v1/jobs/{job_id} - job status
  │   ├── /health - health check
  │   └── /ready - readiness probe
  │
  ├── Background tasks
  │   └── Job queue worker (processes ingestion jobs)
  │
  └── Dependencies
      ├── Neo4jClient (Stage 2)
      ├── GraphAccessor (Stage 3)
      ├── AgentGraph (Stage 3)
      ├── EntityResolver (Stage 2)
      ├── ExtractionConfig (Stage 1)
      └── Job queue (Redis or in-memory)
```

---

## Request/Response Models

### Query Request

```python
class QueryRequest(BaseModel):
    """User query request."""
    query: str
    context: Optional[str] = None
    user_id: Optional[str] = None
    prefer_strategies: Optional[list[str]] = None
    timeout_sec: Optional[int] = 60
```

### Query Response

```python
class QueryResponse(BaseModel):
    """Complete query response."""
    answer: str
    citations: list[Citation]
    retrieval_trace: RetrievalTrace
    confidence: float
    token_usage: TokenUsage
    latency_ms: int
    request_id: str
    gaps: list[str] = []
```

### Ingestion Request

```python
class IngestionRequest(BaseModel):
    """Ingestion job request."""
    document_url: Optional[str] = None
    document_text: Optional[str] = None
    source_id: str
    priority: str = "normal"  # normal, high, low
```

### Job Status Response

```python
class JobStatus(BaseModel):
    """Job status response."""
    job_id: str
    status: str  # queued, processing, completed, failed
    progress: Optional[dict] = None
    result: Optional[dict] = None
    error: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
```

---

## Async Job Handling

### Job Queue

Use a queue (Redis, Celery, or in-memory for MVP) to manage ingestion jobs:

```
User → POST /ingest → Create job_id → Queue job → Return job_id
                                          ↓
                              Background worker
                                  ↓
                      Stage 1: Extract (slow)
                                  ↓
                      Stage 2: Write to Neo4j (idempotent)
                                  ↓
                      Store result & job status
                                  ↓
                      User: GET /jobs/{job_id}
```

**Job status workflow:**
- `queued`: in the queue, waiting to execute
- `processing`: actively running (extracting)
- `completed`: finished, result available
- `failed`: error occurred, stored for debugging

### Streaming Queries

For long-running queries, stream intermediate results via Server-Sent Events (SSE):

```javascript
const eventSource = new EventSource('/api/v1/query/stream?query=...');
eventSource.onmessage = (event) => {
  const message = JSON.parse(event.data);
  if (message.type === 'reasoning_step') {
    console.log('Agent step:', message.step);
  } else if (message.type === 'complete') {
    console.log('Answer:', message.answer.answer_text);
  }
};
```

---

## Auth & Rate Limiting

### Auth

Support multiple auth schemes:
- **API Key:** `Authorization: Bearer api_key_xyz`
- **JWT:** `Authorization: Bearer eyJ...` (token with claims)
- **Session:** Cookie-based (for browser clients)

Extract `user_id` from auth context and pass to Stages 2-3 for audit trails.

### Rate Limiting

Apply per-user and per-IP limits:
- Queries: 100/minute per user, 1000/minute per IP
- Ingestions: 10/minute per user, 100/minute per IP
- Return `429 Too Many Requests` with `Retry-After` header

---

## Error Handling

Map internal errors to HTTP status codes:

| Internal Error | HTTP Status | Response |
|---|---|---|
| Validation error | 400 | `{"error": "Invalid query", "details": [...]}` |
| Auth failure | 401 | `{"error": "Unauthorized"}` |
| Permission denied | 403 | `{"error": "Forbidden"}` |
| Not found | 404 | `{"error": "Resource not found"}` |
| Rate limit | 429 | `{"error": "Too many requests", "retry_after": 60}` |
| Query timeout | 504 | `{"error": "Query timeout", "details": "..."}` |
| Neo4j unavailable | 503 | `{"error": "Service unavailable"}` |
| Internal error | 500 | `{"error": "Internal server error", "request_id": "..."}` |

---

## Structured Logging

All requests logged as JSON:

```json
{
  "timestamp": "2026-03-15T10:00:00Z",
  "request_id": "req_xyz123",
  "method": "POST",
  "path": "/api/v1/query",
  "user_id": "user@example.com",
  "status": 200,
  "latency_ms": 2345,
  "query": "How does X relate to Y?",
  "confidence": 0.85,
  "tokens": 430,
  "retrieval_strategies": ["graph_traversal", "vector_search"],
  "errors": []
}
```

---

## Configuration

| Key | Purpose |
|---|---|
| `fastapi.host` | Bind address (default 0.0.0.0) |
| `fastapi.port` | Listen port (default 8000) |
| `fastapi.workers` | Number of workers (default 4) |
| `fastapi.log_level` | Logging level (default INFO) |
| `auth.scheme` | Auth type: api_key, jwt, session |
| `auth.jwt_secret` | JWT signing secret |
| `ratelimit.query_per_minute` | Queries per minute per user (default 100) |
| `ratelimit.ingest_per_minute` | Ingestions per minute per user (default 10) |
| `job_queue.backend` | redis, celery, or memory (default memory for MVP) |
| `job_queue.max_workers` | Parallel ingestion workers (default 4) |

---

## Module Layout

```
stages/fastapi_service/
├── CONTEXT.md              this file
├── __init__.py             FastAPI app factory
├── main.py                 app startup and configuration
├── schemas.py              Pydantic request/response models
├── routers/                endpoint implementations
│   ├── __init__.py
│   ├── query.py            query endpoint (sync + streaming)
│   ├── ingest.py           ingestion trigger endpoint
│   ├── jobs.py             job status endpoint
│   └── health.py           health & readiness checks
├── middleware/             request/response middleware
│   ├── __init__.py
│   ├── auth.py             authentication
│   ├── ratelimit.py        rate limiting
│   └── logging.py          structured request logging
├── jobs/                   async job handling
│   ├── __init__.py
│   ├── queue.py            job queue abstraction
│   ├── worker.py           background ingestion worker
│   └── models.py           job status models
└── config.py               configuration loading
```

---

## Testing

- **Unit tests:** routers with mocked backends, middleware
- **Integration tests:** with real Neo4j and Stages 2-3
- **Load tests:** concurrent queries, rate limit enforcement
- **Job queue tests:** job lifecycle, retry logic, error handling

---

## Deployment

- **Docker:** Dockerfile with uvicorn + gunicorn for production
- **Health checks:** `/health` and `/ready` endpoints for orchestrators
- **Graceful shutdown:** finish in-flight requests before terminating
- **Monitoring:** Prometheus metrics, structured logging, error tracking

---

## Open Questions

1. **Job persistence:** should completed jobs be kept indefinitely, or garbage-collected after N days?
2. **Streaming format:** SSE vs. WebSocket vs. simple polling?
3. **Authentication:** API keys, JWT, OAuth2, or all three?
4. **Result caching:** cache query results by query hash? TTL?
5. **Batch ingestion:** accept multiple documents in one request?
