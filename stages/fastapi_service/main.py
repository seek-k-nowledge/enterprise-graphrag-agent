"""
FastAPI application factory and configuration for Stage 4.

Creates the FastAPI app with all routers, middleware, and dependencies.
"""

import logging
import uuid
from typing import Optional
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import sys

# Load environment variables from .env file
load_dotenv()

# Configure logging to capture all extraction errors
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)

from .schemas import (
    QueryRequest,
    QueryResponse,
    IngestionRequest,
    IngestionResponse,
    JobStatus,
    ErrorResponse,
    HealthResponse,
    ReadinessResponse,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Global State (would be dependency injected in production)
# ─────────────────────────────────────────────────────────────────────────────

app_state = {
    "neo4j_client": None,
    "graph_accessor": None,
    "agent_graph": None,
    "graph_writer": None,
    "job_queue": {},  # In-memory job store for MVP
    "ready": False,
}


# ─────────────────────────────────────────────────────────────────────────────
# Ingestion Processing
# ─────────────────────────────────────────────────────────────────────────────

def process_ingestion(job_id: str, ingest_req: IngestionRequest) -> None:
    """Process a queued ingestion job: extract document and write to graph."""
    if job_id not in app_state["job_queue"]:
        return

    job_info = app_state["job_queue"][job_id]
    job = job_info["job"]

    try:
        # Update job status
        job.status = "processing"
        logger.info(f"Processing ingestion: {job_id}")

        # Extract document
        from stages.extraction.extractor import extract_document, ExtractionConfig

        config = ExtractionConfig(
            model="claude-haiku-4-5-20251001" if ingest_req.llm_provider == "anthropic" else "gpt-oss-120b",
            entity_types=["Person", "Organization", "Location", "Product", "Event"],
            relation_types=["WORKS_AT", "LOCATED_IN", "CREATED", "RELATED_TO"],
            llm_provider=ingest_req.llm_provider,
            llm_api_key=ingest_req.llm_api_key,
        )

        extraction_result = extract_document(
            text=ingest_req.document_text,
            uri=ingest_req.source_id,
            config=config,
            title=ingest_req.source_id,
        )

        logger.info(
            f"Extraction complete: {len(extraction_result.entities)} entities, "
            f"{len(extraction_result.relations)} relations, "
            f"{len(extraction_result.chunks)} chunks"
        )

        # Write to Neo4j
        if app_state["neo4j_client"] and app_state["graph_writer"]:
            from stages.graph_indexing.entity_resolver import EntityResolver
            from stages.graph_indexing.graph_writer import GraphWriter

            # Resolve entities
            resolver = EntityResolver()
            entity_resolution = resolver.resolve(
                extraction_result.entities,
                document_id=ingest_req.source_id
            )
            canonical_entities = entity_resolution.canonical_entities
            canonical_relations = resolver.resolve_relations(
                extraction_result.relations,
                entity_resolution.candidate_to_canonical
            )

            logger.info(f"Entity resolution: {len(canonical_entities)} canonical entities")

            # Write to graph
            writer = app_state["graph_writer"]
            node_result, rel_result = writer.write_extraction(
                extraction_result,
                canonical_entities,
                canonical_relations,
            )

            logger.info(
                f"Graph write complete: {node_result.entities_created} entities, "
                f"{rel_result.content_relations_created} relations, "
                f"{node_result.chunks_created} chunks"
            )

        # Update job status
        job.status = "completed"
        job.result = {
            "entities_created": extraction_result.__dict__.get("entities", []),
            "chunks_created": len(extraction_result.chunks),
            "relations_created": extraction_result.__dict__.get("relations", []),
        }

        logger.info(f"Ingestion completed: {job_id}")

    except Exception as e:
        error_msg = str(e)
        logger.exception(f"Ingestion processing failed for {job_id}: {error_msg}")
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error(f"Full traceback above ^^")
        job.status = "failed"
        job.error = error_msg


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan (startup/shutdown)
# ─────────────────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown."""
    logger.info("Starting FastAPI service...")

    try:
        # Initialize dependencies (would load from config in production)
        from stages.graph_indexing import create_client as create_neo4j_client
        from stages.reasoning_agent import create_graph_accessor

        # Connect to Neo4j
        neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        neo4j_password = os.getenv("NEO4J_PASSWORD", "graphrag_dev_password")

        try:
            app_state["neo4j_client"] = create_neo4j_client(
                uri=neo4j_uri,
                user=neo4j_user,
                password=neo4j_password,
            )
            logger.info(f"Connected to Neo4j at {neo4j_uri}")
        except Exception as e:
            logger.warning(f"Could not connect to Neo4j: {e}")
            app_state["neo4j_client"] = None

        # Create graph accessor
        if app_state["neo4j_client"]:
            app_state["graph_accessor"] = create_graph_accessor(
                app_state["neo4j_client"]
            )
            logger.info("Created GraphAccessor")

            # Create graph writer for ingestion
            from stages.graph_indexing.graph_writer import GraphWriter
            app_state["graph_writer"] = GraphWriter(app_state["neo4j_client"])
            app_state["graph_writer"].setup_schema()
            logger.info("Created GraphWriter and set up Neo4j schema")

        app_state["ready"] = app_state["neo4j_client"] is not None
        logger.info(f"Service ready: {app_state['ready']}")

    except Exception as e:
        logger.error(f"Startup error: {e}")

    yield

    # Shutdown
    logger.info("Shutting down FastAPI service...")
    if app_state["neo4j_client"]:
        try:
            app_state["neo4j_client"].close()
            logger.info("Closed Neo4j connection")
        except Exception as e:
            logger.error(f"Error closing Neo4j: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Create FastAPI app
# ─────────────────────────────────────────────────────────────────────────────


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="Enterprise GraphRAG",
        description="Multi-agent GraphRAG engine with Neo4j and LangGraph",
        version="1.0.0",
        lifespan=lifespan,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Middleware
    # ─────────────────────────────────────────────────────────────────────────

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request ID
    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    # ─────────────────────────────────────────────────────────────────────────
    # Query Endpoint (Synchronous)
    # ─────────────────────────────────────────────────────────────────────────

    @app.post("/api/v1/query", response_model=QueryResponse)
    async def query(request: Request, query_req: QueryRequest) -> QueryResponse:
        """
        Answer a user query using the agent swarm.

        Returns a grounded answer with citations and reasoning trace.
        """
        request_id = request.state.request_id

        if not app_state["graph_accessor"]:
            raise HTTPException(
                status_code=503,
                detail="Neo4j unavailable",
            )

        try:
            from stages.reasoning_agent import answer_query
            from stages.reasoning_agent.schemas import QueryPayload

            # Convert request to internal payload
            payload = QueryPayload(
                text=query_req.query,
                context=query_req.context,
                user_id=query_req.user_id,
                max_hops=5,  # Default
                top_k=10,  # Default
            )

            # Answer query with optional provider/key override
            answer_payload = answer_query(
                payload,
                app_state["graph_accessor"],
                provider=query_req.llm_provider,
                api_key=query_req.llm_api_key,
            )

            # Convert to response (include trace only if requested)
            include_trace = request.query_params.get("trace") == "true"

            # Convert citations
            citations = [
                {
                    "claim": c.claim,
                    "source_type": c.source_type,
                    "source_id": c.source_id,
                    "source_text": c.source_text,
                    "confidence": c.confidence,
                }
                for c in answer_payload.citations
            ]

            # Build response
            response = QueryResponse(
                answer=answer_payload.answer_text,
                citations=citations,
                retrieval_trace=include_trace and answer_payload.retrieval_trace or None,
                confidence=answer_payload.confidence,
                token_usage={
                    "input_tokens": answer_payload.token_usage.input_tokens,
                    "output_tokens": answer_payload.token_usage.output_tokens,
                    "total_tokens": answer_payload.token_usage.total_tokens,
                },
                latency_ms=answer_payload.latency_ms,
                request_id=request_id,
                gaps=answer_payload.gaps,
            )

            logger.info(
                f"Query answered: {request_id} | "
                f"confidence={response.confidence:.2f} | "
                f"latency={response.latency_ms}ms"
            )

            return response

        except Exception as e:
            logger.error(f"Query error ({request_id}): {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # ─────────────────────────────────────────────────────────────────────────
    # Ingestion Endpoint (Async Job)
    # ─────────────────────────────────────────────────────────────────────────

    @app.post("/api/v1/ingest", response_model=IngestionResponse)
    async def ingest(request: Request, ingest_req: IngestionRequest) -> IngestionResponse:
        """
        Queue a document for ingestion.

        Returns immediately with a job ID. Check /api/v1/jobs/{job_id} for status.
        """
        request_id = request.state.request_id

        # Validate request
        if not ingest_req.document_url and not ingest_req.document_text:
            raise HTTPException(
                status_code=400,
                detail="Either document_url or document_text is required",
            )

        try:
            # Create job
            job_id = f"job_{uuid.uuid4().hex[:12]}"
            job = JobStatus(
                job_id=job_id,
                status="queued",
            )

            # Store job in queue
            app_state["job_queue"][job_id] = {
                "job": job,
                "request": ingest_req,
                "request_id": request_id,
            }

            logger.info(
                f"Ingestion queued: {job_id} | "
                f"source_id={ingest_req.source_id} | "
                f"priority={ingest_req.priority}"
            )

            # Process ingestion synchronously (for MVP; could be made async)
            try:
                process_ingestion(job_id, ingest_req)
                logger.info(f"Ingestion processed successfully: {job_id}")
            except Exception as process_error:
                logger.exception(f"Failed to process ingestion {job_id}: {process_error}")
                job.status = "failed"
                job.error = str(process_error)

            return IngestionResponse(
                job_id=job_id,
                status=job.status,
                message=f"Document ingestion {job.status}",
            )

        except Exception as e:
            logger.exception(f"Ingestion error ({request_id}): {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # ─────────────────────────────────────────────────────────────────────────
    # Job Status Endpoint
    # ─────────────────────────────────────────────────────────────────────────

    @app.get("/api/v1/jobs/{job_id}", response_model=JobStatus)
    async def get_job_status(request: Request, job_id: str) -> JobStatus:
        """Get the status of an ingestion job."""
        request_id = request.state.request_id

        if job_id not in app_state["job_queue"]:
            raise HTTPException(status_code=404, detail="Job not found")

        try:
            job_info = app_state["job_queue"][job_id]
            job = job_info["job"]

            logger.debug(f"Job status: {job_id} | {job.status}")
            return job

        except Exception as e:
            logger.error(f"Job status error ({request_id}): {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # ─────────────────────────────────────────────────────────────────────────
    # Health Check
    # ─────────────────────────────────────────────────────────────────────────

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """Health check endpoint."""
        return HealthResponse(status="healthy")

    # ─────────────────────────────────────────────────────────────────────────
    # Readiness Probe
    # ─────────────────────────────────────────────────────────────────────────

    @app.get("/ready", response_model=ReadinessResponse)
    async def readiness() -> ReadinessResponse:
        """Readiness probe endpoint."""
        neo4j_status = "connected" if app_state["neo4j_client"] else "disconnected"

        return ReadinessResponse(
            ready=app_state["ready"],
            neo4j=neo4j_status,
            stages={
                "extraction": {"name": "extraction", "status": "ok"},
                "graph_indexing": {
                    "name": "graph_indexing",
                    "status": "ok" if app_state["neo4j_client"] else "unavailable",
                },
                "reasoning_agent": {
                    "name": "reasoning_agent",
                    "status": "ok" if app_state["graph_accessor"] else "unavailable",
                },
                "fastapi_service": {"name": "fastapi_service", "status": "ok"},
            },
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Graph Endpoint (for visualization)
    # ─────────────────────────────────────────────────────────────────────────

    @app.get("/api/v1/graph")
    async def get_graph(limit: int = 50):
        """Fetch graph data for visualization with schema-agnostic queries.

        Returns nodes and edges as JSON for client-side rendering.
        Handles both connected nodes (with relationships) and isolated nodes.
        Uses dynamic property fallbacks to work with any node schema.
        """
        if not app_state["neo4j_client"]:
            return {"nodes": [], "edges": [], "error": "Neo4j unavailable"}

        try:
            # Query 1: Fetch Entity nodes and their semantic relationships
            # Prioritizes Entity nodes (with real names) over Chunk/Document scaffolding
            cypher_edges = """
            MATCH (e:Entity)-[r:RELATED_TO|MENTIONED_IN|WORKS_AT|LOCATED_IN|CREATED]->(target)
            WHERE target:Entity OR target:Chunk OR target:Document
            RETURN
                'Entity' AS source_type,
                COALESCE(e.canonical_name, e.name, e.id, 'Unknown') AS source,
                type(r) AS rel,
                labels(target)[0] AS target_type,
                CASE
                    WHEN target:Entity THEN COALESCE(target.canonical_name, target.name, target.id, 'Unknown')
                    WHEN target:Chunk THEN COALESCE(target.document_id, 'Chunk')
                    WHEN target:Document THEN COALESCE(target.name, target.id, 'Document')
                    ELSE 'Unknown'
                END AS target,
                elementId(e) AS source_id,
                elementId(target) AS target_id
            LIMIT $limit
            """

            result_edges = app_state["neo4j_client"].query(
                cypher_edges,
                parameters={"limit": limit},
                read_only=True
            )

            # Build nodes and edges from relationships
            nodes = {}
            edges = []

            for record in result_edges.records:
                src = record.get("source") or "Unknown"
                src_type = record.get("source_type") or "Node"
                tgt = record.get("target") or "Unknown"
                tgt_type = record.get("target_type") or "Node"
                rel = record.get("rel") or "link"

                # Convert to string and use as identifier
                src = str(src)
                tgt = str(tgt)

                # Add source node
                if src not in nodes:
                    nodes[src] = {"id": src, "label": src, "type": src_type}

                # Add target node
                if tgt not in nodes:
                    nodes[tgt] = {"id": tgt, "label": tgt, "type": tgt_type}

                # Add edge
                edges.append({
                    "source": src,
                    "target": tgt,
                    "label": rel,
                    "type": rel
                })

            # Query 2: Fetch isolated Entity nodes (only if we have room in limit)
            # Prioritize Entity nodes over Chunk/Document if any exist
            if len(nodes) < limit:
                remaining = limit - len(nodes)
                cypher_isolated = """
                MATCH (n:Entity)
                WHERE NOT EXISTS((n)-[]-())
                RETURN
                    'Entity' AS node_type,
                    COALESCE(n.canonical_name, n.name, n.id, 'UnnamedEntity') AS node_name,
                    elementId(n) AS node_id
                LIMIT $limit
                UNION
                MATCH (n:Chunk|Document)
                WHERE NOT EXISTS((n)-[]-())
                RETURN
                    labels(n)[0] AS node_type,
                    COALESCE(n.name, n.document_id, n.id, 'UnnamedNode') AS node_name,
                    elementId(n) AS node_id
                LIMIT 0
                """

                result_isolated = app_state["neo4j_client"].query(
                    cypher_isolated,
                    parameters={"limit": remaining},
                    read_only=True
                )

                for record in result_isolated.records:
                    name = str(record.get("node_name", "Unknown"))
                    node_type = record.get("node_type") or "Node"
                    node_id = str(record.get("node_id", name))

                    if node_id not in nodes:
                        nodes[node_id] = {"id": node_id, "label": name, "type": node_type}

            logger.info(f"Graph endpoint: {len(nodes)} nodes, {len(edges)} edges")

            return {
                "nodes": list(nodes.values()),
                "edges": edges,
                "count": {"nodes": len(nodes), "edges": len(edges)}
            }

        except Exception as e:
            logger.error(f"Graph fetch error: {e}")
            return {
                "nodes": [],
                "edges": [],
                "error": str(e)
            }

    # ─────────────────────────────────────────────────────────────────────────
    # Error Handler
    # ─────────────────────────────────────────────────────────────────────────

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions with consistent error format."""
        request_id = getattr(request.state, "request_id", "unknown")
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.detail,
                "request_id": request_id,
            },
        )

    logger.info("FastAPI app created and configured")
    return app


# ─────────────────────────────────────────────────────────────────────────────
# Application instance
# ─────────────────────────────────────────────────────────────────────────────

app = create_app()
