# CONTEXT.md — 4-Stage GraphRAG Pipeline Router

This document is the architectural contract for the Enterprise GraphRAG & Multi-Agent Swarm Engine. It describes the four pipeline stages, the boundary each one owns, and how a request is routed across them.

Stages live in `stages/`, in execution order (Stage 01 → 04 below). Shared configuration lives in `_config/`. Every stage is intended to be independently runnable and independently testable — a stage communicates with its neighbors through a declared payload, never by reaching into another stage's internals.

> Status: scaffolding. The directories and contracts below are established; the implementations are not yet written. Treat the payload shapes as the intended design, not as code that exists.

---

## Pipeline overview

```
                            ┌─────────────────────────────┐
   source documents ───────▶│ extraction                  │
                            │ parse · chunk · extract     │
                            │ entities & relations        │
                            └──────────────┬──────────────┘
                                           │ ExtractionResult
                                           ▼
                            ┌─────────────────────────────┐
                            │ graph_indexing              │
                            │ resolve entities · upsert   │
                            │ nodes/edges · embed chunks  │
                            └──────────────┬──────────────┘
                                           │ Neo4j graph + vector index
                                           ▼
   user query ─────────────▶┌─────────────────────────────┐
                            │ reasoning_agent             │
                            │ route · retrieve (graph +   │
                            │ vector) · multi-agent swarm │
                            └──────────────┬──────────────┘
                                           │ AnswerPayload
                                           ▼
                            ┌─────────────────────────────┐
                            │ fastapi_service             │
                            │ HTTP surface · auth · async │
                            │ jobs · streaming responses  │
                            └─────────────────────────────┘
```

There are two entry paths through the system. **Ingestion** flows `01 → 02` and is asynchronous and batch-oriented. **Query** flows `03 → 04` (or, from the caller's perspective, `04 → 03 → 04`) and is request/response. Stage 02's output — the populated graph — is the only thing the query path depends on, which is what lets ingestion and serving scale separately.

---

## Stage 01 — `stages/extraction/`

**Owns:** turning unstructured source material into a typed, chunk-attributed set of candidate entities and relations.

Responsibilities:
- Load documents from configured sources; normalize to text plus provenance metadata (source id, page/offset, timestamp).
- Chunk with `langchain-text-splitters`, preserving the offsets needed to cite a passage later.
- Run LLM-backed extraction to produce candidate nodes and typed edges against the schema declared in `_config/`.
- Emit records that are self-describing and provenance-complete. This stage does not talk to Neo4j.

**Outputs** an `ExtractionResult` per document: the chunk list, candidate entities (surface form, type, source chunk), candidate relations (subject, predicate, object, source chunk), and document-level metadata.

**Key constraint:** extraction is the only stage permitted to hallucinate — everything downstream treats its output as *candidate* data. Confidence and provenance must survive to stage 02 so that resolution can adjudicate.

---

## Stage 02 — `stages/graph_indexing/`

**Owns:** the knowledge graph and its indexes. This is the only stage that writes to Neo4j.

Responsibilities:
- Entity resolution and deduplication: collapse candidate entities into canonical nodes.
- Idempotent upserts (`MERGE`-based) so re-ingesting a document does not fan out duplicates.
- Constraint and index management, including the vector index over chunk embeddings.
- Chunk embedding, so that the same store backs both graph traversal and semantic search.
- Graph-level enrichment via APOC (community detection, path materialization, bulk operations).

**Outputs** a populated Neo4j graph with a vector index — addressed by stage 03 through a read-only accessor, not by re-implementing Cypher in the agent layer.

**Key constraint:** every write must be idempotent and re-runnable. Ingestion will be retried.

---

## Stage 03 — `stages/reasoning_agent/`

**Owns:** the router, retrieval strategy, and the multi-agent swarm. This is the stage the project's name is really about.

Responsibilities:
- **Route** the incoming query to a retrieval strategy: graph traversal for multi-hop and relational questions, vector search for semantic/descriptive ones, hybrid for most real queries, and direct Cypher for aggregate/structural ones.
- Execute retrieval read-only against the stage 02 graph.
- Orchestrate the agent swarm with LangGraph — specialist agents (retrieval, Cypher authoring, synthesis, verification) coordinated as a stateful graph with checkpointing, so long reasoning runs are resumable and inspectable.
- Ground the answer: every claim carries the node, edge, or chunk it came from.

**Outputs** an `AnswerPayload`: the answer text, the citation set, the retrieval trace (strategy chosen, subgraph touched), and token/latency accounting.

**Key constraint:** read-only with respect to the graph. If reasoning discovers that the graph is wrong or incomplete, it reports that; it does not repair it inline. Corrections are re-ingestion, which means stages 01–02.

---

## Stage 04 — `stages/fastapi_service/`

**Owns:** the HTTP surface and everything operational around it.

Responsibilities:
- FastAPI application, routers, and Pydantic v2 request/response models.
- Query endpoints (including streaming), ingestion trigger endpoints, and health/readiness checks that report Neo4j reachability.
- Async job handling for ingestion, which is far too slow to serve inline.
- Auth, rate limiting, error mapping, and structured request logging.

**Key constraint:** this stage contains no reasoning and no Cypher. It validates, delegates to stage 03 or the ingestion path, and shapes the response. Business logic leaking into route handlers is the failure mode to watch for.

---

## `_config/`

Cross-cutting configuration that no single stage owns: the graph schema (entity types, relation types, and which extractions are admissible), model and provider selection per stage, chunking and retrieval parameters, and prompt templates. Settings are loaded through `pydantic-settings`, backed by `.env` — see `.env.example` for the expected keys.

`_config/neo4j/import/` is bind-mounted into the Neo4j container at `/import` for bulk CSV loads.

---

## Routing contract

The "router" in this document's title is stage 03's query classifier, but the term applies to the pipeline as a whole: each stage is a router in that it accepts one declared payload shape and emits another, and knows nothing about how its neighbors are implemented.

| Boundary | Payload | Direction |
|---|---|---|
| 01 → 02 | `ExtractionResult` (chunks + candidate entities/relations + provenance) | ingestion |
| 02 → 03 | Neo4j graph + vector index, via read-only accessor | query |
| 03 → 04 | `AnswerPayload` (answer + citations + trace + usage) | query |
| 04 → 01 | ingestion job request | ingestion trigger |

Consequences worth stating explicitly: stage 04 never imports stage 02, stage 03 never writes to the graph, and stage 01 never sees a database connection. When a change seems to require violating one of these, the payload contract is what should change — deliberately — rather than the boundary being crossed quietly.

---

## Local infrastructure

`docker-compose.yml` provides the Neo4j 5 backing store with APOC enabled, on Bolt `7687` and Browser `7474`, with credentials defaulting to `neo4j` / `graphrag_dev_password` (override `NEO4J_PASSWORD` in `.env`). Data, logs, and plugins persist in named volumes, so a `docker compose down` does not discard the graph.

```
docker compose up -d      # start Neo4j
docker compose down       # stop, keeping volumes
```
