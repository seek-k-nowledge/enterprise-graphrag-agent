# Stage 2 — Graph Indexing & Neo4j Integration

**Not yet implemented.** This document establishes the contract; implementation begins with `neo4j_client.py`. Read the root [`../../CONTEXT.md`](../../CONTEXT.md) first for the pipeline-wide picture.

---

## Mandate

Accept candidate entities and relations from Stage 1, resolve them into a canonical knowledge graph, embed chunk text for semantic search, and write everything idempotently to Neo4j so that Stage 3 can query it read-only.

**This stage owns:** entity resolution and deduplication, constraint and index management, chunk embedding, idempotent upserts, and Neo4j as the write-once source of truth.

**This stage does not own:** reasoning, Cypher authoring, vector search orchestration, or any query logic beyond basic CRUD. Stage 3 queries the graph through a read-only accessor and never writes back. Stage 1 output is *candidate* data; this stage adjudicates and canonicalizes it.

**The key constraint** (restated from root): every write must be idempotent and re-runnable. Re-ingesting the same document must produce the same nodes/edges in the graph, with no duplicates or version drift. If an entity's resolution changes (e.g., a new rule in `_config/` that merges two previously-separate nodes), that is a *schema* decision requiring a full re-ingest, not a bug fix.

---

## Payload contract

### Input: `ExtractionResult` (from Stage 1)

Consumed directly from Stage 1's output. See [`../extraction/CONTEXT.md`](../extraction/CONTEXT.md) for the full shape; summarized:

```python
class ExtractionResult(BaseModel):
    metadata: DocumentMetadata     # document_id, uri, content_sha256, model_id, schema_version
    chunks: list[Chunk]            # id, text, offsets
    entities: list[CandidateEntity]  # surface_form, type, canonical_name (hint), description
    relations: list[CandidateRelation]  # subject_id, target_id, type, evidence
    errors: list[ExtractionError]  # non-fatal hallucinations dropped by stage 1
```

### Output: Neo4j Graph + Vector Index

Not a Pydantic model — the graph and indexes themselves are the output. For Stage 3 consumption, we expose:

```python
class GraphWriteResult(BaseModel):
    """Result of processing one ExtractionResult into the graph."""
    document_id: str
    nodes_created: int
    nodes_merged: int
    edges_created: int
    edges_merged: int
    chunks_embedded: int
    vectors_indexed: int
    canonical_entities: dict[str, CanonicalNode]  # candidate_id → canonical node for tracing
    canonical_relations: dict[str, CanonicalRelation]
    resolution_metadata: dict  # stats on deduplication: fuzzy_matches, semantic_merges, etc.
    timestamp: datetime
```

The graph itself is addressed by Stage 3 through a `GraphAccessor` (read-only), never by Stage 3 reimporting or calling Cypher directly.

---

## Neo4j Schema

### Node Types (Labels)

```cypher
(:Entity {
    id: String (unique),           # entity_type:canonical_name, deterministic
    entity_type: String,           # "Person", "Organization", "Location", etc.
    canonical_name: String,        # single authoritative name
    surface_forms: [String],       # all variations ever seen: ["Acme Corp", "Acme Corporation", "ACME"]
    description: String,
    first_seen: DateTime,          # when first canonical form was inferred
    updated: DateTime,
    sources: [String]              # document_ids mentioning this entity
})

(:Chunk {
    id: String (unique),           # document_id:chunk_index
    text: String,
    start_char: Int,
    end_char: Int,
    document_id: String,
    embedding: List[Float],        # vector, indexed for semantic search
    source_uri: String,
    created: DateTime
})

(:Document {
    id: String (unique),           # content-derived, deterministic
    uri: String,
    content_sha256: String,
    ingested_at: DateTime,
    extraction_model: String,
    schema_version: String
})
```

### Relationship Types (Edges)

```cypher
(entity1)-[:RELATION_TYPE {
    description: String,
    evidence: String,              # verbatim span from the extraction that supports this edge
    supporting_chunks: [String],   # chunk IDs providing evidence
    relation_count: Int,           # how many times this edge was asserted across chunks
    confidence: Float,             # 0–1, based on support breadth and consistency
    created: DateTime,
    updated: DateTime
}]->(entity2)

(:Entity)-[:MENTIONED_IN {chunk_count: Int}]->(:Chunk)  # entity appears in chunk
(:Chunk)-[:FROM]->(:Document)                           # chunk belongs to document
(:Entity)-[:FROM]->(entity:Entity {is_canonical: true}) # legacy: entity merged into this canonical
```

### Constraints & Indexes

```cypher
-- Uniqueness
CREATE CONSTRAINT entity_id_unique FOR (e:Entity) REQUIRE e.id IS UNIQUE
CREATE CONSTRAINT chunk_id_unique FOR (c:Chunk) REQUIRE c.id IS UNIQUE
CREATE CONSTRAINT document_id_unique FOR (d:Document) REQUIRE d.id IS UNIQUE

-- Performance
CREATE INDEX entity_type_idx FOR (e:Entity) ON (e.entity_type)
CREATE INDEX chunk_document_idx FOR (c:Chunk) ON (c.document_id)
CREATE INDEX document_uri_idx FOR (d:Document) ON (d.uri)

-- Vector search (requires Neo4j 5.16+ with vector index support)
CREATE VECTOR INDEX chunk_embedding_idx FOR (c:Chunk) ON (c.embedding) 
  OPTIONS {indexConfig: {vector: {dimensions: 1536, similarity_function: "cosine"}}}
```

---

## Pipeline: From Candidate to Graph

### 1. Resolve Entities

**Input:** list of `CandidateEntity` from `ExtractionResult`.

**Process:**

1. **Within-document deduplication:** collapse multiple candidates with identical `(entity_type, canonical_name)`. Keep all surface forms, merge chunk references.
2. **Cross-document deduplication (resolution):**
   - **String-match rule:** entities with identical `(entity_type, canonical_name)` from *any* document → single canonical node.
   - **Fuzzy-match rule:** entities with normalized names within `_config/resolution.fuzzy_threshold` edit distance (default 0.85, from `difflib.SequenceMatcher`) and same type → candidate for merge. Manual review step flagged in `resolution_metadata.fuzzy_candidates` (not auto-merged yet).
   - **Type-specific rules:** `_config/resolution.rules` may override: e.g., all `Organization` entities with headquarters in the same city are often the same entity, or a hardcoded list of aliases ("Acme Corp" → "Acme Corporation"). Apply these before fuzzy matching.
   - **Semantic merge (future):** entities with semantically similar descriptions (embedding distance < threshold) are *candidates* for merge, not auto-merged. Requires human adjudication or a confidence threshold calibrated per domain.

2. **Output:** mapping `candidate_id → canonical_node_id` and a list of `CanonicalNode` objects for insertion.

**Idempotency:** re-running on the same `ExtractionResult` produces identical canonical nodes by `id`. Re-running on a new document may merge its candidates with existing canonical nodes (cross-document resolution), so the graph grows but never branches.

### 2. Resolve Relations

**Input:** list of `CandidateRelation`, resolved `CandidateEntity` → `CanonicalNode` mapping.

**Process:**

1. Map relation source/target from candidate IDs to canonical node IDs.
2. Collapse multiple relations between the same pair of canonical nodes into one edge with aggregated metadata:
   - `relation_count`: how many times this edge was asserted
   - `supporting_chunks`: all chunks that mentioned it
   - `confidence`: `len(supporting_chunks) / total_mentions_of_source_entity` (0–1 measure of support breadth)
3. Merge descriptions and evidence (keep all, or summarize for the graph? → currently store all in a list)

**Idempotency:** same as entities. Re-ingesting produces the same edges; new ingestions may increase `relation_count` and `supporting_chunks` but do not duplicate edges.

### 3. Upsert to Neo4j

**Process:**

1. **Document node:** MERGE on `id`, set `uri`, `ingested_at`, `extraction_model`, `schema_version`.
2. **Chunk nodes:** MERGE on `id` (deterministic), set all text and offset fields. Do NOT embed yet; batching embedding saves API calls.
3. **Entity nodes:** MERGE on `id` (entity_type:canonical_name), merging `surface_forms` list and `sources` list.
4. **Relationships:**
   - `(Entity)-[:MENTIONED_IN {chunk_count}]-(Chunk)` — MERGE, incrementing chunk_count.
   - `(Chunk)-[:FROM]-(Document)` — MERGE (idempotent by design).
   - `(Entity)-[:RELATION_TYPE]-(Entity)` — MERGE on `(source_id, relation_type, target_id)`, merging `supporting_chunks`, `evidence` lists, and recalculating `confidence`.

All operations are MERGE-based, so re-ingestion is safe. Transactions are batched by document to keep memory bounded.

**Constraints & Indexes:** created in a setup phase, before any writes. Not recreated per ingest.

### 4. Embed Chunks

**Process:**

1. Collect all `Chunk` nodes that were inserted/updated in this ingest (use `RETURNED` values from the MERGE).
2. Batch-embed chunk text via embedding provider (Anthropic or OpenAI, configured in `_config/`).
3. Update chunk nodes with embedding vector: `SET c.embedding = $vector`.
4. Ensure vector index exists; update it if needed (usually once at setup, but Neo4j indexes auto-update).

**Provider selection:** `_config/embedding.model` and `embedding.provider`. Embedding is deterministic, so re-running with the same model/provider produces identical vectors.

**Batching:** respect API rate limits; default 100 chunks per batch. Track which chunks have been embedded to avoid re-embedding on retries.

### 5. Optional: APOC Enrichment

**Process (can run on-write or on-demand):**

- **Community detection:** `apoc.algo.community.lpa()` to find clusters of related entities.
- **Path materialization:** pre-compute common paths (e.g., all 2-hop reachable nodes from a given entity) for faster traversal in Stage 3.
- **Centrality measures:** PageRank, betweenness for prioritizing results.

These are optional for MVP; flag in `_config/apoc.enabled`.

---

## Entity Resolution Strategy

### Determinism & Calibration

Resolution must be **deterministic** so the same input always produces the same graph. This means:

1. **No randomness** in algorithms.
2. **Ordered processing** — when comparing candidates, process in a consistent order (sorted by ID).
3. **Threshold parameters** are in `_config/`, not hardcoded, so changes are deliberate and auditable.

### Calibration

Tuning is empirical:

- Run a small ingestion with fuzzy matching disabled (only exact-match and rules).
- Measure precision/recall by sampling and labeling results: did entities merge when they should? Did separate entities stay separate?
- Adjust `fuzzy_threshold` and rules iteratively.
- Log all fuzzy matches and rule applications to `resolution_metadata` for audit.

### When Resolution Changes

If `_config/resolution` rules change, the *new* rules apply only to *new* ingestions. Existing graph nodes do not retroactively merge — that is a full re-ingest of all documents under the new rules, which is a schema migration, not an incremental update. Document this as a breaking change.

---

## Idempotency Guarantees

**What it means:** ingesting the same `ExtractionResult` twice produces an identical graph state. More precisely:

```
ingest(result) ; ingest(result)  ≡  ingest(result)
```

**How we achieve it:**

1. **Deterministic IDs:** `CanonicalNode.id` is derived from `(entity_type, canonical_name)`, not from timestamps or counters.
2. **MERGE operations:** Neo4j's `MERGE` clause only creates a node if it does not exist; re-running is idempotent.
3. **Metadata updates:** fields like `sources` (document ids), `supporting_chunks`, and timestamps are lists/sets that accumulate rather than overwrite.
4. **No shadow state:** all decisions (resolution, which chunks to embed) are derived from the `ExtractionResult` and `_config/`, never from side effects or database state read before writing.

**Edge case:** if an `ExtractionResult` is modified between ingestions (e.g., Stage 1 reprocesses a document with a better model), the graph updates reflect the new data, but old data is not deleted. This is intentional — the graph is append-only from Stage 2's perspective. Deletion is a separate operation and requires explicit schema governance.

---

## Configuration: `_config/` keys consumed

| Key | Purpose |
|---|---|
| `neo4j.uri` | Bolt connection string, e.g., `bolt://localhost:7687` |
| `neo4j.user` / `neo4j.password` | Credentials |
| `neo4j.batch_size` | Upsert batch size (nodes/relations per transaction) |
| `embedding.model` | Model ID for chunk embeddings, e.g., `text-embedding-3-large` (OpenAI) or `claude-embedding-20250115` (Anthropic) |
| `embedding.provider` | `"anthropic"` or `"openai"` |
| `embedding.batch_size` | Chunks per embedding API call (default 100) |
| `embedding.dimensions` | Vector dimension (1536 for OpenAI 3-large, 1024 for Anthropic) |
| `resolution.fuzzy_threshold` | Edit distance threshold for fuzzy matching (0–1, default 0.85) |
| `resolution.rules` | List of domain-specific merge rules (JSON file path) |
| `apoc.enabled` | Whether to run APOC enrichment on write (default false for MVP) |
| `graph_schema.entity_types` | Admissible entity type labels (validation) |
| `graph_schema.relation_types` | Admissible edge labels (validation) |

Loaded through `pydantic-settings`, backed by `.env`. `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` are the environment variable equivalents.

---

## Module Layout

```
stages/graph_indexing/
├── CONTEXT.md              this file
├── __init__.py             public surface: process_extraction()
├── schemas.py              payload models + internal data structures
├── neo4j_client.py         driver wrapper, connection pooling, query execution
├── entity_resolver.py      resolution algorithm
├── graph_writer.py         MERGE-based upserts, constraint/index creation
├── embedder.py             chunk embedding and vector index operations
└── config.py               (or imported from _config/) resolution rules, thresholds
```

Entry point: `process_extraction(extraction_result: ExtractionResult) -> GraphWriteResult`. Orchestrates: resolve → write → embed → return metadata.

---

## Invariants & Validation

Enforced at write time:

| # | Invariant | Check |
|---|---|---|
| 1 | Every `CanonicalEntity.id` is unique within the ingest | enforced by MERGE |
| 2 | Every entity node in Neo4j has exactly one `id` (primary key) | enforced by constraint |
| 3 | Relation source/target resolve to existing nodes | check before MERGE |
| 4 | `surface_forms` list contains at least the `canonical_name` | validator |
| 5 | Chunk embeddings have the correct dimension (1536 or 1024) | check on set |
| 6 | All referenced chunks exist in Neo4j before MERGE relations | query-driven |
| 7 | Document ID is identical to Stage 1's document ID | pass-through from metadata |

Violations are logged with the offending data for debugging (resolution bugs, config errors, etc.) but do not fail the whole ingest — chunks in error are recorded and can be retried.

---

## Testing

**Not yet written.** High-priority tests:

- **Neo4j connectivity:** fixture starts docker-compose Neo4j, fixture cleanup stops it. Parametrized over local / Docker Desktop / CI runner paths.
- **Idempotency:** ingest the same `ExtractionResult` twice, assert the graph is identical both times. Count nodes, edges, vector index size.
- **Resolution determinism:** run entity resolution on the same candidates in different orders, assert results are identical.
- **Fuzzy matching calibration:** golden set of entity pairs with known true/false merges, measure precision/recall at different thresholds.
- **Upsert correctness:** MERGE a node, read it back, modify and MERGE again, assert metadata was updated not duplicated.
- **Embedding round-trip:** embed chunks, query by similarity, assert correct chunks are returned.
- **Constraint violation handling:** attempt to insert a duplicate entity ID, assert constraint error is caught and logged.

Fake `neo4j.driver.Session` for unit tests where Neo4j is not needed; use real driver against test container for integration tests.

---

## Open Questions

1. **Fuzzy matching strategy:** should we auto-merge fuzzy candidates or require manual review? Current design: flag for review and provide a merge workflow, but do not auto-merge.
2. **Semantic resolution (embeddings for entity matching):** should entity descriptions be embedded and compared? Deferred to post-MVP; would require a second embedding model or reuse of chunk embeddings.
3. **Relation merging:** when multiple documents assert the same relation, do we keep separate edges (one per document) or collapse into one edge with `relation_count`? Current design: collapse into one edge, tracking count.
4. **APOC on-write vs on-demand:** community detection and centrality measures are expensive. Should they run every ingest or only when queried? Current design: on-demand (flag in config); on-write is future optimization.
5. **Chunk deletion:** if a document is re-ingested and a chunk is no longer present (document shortened, or section removed), should the old chunk be deleted or marked as stale? Current design: keep forever (append-only); explicit deletion requires a separate operation.
6. **Vector embedding caching:** should we cache embeddings so the same chunk text always produces the same vector, even if the embedding model changes? Current design: no caching; re-embedding with a new model updates the vector.
7. **Resolution rollback/audit trail:** if a resolution rule is later found to be wrong and entities need to be un-merged, what is the procedure? Current design: full re-ingest with corrected rules; no audit trail yet.
