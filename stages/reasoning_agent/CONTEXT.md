# Stage 3 — Reasoning Agent & Query Orchestration

**Not yet implemented.** This document establishes the contract; implementation begins with `graph_accessor.py`. Read the root [`../../CONTEXT.md`](../../CONTEXT.md) first for the pipeline-wide picture.

---

## Mandate

Accept a user query, retrieve relevant knowledge from the Stage 2 graph, and synthesize a grounded answer using a multi-agent swarm. Every claim in the answer must carry citations to the graph nodes, edges, or chunks that support it.

**This stage owns:** query routing, retrieval orchestration, LangGraph-based agent swarm, answer grounding, and citation tracking.

**This stage does not own:** Neo4j writes, entity resolution, embedding, or HTTP handling. It is read-only with respect to the graph. If reasoning discovers a gap or error in the graph, it reports it; correction requires re-ingestion (stages 01–02).

**The key constraint** (restated from root): read-only. This stage is a consumer of Stage 2's graph, never a writer to it.

---

## Payload Contract

### Input: User Query

```python
class QueryPayload(BaseModel):
    """User query with optional context."""
    text: str                          # the question
    context: Optional[str] = None      # optional background
    user_id: Optional[str] = None      # for tracking and personalization
    session_id: Optional[str] = None   # for checkpoint and resumption
```

### Output: Answer Payload

```python
class AnswerPayload(BaseModel):
    """Grounded answer with full provenance."""
    answer_text: str
    citations: list[Citation]          # each claim tied to a graph element
    retrieval_trace: RetrievalTrace    # strategy, subgraph touched, reasoning steps
    confidence: float                  # 0-1, based on citation coverage
    token_usage: TokenUsage            # input/output tokens, by model
    latency_ms: int                    # end-to-end execution time
```

Where:

```python
class Citation(BaseModel):
    """A source for a claim."""
    claim: str                         # the specific text being cited
    source_type: str                   # "node", "edge", "chunk"
    source_id: str                     # Neo4j entity ID
    source_text: str                   # content of the cited entity
    confidence: float                  # how strongly this supports the claim

class RetrievalTrace(BaseModel):
    """Audit trail of retrieval decisions."""
    query_classification: str          # "multi_hop", "semantic", "aggregate", "hybrid"
    retrieval_strategies: list[str]    # ["graph_traversal", "vector_search"]
    subgraph_stats: dict               # nodes_touched, edges_touched, chunk_passages
    reasoning_steps: list[dict]        # intermediate agent decisions
    errors: list[dict]                 # retrieval errors or fallbacks
```

---

## Architecture: Query Routing & Retrieval Strategies

### 1. Query Classification

The router classifies incoming queries into retrieval strategies based on linguistic patterns and query structure:

```
┌─────────────────────┐
│  User Query         │
└──────────┬──────────┘
           │
           ▼
┌──────────────────────────────────────────┐
│ Query Classifier (LLM)                   │
│ - Multi-hop relations?                   │
│ - Semantic/descriptive?                  │
│ - Aggregate/structural?                  │
└──────────┬───────────────────────────────┘
           │
      ┌────┴────┬────────────┬────────────┐
      │          │            │            │
      ▼          ▼            ▼            ▼
   Multi-hop  Semantic    Aggregate    Hybrid
  (Graph)    (Vector)     (Cypher)     (All 3)
```

**Classification Patterns:**

| Query Type | Indicators | Strategy |
|---|---|---|
| **Multi-hop** | "How does X relate to Y?", "Path from A to B?", temporal/causal chains | Graph traversal (Cypher) |
| **Semantic** | "Describe X", "What is X?", "Similar to X?", free-form descriptive | Vector search + synthesis |
| **Aggregate** | "Count X", "Which X has most Y?", "Top 10 by Z?", structural stats | Direct Cypher |
| **Hybrid** | Most real queries | All strategies, ranked and merged |

### 2. Retrieval Strategies

#### Strategy A: Graph Traversal (Multi-hop)

**When:** "How does company X relate to person Y?" or multi-step reasoning.

**How:**
1. Identify seed entities from query (named entity recognition or dense retrieval)
2. Author Cypher queries to traverse paths (1-3 hops typically)
3. Return subgraph: nodes, edges, and supporting chunks
4. Agent synthesizes answer from traversal results

**Cypher Patterns:**
```cypher
-- 1-hop: direct relationships
MATCH (a:Entity {canonical_name: $entity_a})-[r]->(b:Entity)
RETURN a, r, b, r.supporting_chunks as evidence

-- 2-hop: paths through intermediaries
MATCH (a:Entity {canonical_name: $entity_a})-[r1]->(x)-[r2]->(b)
WHERE x.entity_type IN $intermediate_types
RETURN path(a, r1, x, r2, b), [c IN (r1.supporting_chunks + r2.supporting_chunks) | c]

-- Constrained: type matching, relationship filtering
MATCH (a)-[r:IS_A|WORKS_AT|LOCATED_IN]->(b)
WHERE a.entity_type = $type_a AND r.confidence > $min_confidence
RETURN a, r, b
```

**Output:** Subgraph with node/edge/chunk citations.

#### Strategy B: Vector Search (Semantic)

**When:** "Describe X" or "What is known about X?" or free-form question.

**How:**
1. Embed the query with the same embedding model as chunks
2. Search vector index (cosine similarity) for top-K relevant chunks
3. Retrieve full context: chunks + their source entities and relations
4. Synthesis agent writes answer grounded in chunk passages

**Process:**
```
Query → Embed → Vector Search → Top-K Chunks → 
  Get Source Entities → Get Relations → Subgraph → Synthesize
```

**Output:** Ranked list of chunks (by similarity), their source entities/relations.

#### Strategy C: Direct Cypher (Aggregate/Structural)

**When:** "How many people work at company X?" or structural queries.

**How:**
1. Author Cypher directly for the question (Cypher-writing agent)
2. Execute and format results
3. Answer is typically already structured (table, count, etc.)

**Examples:**
```cypher
MATCH (p:Entity {entity_type: "Person"})-[:WORKS_AT]->(c:Entity {canonical_name: $company})
RETURN count(distinct p) as employee_count

MATCH (c:Entity {entity_type: "Company"})-[:LOCATED_IN]->(l:Entity)
RETURN c.canonical_name, l.canonical_name ORDER BY count(*) DESC LIMIT 10
```

**Output:** Structured result (count, list, table).

#### Strategy D: Hybrid (Most Queries)

**When:** Default for real-world questions.

**How:**
1. Run all applicable strategies in parallel
2. Rank results by relevance and confidence
3. Synthesis agent merges answers, deduplicates
4. Return highest-confidence grounded answer

---

## LangGraph Agent Swarm

### Architecture

```
QueryPayload
    │
    ▼
┌─────────────────────────────────────────┐
│ Query Router Agent                      │
│ - Classify query                        │
│ - Select retrieval strategy             │
└────────────────┬────────────────────────┘
                 │
         ┌───────┴────────┬──────────────┐
         │                │              │
         ▼                ▼              ▼
    ┌────────────┐  ┌─────────────┐  ┌──────────┐
    │ Retrieval  │  │Cypher Agent │  │Vector    │
    │ Agent      │  │(Structural) │  │Search    │
    │(Multi-hop) │  │             │  │Agent     │
    └────┬───────┘  └─────┬───────┘  └────┬─────┘
         │                │              │
         └────────────────┼──────────────┘
                          │
                          ▼
              ┌──────────────────────────┐
              │ Synthesis Agent          │
              │ - Merge results          │
              │ - Ground with citations  │
              │ - Write natural language │
              └────────┬─────────────────┘
                       │
                       ▼
              ┌──────────────────────────┐
              │ Verification Agent       │
              │ - Check consistency      │
              │ - Validate citations     │
              │ - Flag gaps or errors    │
              └────────┬─────────────────┘
                       │
                       ▼
                  AnswerPayload
```

### Agents

#### Router Agent
- **Input:** QueryPayload
- **Responsibility:** Classify query type and select retrieval strategy
- **Output:** {strategy: str, seed_entities: list[str], parameters: dict}
- **Model:** Claude 3.5 Sonnet or Opus (reasoning-heavy)

#### Retrieval Agent (Multi-hop)
- **Input:** Query, seed entities, graph accessor
- **Responsibility:** Author and execute Cypher for multi-hop traversal
- **Output:** {nodes: list, edges: list, chunks: list, confidence: float}
- **Model:** Claude (code-writing, Cypher authoring)

#### Cypher Agent (Structural)
- **Input:** Query, schema
- **Responsibility:** Author Cypher for aggregate/structural queries
- **Output:** {result: str, cypher: str, explanation: str}
- **Model:** Claude (Cypher specialist)

#### Vector Search Agent
- **Input:** Query, graph accessor
- **Responsibility:** Embed query, search vector index, retrieve context
- **Output:** {chunks: list[dict], entities: list, relations: list}
- **Model:** LLM (lightweight; mostly orchestrating vector search)

#### Synthesis Agent
- **Input:** Results from retrieval agents
- **Responsibility:** Merge results, ground in citations, write natural language answer
- **Output:** {answer_text: str, citations: list[Citation]}
- **Model:** Claude (synthesis, grounding)

#### Verification Agent
- **Input:** Proposed answer + citations
- **Responsibility:** Check consistency, validate citations, identify gaps
- **Output:** {valid: bool, issues: list[str], confidence: float}
- **Model:** Claude (critical thinking)

### State Machine & Checkpointing

All agents run within a LangGraph state machine:

```python
class AgentState(BaseModel):
    query: QueryPayload
    router_output: dict               # classification + strategy
    retrieval_results: dict           # results from all strategies
    synthesized_answer: dict          # draft answer + citations
    verification_result: dict         # issues and final confidence
    final_answer: AnswerPayload       # output
```

**Checkpointing:**
- Enabled via `langgraph.checkpoint` (SQLite or PostgreSQL backend)
- Long-running queries are resumable from the last completed step
- Enables inspection and debugging of agent decisions
- Session ID links checkpoints to user sessions

---

## Graph Accessor: Read-Only Interface to Stage 2

The `GraphAccessor` encapsulates all Neo4j reads and provides a clean boundary between Stage 3 and Stage 2. Stage 3 never imports Neo4jClient directly.

```python
class GraphAccessor:
    """Read-only interface to the Stage 2 graph."""

    def get_entity(self, entity_id: str) -> Optional[Entity]
    def search_entities(self, text: str, entity_type: Optional[str]) -> list[Entity]
    def get_entity_relations(self, entity_id: str) -> list[Relation]
    def traverse_path(self, start_id: str, max_hops: int) -> Subgraph
    def search_chunks_by_embedding(self, query_embedding: list[float], top_k: int) -> list[Chunk]
    def get_chunks_for_entity(self, entity_id: str) -> list[Chunk]
    def get_entity_by_name(self, name: str, entity_type: Optional[str]) -> Optional[Entity]
    
    # Returns Subgraph(nodes, edges, chunks) for easy serialization
```

**Constraints:**
- No write methods
- All queries are parameterized (prevent injection)
- Timeout on expensive queries (max 30s)
- Cache results for repeated queries (optional, with TTL)

---

## Answer Grounding & Citation

Every claim in the answer must trace back to the graph. Citation has three forms:

1. **Node Citation:** "Entity X is a [type]" → cite the Entity node
2. **Edge Citation:** "X and Y are [related]" → cite the Relation edge + confidence
3. **Chunk Citation:** "X is described as [verbatim]" → cite the Chunk with offset

**Grounding Process:**

1. Synthesis agent writes answer claim by claim
2. For each claim, identify supporting graph element (node, edge, or chunk)
3. Record Citation(claim, source_type, source_id, source_text, confidence)
4. Verification agent checks: can a reader follow each citation back to the source?

**Confidence Calculation:**

```
Per-citation confidence = (citation_breadth × relation_confidence) / total_mentions

Where:
- citation_breadth = number of distinct chunks/sources supporting the claim
- relation_confidence = edge weight (0-1) if the claim comes from a relation
```

**Fallback:** If a claim cannot be grounded, synthesis agent flags it for the verification agent and excludes it from the final answer or marks it as uncertain.

---

## Configuration: `_config/` keys consumed

| Key | Purpose |
|---|---|
| `reasoning.retrieval_strategies` | List of enabled strategies (default: ["graph_traversal", "vector_search"]) |
| `reasoning.query_classifier_model` | Model for query routing (Sonnet/Opus) |
| `reasoning.cypher_writer_model` | Model for Cypher authoring |
| `reasoning.synthesis_model` | Model for answer synthesis |
| `reasoning.max_traversal_hops` | Max hops in graph traversal (default 3) |
| `reasoning.vector_search_top_k` | Top-K chunks for vector search (default 10) |
| `reasoning.min_citation_confidence` | Min confidence to include a citation (default 0.3) |
| `reasoning.enable_checkpointing` | Enable LangGraph checkpointing (default True) |
| `reasoning.checkpoint_backend` | "sqlite" or "postgres" |
| `reasoning.query_timeout_sec` | Timeout for Neo4j queries (default 30) |
| `reasoning.enable_caching` | Cache repeated queries (default True) |
| `reasoning.cache_ttl_sec` | Cache time-to-live (default 3600) |

---

## Module Layout

```
stages/reasoning_agent/
├── CONTEXT.md              this file
├── __init__.py             public surface: answer_query()
├── schemas.py              payload models (QueryPayload, AnswerPayload, RetrievalTrace, etc.)
├── graph_accessor.py       read-only interface to Stage 2 graph
├── query_router.py         query classification and strategy selection
├── retrieval/              retrieval strategies
│   ├── __init__.py
│   ├── base.py             abstract base for retrievers
│   ├── graph_traversal.py  multi-hop Cypher-based retrieval
│   ├── vector_search.py    semantic vector search
│   └── cypher_direct.py    structural/aggregate queries
├── agents/                 LangGraph-based agent swarm
│   ├── __init__.py
│   ├── router_agent.py     query classification
│   ├── retrieval_agent.py  multi-hop traversal orchestration
│   ├── cypher_agent.py     structural query authoring
│   ├── synthesis_agent.py  answer grounding and writing
│   └── verification_agent.py  consistency and citation checking
└── agent_graph.py          LangGraph state machine orchestration
```

Entry point: `answer_query(query_payload: QueryPayload, graph_accessor: GraphAccessor) -> AnswerPayload`. Coordinates all agents.

---

## Invariants & Constraints

| # | Invariant | Check |
|---|---|---|
| 1 | Every claim in answer_text has a citation | verification agent validates |
| 2 | Every citation points to a valid graph element | graph accessor pre-validates |
| 3 | Citation source_text is verbatim from the graph | enforced on retrieval |
| 4 | Confidence is 0-1 and calibrated per strategy | computed deterministically |
| 5 | Retrieval is read-only (no mutations) | code review + type hints |
| 6 | Session checkpoints use session_id as key | checkpoint backend |
| 7 | Cypher queries are parameterized | no string concatenation |
| 8 | Multi-hop traversal does not exceed max_hops | query construction |
| 9 | Vector search respects top_k limit | query parameter |
| 10 | Query timeout enforced per query | Neo4j client timeout |

---

## Testing

**Not yet written.** High-priority tests:

- **Query classification:** parametrized tests for each query type (multi-hop, semantic, aggregate)
- **Graph accessor:** mock Neo4j, assert correct Cypher is generated
- **Retrieval strategies:** isolated tests for each (no agent orchestration)
- **Agent decisions:** trace through examples (routing, synthesis, verification)
- **Citation grounding:** verify every claim has a valid citation
- **Checkpointing:** save/restore agent state, verify continuity
- **End-to-end:** full query → answer with citations (golden examples)

Mock the Neo4jClient; keep live Neo4j tests separate and opt-in.

---

## Open Questions

1. **Query ambiguity:** if a query is ambiguous (e.g., "Apple" could be company or fruit), should the router ask for clarification or run all interpretations? Current design: run all and merge.
2. **Answer length:** how long should answers be? Current design: as long as needed to ground all citations; synthesis agent decides.
3. **Citation density:** should every sentence have a citation or just key claims? Current design: every *claim* (semantic unit) has a citation, not every word.
4. **Confidence calibration:** is the confidence formula correct? Current design: empirical hypothesis; refine based on user feedback and comparison to ground truth.
5. **Reasoning transparency:** should intermediate reasoning steps be exposed to the user, or just the final answer? Current design: expose in `retrieval_trace` if requested.
6. **Fallback behavior:** if all retrieval strategies fail, return no answer or return "not found"? Current design: return "not found" with explanation in retrieval_trace.
7. **Multi-modal synthesis:** should the answer include tables, graphs, or structured data? Current design: text-only for MVP; structures in retrieval_trace.
8. **Personalization:** should entity/relation rankings change based on user profile or session history? Current design: not yet; query-independent ranking for MVP.
