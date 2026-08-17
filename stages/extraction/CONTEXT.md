# Stage 1 — Extraction

**Partly implemented.** `schemas.py` and `extractor.py` exist and match this document; loaders and the orchestration entrypoint do not. Read the root [`../../CONTEXT.md`](../../CONTEXT.md) first for the pipeline-wide picture.

---

## Mandate

Turn unstructured source material into a typed, provenance-complete set of **candidate** entities and relations, chunked and attributed well enough that any downstream claim can be traced back to the exact passage that produced it.

**This stage owns:** document loading, normalization, chunking, LLM-backed triple extraction, and schema validation of what the model returns.

**This stage does not own:** entity resolution, deduplication, canonicalization, embedding, Cypher, or any database connection. Stage 1 never imports a Neo4j driver and never sees `NEO4J_URI`. Two candidate entities with the surface forms `"Acme Corp"` and `"Acme Corporation"` are emitted as two records; deciding they are one node is stage 02's job.

**The key constraint** (restated from the root contract because it drives every design decision here): extraction is the only stage permitted to hallucinate. Everything it emits is a *candidate*. That is why confidence and provenance are required on every record rather than optional — stage 02 adjudicates, and it can only adjudicate what it can trace and score.

---

## Payload contract: `ExtractionResult`

The stage boundary. One `ExtractionResult` per source document, as Pydantic v2 models.

Defined in [`schemas.py`](schemas.py); the authoritative field list is the code. Summarized:

```python
class Chunk(BaseModel):
    id: str                   # deterministic: f"{document_id}:{index}"
    text: str
    start_char: int           # offset into the NORMALIZED document text
    end_char: int             # end_char - start_char == len(text), enforced

class CandidateEntity(BaseModel):
    id: str                   # deterministic, derived from (entity_type, canonical_name)
    entity_type: str
    canonical_name: str       # best single name; a hint for stage 02, not a decision
    surface_form: str         # verbatim span as it appears in the source
    description: str = ""
    chunk_ids: list[str]      # every chunk of this document that mentions it

class CandidateRelation(BaseModel):
    source_entity_id: str     # must reference a CandidateEntity in this result
    target_entity_id: str
    relation_type: str
    description: str = ""
    evidence: str             # verbatim span asserting the relation
    chunk_ids: list[str]

class ExtractionResult(BaseModel):
    metadata: DocumentMetadata    # document_id, uri, content_sha256, model id, schema version…
    chunks: list[Chunk]
    entities: list[CandidateEntity]
    relations: list[CandidateRelation]
    errors: list[ExtractionError] = []   # non-fatal failures and dropped records
```

`chunk_ids` is a list rather than a single reference because one entity is normally named in several chunks of a document — overlapping chunks guarantee it. Stage 02 can therefore weigh a candidate by how much of the document supports it.

**Invariants.** Enforced two ways: structurally by Pydantic validators (a violating `ExtractionResult` cannot be constructed), and by the extractor, which drops offending records into `errors` rather than raising.

| # | Invariant | Enforced by |
|---|---|---|
| 1 | Relation `source_entity_id` / `target_entity_id` exist in `entities` | validator (raises) |
| 2 | Every `chunk_ids` entry exists in `chunks` | validator (raises) |
| 3 | Chunk offsets are self-consistent: `end_char - start_char == len(text)` | validator (raises) |
| 4 | Chunk and entity ids are unique within a result | validator (raises) |
| 5 | `surface_form` is a verbatim span of its chunk | extractor (drops + records) |
| 6 | `evidence` is a verbatim span of its chunk | extractor (drops + records) |
| 7 | A relation only references entities reported for the same chunk | extractor (drops + records) |
| 8 | A failed chunk yields an `ExtractionError`, not a missing chunk | extractor |
| 9 | `entity_type` / `relation_type` ∈ `_config/` graph schema | **not yet enforced** — the types are in the prompt but not validated post-hoc |

Invariants 5–7 are the ones an LLM violates in practice, so expect records to be dropped and treat the drop rate as a live quality metric — a rising `surface_form_check` count means the prompt or the model tier has stopped working. Verbatim matching tolerates differing whitespace runs (models reproduce wording reliably but not always the exact spaces across a line wrap) and nothing else; paraphrase is rejected, which is the point.

**Not carried:** per-record `confidence`. An earlier draft of this contract required it, but the implemented models omit it, so stage 02 must adjudicate on provenance breadth (`len(chunk_ids)`) rather than a self-reported score. Self-reported LLM confidence is weakly calibrated anyway, but if stage 02's resolution wants a numeric prior, this is the gap to close and it is a schema change on both sides.

---

## Pipeline

### 1. Load

Loaders produce `(normalized_text, DocumentMetadata)`. Normalization is deliberately minimal — collapse line-ending variants, strip control characters, preserve everything else. **Do not** normalize whitespace aggressively or reflow paragraphs: `char_start`/`char_end` are offsets into this normalized text, and the invariant that they can be resolved back to a citable span is only as good as the text's stability.

Offsets are into the *normalized* text, not the raw bytes. Store the normalized text (or make normalization deterministic and re-runnable) or citations become unresolvable later.

### 2. Chunk

`RecursiveCharacterTextSplitter` from `langchain_text_splitters` (verified present at 1.1.2), configured from `_config/`. Requirements:

- Offsets must survive. `add_start_index=True` supplies them, but `chunk_text()` **verifies** each one against the source and recovers by search when it does not resolve — whitespace stripping can shift a boundary, and an offset that silently points at the wrong passage is worse than no offset.
- Overlap is expected (default 200 chars against 1200). A relation whose subject and object straddle a boundary is otherwise unrecoverable, so the same triple arriving from adjacent chunks is accepted and merged.
- `id` is deterministic: `f"{document_id}:{index}"`.

### 3. Extract

One structured-output LLM call per chunk, via `prompt | model.with_structured_output(ChunkExtraction)`, so schema conformance and retry live in the provider integration rather than in hand-rolled JSON scraping.

**The LLM sees a narrower schema than the stage boundary.** `ChunkExtraction` / `ExtractedEntity` / `ExtractedRelation` ask only for what is readable off the passage in front of the model — names, types, verbatim spans. No ids, no offsets, no chunk references. The model therefore *cannot* fabricate an id or a cross-chunk claim; `assemble()` derives ids from `(entity_type, canonical_name)` and resolves relations against the entities reported for the same chunk.

Relations are reported by entity **name** and resolved locally. A relation naming an entity the model did not report in the same response is dropped (invariant 7) — it is the cheapest available check against the model asserting an edge for something it never grounded.

Note that the `Field(description=...)` strings on those models are shipped to the provider as part of the schema, so they are written as instructions to the model, not as notes to a reader. Editing them is a prompt change.

The empty case matters. Most chunks of a real corpus assert nothing extractable, and a model that feels obliged to fill the lists will invent content — so both lists default to empty and the prompt names that as a normal outcome.

Chunks are independent, so they run through `.batch(..., max_concurrency=N, return_exceptions=True)`. One failed chunk records an `ExtractionError` and the rest of the document still lands.

Prompt text currently lives in `SYSTEM_PROMPT` in `extractor.py`, not in `_config/` — see open questions.

### 4. Validate

Enforce every invariant above. Emit the `ExtractionResult`. Records that fail validation go to `errors` with the offending payload attached, so prompt and schema problems are debuggable rather than invisible.

---

## Model selection

Selection lives in `_config/`, per stage, never hardcoded at a call site. Both `langchain-anthropic` (1.5.6) and `langchain-openai` (1.5.1) are installed; the config decides, and the resolved model id is recorded on every `ExtractionResult` via `metadata.extraction_model` so a graph built with one model is distinguishable from one built with another.

Extraction is the highest-volume LLM call in the system — one per chunk, over the whole corpus — so it is the stage where the cost/quality tradeoff actually bites. A two-tier default, per your instruction to default cheap and escalate for hard documents:

| Tier | Model | Cost (in/out per MTok) | Use for |
|---|---|---|---|
| Bulk (default) | `claude-haiku-4-5` | $1 / $5 | The corpus body — prose, clean text, well-formed documents |
| Escalation | `claude-opus-5` | $5 / $25 | Documents flagged hard (below) |

`claude-sonnet-5` ($3 / $15) is the middle option if bulk quality proves short and Opus proves too expensive for the volume. Note that Haiku 4.5 has a **200K context window** against 1M for Sonnet 5 and Opus 5 — irrelevant per-chunk, but it constrains any future whole-document extraction path.

**Escalation triggers** — decide these mechanically, before spending tokens, not by retrying and hoping:

- Structural density: tables, forms, nested lists, multi-column layouts.
- Extraction quality signals from the bulk pass: high validation-failure rate on a document, zero extractions from chunks that clearly contain entities, malformed structured output after retry.
- Explicit per-source override in `_config/` — some sources are known-hard and should skip the cheap pass entirely.

Treat the tiering as a hypothesis to measure, not a settled choice. The right escalation rate is an empirical question and the honest way to answer it is a labelled sample: run both tiers over the same documents, compare extraction quality, and move the default if bulk quality doesn't hold. Two things to keep in view while doing that — a cheap model that misses relations produces a *quietly* incomplete graph, and stage 03's answers are bounded by what stage 01 found, so extraction misses are the most expensive kind of error in this pipeline even when the token bill looks better. Set `temperature=0` for reproducibility, and record model id and schema version on every result so a quality regression can be attributed rather than guessed at.

---

## `_config/` keys consumed

| Key | Purpose |
|---|---|
| `graph_schema.entity_types` | Admissible `entity_type` values (validation gate) |
| `graph_schema.relation_types` | Admissible `predicate` values, with subject/object type constraints |
| `graph_schema.version` | Stamped onto `metadata.schema_version` |
| `extraction.model` / `extraction.escalation_model` | Tier model ids |
| `extraction.escalation_rules` | Structural and per-source escalation triggers |
| `extraction.max_concurrency` | Bound on parallel chunk calls |
| `chunking.chunk_size` / `chunk_overlap` / `separators` | Splitter configuration |
| `prompts.extraction` | Prompt template path |

Loaded through `pydantic-settings` (2.15, installed), backed by `.env`. Provider API keys come from the environment (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`) and are never read directly in stage code — the provider integration handles that.

---

## Idempotency and provenance

`document_id` is derived from content, not from a filename or an ingestion counter: re-ingesting the same bytes must yield the same `document_id` so stage 02's `MERGE` operations collapse rather than duplicate. Carry `content_sha256` separately so a *changed* document at the same URI is detectable.

Extraction is expensive and re-runs are routine (schema changes, prompt changes, model changes). Cache on `(content_sha256, schema_version, model_id, prompt_version, chunking_params)` — anything less will either serve stale results after a prompt edit or re-bill the whole corpus after a trivial one.

---

## Module layout

```
stages/extraction/
├── CONTEXT.md          this file
├── __init__.py         public surface
├── schemas.py          payload models (stage boundary) + LLM-facing models
├── extractor.py        normalize · chunk · extract · assemble
└── loaders/            NOT YET WRITTEN — format-specific loaders → (text, metadata)
```

Validation lives with the models it constrains (`schemas.py`) rather than in a separate `validate.py`, so an invalid `ExtractionResult` is unconstructible instead of merely detectable. The load step is the remaining gap: `extract_document()` currently takes text that a caller has already read, which is fine for plain text and defers the harder formats until the corpus is known.

The directory was renamed from `01_extraction` so it is importable as `stages.extraction`; ordering lives in this documentation, not in the path.

---

## Testing

There is **no test suite yet** — this is the largest outstanding gap in the stage. The behavior below was verified once by a throwaway script (offsets round-trip, hallucinated `surface_form` dropped, paraphrased `evidence` dropped, unreported relation target dropped, cross-chunk merge, id determinism, dangling-reference rejection) and that script was not kept. Turning it into `pytest` cases is the next task.

- **Invariant tests** are the highest-value tests here and need no LLM: construct `ExtractionResult` objects that violate invariants 1–4 and assert they are rejected; feed `assemble()` output that violates 5–7 and assert the records land in `errors`.
- **Fake the chain, not the model.** `extract_document(..., extractor=...)` accepts any `Runnable`, so a `RunnableLambda` returning `ChunkExtraction` exercises the whole path with no provider call and no mocking library.
- **Offset round-trip**: for a corpus sample, assert every `surface_form` and `evidence` resolves to its recorded span in the source text. This catches normalization and chunking regressions, which are the failure modes most likely to silently break citations.
- **Golden extractions**: a small labelled set with known entities and relations, used to compare model tiers and detect prompt regressions. This is what makes the bulk-vs-escalation question answerable.
- **Mock the LLM** for pipeline tests; keep live-provider tests separate and opt-in.

---

## Open questions

Flagged rather than answered, since they need decisions this document can't make alone:

1. **Which source formats matter first?** The loader set is unspecified because the corpus is unspecified. PDF in particular determines whether page-level provenance and layout-aware chunking are required from day one.
2. **Is the graph schema fixed or open?** A closed schema (validation gate as written) trades recall for precision. An open schema with type discovery pushes significant burden onto stage 02's resolution and needs a different validation posture here.
3. **Does chunk-level extraction suffice, or is a document-level pass needed** for cross-chunk relations? A 1M-context model makes whole-document extraction viable for the escalation tier, at materially higher cost per document, and would change the chunking contract. Note the current design cannot express a relation between entities that never co-occur in a chunk — overlap mitigates this, it does not solve it.
4. **Where does `_config/` end and code begin?** `ExtractionConfig` is the intended load target, but `SYSTEM_PROMPT` and the escalation-trigger logic still live in `extractor.py`. Prompts in particular should move to `_config/` with a `prompt_version` so the cache key in *Idempotency* is computable.
5. **Should type validation be a hard gate (invariant 9)?** Allowed types currently reach the model through the prompt only; nothing rejects an off-schema type after the fact. Closing this is a few lines, but it presupposes the answer to question 2.
