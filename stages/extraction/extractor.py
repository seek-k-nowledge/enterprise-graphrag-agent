"""Chunking and LLM-backed triple extraction for Stage 1.

The flow, per document:

1. :func:`normalize_text` — stabilize the text so offsets stay meaningful.
2. :func:`chunk_text` — split with ``RecursiveCharacterTextSplitter``, keeping
   the character offsets needed to cite a passage later.
3. :func:`extract_chunks` — one structured-output LLM call per chunk, run
   concurrently. Per-chunk failures are recorded, never fatal.
4. :func:`assemble` — resolve the model's local entity names to deterministic
   ids, enforce the verbatim-span checks, merge duplicates within the document,
   and emit an :class:`ExtractionResult`.

:func:`extract_document` runs all four. Nothing here touches Neo4j: this stage
emits candidate data and stage 02 owns resolution and persistence.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import unicodedata
from datetime import datetime, timezone

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, ConfigDict, Field

from stages.extraction.schemas import (
    CandidateEntity,
    CandidateRelation,
    Chunk,
    ChunkExtraction,
    DocumentMetadata,
    ExtractionError,
    ExtractionResult,
    entity_id,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ExtractionConfig",
    "assemble",
    "build_extractor",
    "chunk_text",
    "extract_chunks",
    "extract_document",
    "normalize_text",
]

# Default tiering, per the model-selection section of CONTEXT.md: cheap model
# for the corpus body, stronger model for documents flagged hard. Both are
# overridable here and belong in _config/ once that exists.
# Using Groq models for cost-effectiveness
DEFAULT_MODEL = "llama-3.3-70b-versatile"
ESCALATION_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """\
You extract a knowledge graph from text. You will be given one passage from a \
larger document.

Identify the entities the passage names and the relations it asserts between \
them, then report them in the required structure.

Allowed entity types (use these exactly; ignore anything that fits none of them):
{entity_types}

Allowed relation types (use these exactly; ignore anything that fits none of them):
{relation_types}

Rules:
- Extract only what this passage states. Do not use outside knowledge, and do \
not infer relations that the passage merely implies.
- Copy `surface_form` and `evidence` verbatim from the passage, character for \
character. Records whose spans are not found in the passage are discarded, so \
paraphrasing loses the record.
- Every relation's `source` and `target` must match the `name` of an entity you \
report in the same response.
- A passage that names no allowed entities, or asserts no allowed relations, is \
normal and expected. Return empty lists for those. Do not invent content to \
fill them.\
"""

USER_PROMPT = "Passage:\n\n{chunk_text}"

# Conservative starting point for prose. Overlap is deliberate: a relation whose
# subject and object straddle a boundary is otherwise unrecoverable. The cost is
# the same triple arriving from adjacent chunks, which `assemble` merges.
DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 200

_CONTROL_CHARS = {"\t", "\n"}
_MAX_PAYLOAD_CHARS = 300


class ExtractionConfig(BaseModel):
    """Stage 1 configuration.

    Mirrors the ``extraction.*``, ``chunking.*`` and ``graph_schema.*`` keys
    described in ``CONTEXT.md``. Constructed by the caller for now; it is the
    shape ``_config/`` should load into.
    """

    model_config = ConfigDict(extra="forbid")

    model: str = DEFAULT_MODEL
    model_provider: str | None = Field(
        default=None, description="Inferred from the model name when omitted."
    )
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    entity_types: list[str] = Field(min_length=1)
    relation_types: list[str] = Field(min_length=1)
    chunk_size: int = Field(default=DEFAULT_CHUNK_SIZE, gt=0)
    chunk_overlap: int = Field(default=DEFAULT_CHUNK_OVERLAP, ge=0)
    max_concurrency: int = Field(default=4, gt=0)
    schema_version: str | None = None

    def model_post_init(self, _context: object) -> None:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")


def normalize_text(text: str) -> str:
    """Stabilize text so character offsets remain resolvable.

    Deliberately minimal: line endings and NFC composition only. Aggressive
    whitespace collapsing or paragraph reflow would break the guarantee that a
    chunk's ``start_char``/``end_char`` resolve back to a citable span, which is
    what the whole provenance chain rests on. Normalization must stay
    deterministic — offsets are meaningless against text that cannot be
    reproduced.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return "".join(
        char
        for char in text
        if char in _CONTROL_CHARS or unicodedata.category(char) not in {"Cc", "Cf"}
    )


def _document_id(content_sha256: str) -> str:
    return f"doc_{content_sha256[:16]}"


def chunk_text(text: str, document_id: str, config: ExtractionConfig) -> list[Chunk]:
    """Split normalized text into chunks that carry their source offsets.

    ``add_start_index`` gives the splitter's own offset. It is verified against
    the text rather than trusted: whitespace stripping can shift a boundary, and
    an offset that does not resolve is worse than no offset at all — it produces
    citations that point at the wrong passage.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        add_start_index=True,
    )

    chunks: list[Chunk] = []
    search_from = 0
    for index, doc in enumerate(splitter.create_documents([text])):
        body = doc.page_content
        start = doc.metadata.get("start_index", -1)
        if start < 0 or text[start : start + len(body)] != body:
            # Recover rather than emit a bad offset. Searching forward from the
            # previous chunk's start keeps repeated passages in document order.
            start = text.find(body, search_from)
            if start < 0:
                start = text.find(body)
            if start < 0:
                logger.warning(
                    "chunk %d of %s could not be located in the source text; skipping",
                    index,
                    document_id,
                )
                continue
        search_from = start
        chunks.append(
            Chunk(
                id=f"{document_id}:{index}",
                text=body,
                start_char=start,
                end_char=start + len(body),
            )
        )
    return chunks


def build_extractor(config: ExtractionConfig) -> Runnable:
    """Build the prompt → model → structured-output chain.

    ``with_structured_output`` pushes schema conformance down to the provider
    integration, so there is no hand-rolled JSON parsing or retry loop here.
    The model is instantiated lazily by the caller's first invocation, so
    building a chain does not require credentials.
    """
    try:
        from langchain_groq import ChatGroq

        model: BaseChatModel = ChatGroq(
            model_name=config.model,
            temperature=config.temperature,
            api_key=os.getenv("GROQ_API_KEY"),
        )
    except ImportError:
        # Fallback to init_chat_model
        model: BaseChatModel = init_chat_model(
            config.model,
            model_provider=config.model_provider or "groq",
            temperature=config.temperature,
        )
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", USER_PROMPT)]
    ).partial(
        entity_types="\n".join(f"- {t}" for t in config.entity_types),
        relation_types="\n".join(f"- {t}" for t in config.relation_types),
    )
    return prompt | model.with_structured_output(ChunkExtraction)


def extract_chunks(
    chunks: list[Chunk],
    config: ExtractionConfig,
    extractor: Runnable | None = None,
) -> tuple[dict[str, ChunkExtraction], list[ExtractionError]]:
    """Run extraction over ``chunks``, returning results keyed by chunk id.

    Chunks are independent, so they run concurrently up to
    ``config.max_concurrency``. ``return_exceptions`` keeps one failed chunk
    from discarding a whole document's work — the failure is recorded and the
    remaining chunks still land.
    """
    if not chunks:
        return {}, []

    extractor = extractor or build_extractor(config)
    outcomes = extractor.batch(
        [{"chunk_text": chunk.text} for chunk in chunks],
        config={"max_concurrency": config.max_concurrency},
        return_exceptions=True,
    )

    results: dict[str, ChunkExtraction] = {}
    errors: list[ExtractionError] = []
    for chunk, outcome in zip(chunks, outcomes, strict=True):
        if isinstance(outcome, BaseException):
            logger.warning("extraction failed for chunk %s: %s", chunk.id, outcome)
            errors.append(
                ExtractionError(
                    chunk_id=chunk.id,
                    stage="llm_call",
                    message=f"{type(outcome).__name__}: {outcome}",
                )
            )
        elif isinstance(outcome, ChunkExtraction):
            results[chunk.id] = outcome
        else:
            # Structured output should make this unreachable; recorded rather
            # than asserted so a provider-side change surfaces as data.
            errors.append(
                ExtractionError(
                    chunk_id=chunk.id,
                    stage="llm_call",
                    message=f"unexpected result type {type(outcome).__name__}",
                    payload=_truncate(repr(outcome)),
                )
            )
    return results, errors


def _truncate(value: str) -> str:
    if len(value) <= _MAX_PAYLOAD_CHARS:
        return value
    return f"{value[:_MAX_PAYLOAD_CHARS]}…"


def _contains(haystack: str, needle: str) -> bool:
    """Verbatim containment, tolerating only whitespace-run differences.

    Models reliably reproduce wording but not always the exact run of spaces or
    newlines inside a span, particularly across a line wrap. Collapsing
    whitespace on both sides keeps that from discarding otherwise-good records,
    while still rejecting paraphrase — which is the thing the check exists to
    catch.
    """
    if needle in haystack:
        return True
    collapse = re.compile(r"\s+")
    return collapse.sub(" ", needle).strip() in collapse.sub(" ", haystack)


def assemble(
    metadata: DocumentMetadata,
    chunks: list[Chunk],
    per_chunk: dict[str, ChunkExtraction],
    errors: list[ExtractionError] | None = None,
) -> ExtractionResult:
    """Turn per-chunk model output into one validated ``ExtractionResult``.

    Three things happen here, and each drops records rather than passing
    doubtful data downstream:

    * **Verbatim enforcement.** An entity whose ``surface_form`` is not in its
      chunk, or a relation whose ``evidence`` is not in its chunk, is discarded.
      Such a record cannot be cited, and an uncitable edge is worse than a
      missing one.
    * **Local resolution.** The model reports relations by entity *name*; those
      are resolved against the entities it reported for the same chunk. A
      relation naming an entity the model did not report is discarded.
    * **Within-document merge.** The same entity or relation seen in several
      chunks becomes one candidate carrying every supporting chunk id. This is a
      lexical merge only — cross-document resolution is stage 02's.
    """
    errors = list(errors or [])
    chunk_text_by_id = {chunk.id: chunk.text for chunk in chunks}

    entities: dict[str, CandidateEntity] = {}
    relations: dict[str, CandidateRelation] = {}

    for chunk_id, extraction in per_chunk.items():
        source_text = chunk_text_by_id.get(chunk_id)
        if source_text is None:
            errors.append(
                ExtractionError(
                    chunk_id=chunk_id,
                    stage="assemble",
                    message="extraction result for a chunk that is not in the document",
                )
            )
            continue

        # Entities first: relations are resolved against this chunk's entities.
        ids_by_name: dict[str, str] = {}
        for extracted in extraction.entities:
            if not _contains(source_text, extracted.surface_form):
                errors.append(
                    ExtractionError(
                        chunk_id=chunk_id,
                        stage="surface_form_check",
                        message=(
                            f"surface_form for entity {extracted.name!r} is not a verbatim "
                            f"span of the chunk"
                        ),
                        payload=_truncate(extracted.surface_form),
                    )
                )
                continue

            candidate_id = entity_id(extracted.entity_type, extracted.name)
            ids_by_name[extracted.name.casefold()] = candidate_id

            existing = entities.get(candidate_id)
            if existing is None:
                entities[candidate_id] = CandidateEntity(
                    id=candidate_id,
                    entity_type=extracted.entity_type,
                    canonical_name=extracted.name,
                    surface_form=extracted.surface_form,
                    description=extracted.description,
                    chunk_ids=[chunk_id],
                )
            else:
                existing.chunk_ids = list(dict.fromkeys([*existing.chunk_ids, chunk_id]))
                # Keep the fullest description seen; length is a crude proxy for
                # informativeness, but it beats last-write-wins.
                if len(extracted.description) > len(existing.description):
                    existing.description = extracted.description

        for extracted in extraction.relations:
            source_id = ids_by_name.get(extracted.source.casefold())
            target_id = ids_by_name.get(extracted.target.casefold())
            if source_id is None or target_id is None:
                missing = extracted.source if source_id is None else extracted.target
                errors.append(
                    ExtractionError(
                        chunk_id=chunk_id,
                        stage="relation_resolution",
                        message=(
                            f"relation {extracted.relation_type!r} references entity "
                            f"{missing!r}, which was not reported for this chunk"
                        ),
                    )
                )
                continue
            if source_id == target_id:
                errors.append(
                    ExtractionError(
                        chunk_id=chunk_id,
                        stage="relation_resolution",
                        message=(
                            f"relation {extracted.relation_type!r} points an entity at "
                            f"itself ({extracted.source!r})"
                        ),
                    )
                )
                continue
            if not _contains(source_text, extracted.evidence):
                errors.append(
                    ExtractionError(
                        chunk_id=chunk_id,
                        stage="evidence_check",
                        message=(
                            f"evidence for relation {extracted.relation_type!r} is not a "
                            f"verbatim span of the chunk"
                        ),
                        payload=_truncate(extracted.evidence),
                    )
                )
                continue

            candidate = CandidateRelation(
                source_entity_id=source_id,
                target_entity_id=target_id,
                relation_type=extracted.relation_type,
                description=extracted.description,
                evidence=extracted.evidence,
                chunk_ids=[chunk_id],
            )
            existing = relations.get(candidate.key)
            if existing is None:
                relations[candidate.key] = candidate
            else:
                existing.chunk_ids = list(dict.fromkeys([*existing.chunk_ids, chunk_id]))

    return ExtractionResult(
        metadata=metadata,
        chunks=chunks,
        entities=list(entities.values()),
        relations=list(relations.values()),
        errors=errors,
    )


def extract_document(
    text: str,
    uri: str,
    config: ExtractionConfig,
    *,
    title: str | None = None,
    mime_type: str = "text/plain",
    source_timestamp: datetime | None = None,
    extractor: Runnable | None = None,
) -> ExtractionResult:
    """Extract one document end to end.

    ``text`` is raw document text; normalization happens here so that the
    offsets on the returned chunks are offsets into the *normalized* text. A
    caller that needs to resolve those offsets later must normalize the same way
    — :func:`normalize_text` is deterministic for that reason.

    Passing ``extractor`` reuses one chain across documents (and is the seam
    tests use to substitute a fake, avoiding any provider call).
    """
    normalized = normalize_text(text)
    content_sha256 = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    document_id = _document_id(content_sha256)

    metadata = DocumentMetadata(
        document_id=document_id,
        uri=uri,
        content_sha256=content_sha256,
        mime_type=mime_type,
        title=title,
        source_timestamp=source_timestamp,
        ingested_at=datetime.now(timezone.utc),
        extraction_model=config.model,
        schema_version=config.schema_version,
    )

    chunks = chunk_text(normalized, document_id, config)
    if not chunks:
        logger.info("document %s produced no chunks", uri)
        return ExtractionResult(metadata=metadata)

    per_chunk, errors = extract_chunks(chunks, config, extractor=extractor)
    result = assemble(metadata, chunks, per_chunk, errors)

    logger.info(
        "extracted %s: %d chunks, %d entities, %d relations, %d errors",
        uri,
        len(result.chunks),
        len(result.entities),
        len(result.relations),
        len(result.errors),
    )
    return result
