"""Payload models for Stage 1 (extraction).

Two families of model live here:

* The **stage boundary** — ``ExtractionResult`` and its parts. This is the
  contract stage 02 (graph indexing) consumes; see ``CONTEXT.md``.
* The **LLM boundary** — ``ChunkExtraction`` and its parts, the schema handed to
  ``with_structured_output()``. The model works in local names and never sees
  document ids or chunk ids, so it cannot invent them; ``extractor`` resolves
  names to ids and stitches per-chunk results into an ``ExtractionResult``.

Everything here is *candidate* data. Entity resolution, deduplication across
documents, and canonicalization belong to stage 02 — this stage only reports
what the text appears to say, with enough provenance for stage 02 to adjudicate.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "CandidateEntity",
    "CandidateRelation",
    "Chunk",
    "ChunkExtraction",
    "DocumentMetadata",
    "ExtractedEntity",
    "ExtractedRelation",
    "ExtractionError",
    "ExtractionResult",
    "entity_id",
    "relation_key",
]

# A non-empty, stripped string. Used for every field where an empty value would
# be a silent data-quality failure rather than a legitimate absence.
NonEmptyStr = Annotated[str, Field(min_length=1)]

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _fingerprint(*parts: str) -> str:
    """Stable short digest of ``parts``.

    Deterministic across runs and processes (unlike :func:`hash`), so re-running
    extraction over unchanged input produces identical ids and stage 02's
    ``MERGE`` collapses rather than duplicating.
    """
    joined = "\x00".join(parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def entity_id(entity_type: str, canonical_name: str) -> str:
    """Derive a deterministic entity id from its type and canonical name.

    Case- and punctuation-insensitive, so ``"Acme Corp"`` and ``"ACME  Corp"``
    collide into one candidate *within a document*. This is deliberately a
    cheap, purely lexical merge — it is not entity resolution. ``"Acme Corp"``
    and ``"Acme Corporation"`` remain two candidates, and reconciling them is
    stage 02's job.
    """
    slug = _SLUG_RE.sub("-", canonical_name.casefold()).strip("-")
    return f"ent_{_fingerprint(entity_type.casefold(), slug)}"


def relation_key(source_entity_id: str, target_entity_id: str, relation_type: str) -> str:
    """Identity of a relation, for deduplication within a document."""
    return f"{source_entity_id}|{relation_type.casefold()}|{target_entity_id}"


# --------------------------------------------------------------------------- #
# Stage boundary: what stage 02 consumes
# --------------------------------------------------------------------------- #


class Chunk(BaseModel):
    """A span of the normalized document text, with the offsets to cite it."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Deterministic: f'{document_id}:{index}'.")
    text: NonEmptyStr
    start_char: int = Field(ge=0, description="Offset into the normalized document text.")
    end_char: int = Field(gt=0)

    @model_validator(mode="after")
    def _offsets_span_text(self) -> Chunk:
        if self.end_char <= self.start_char:
            raise ValueError(f"chunk {self.id}: end_char must be greater than start_char")
        span = self.end_char - self.start_char
        if span != len(self.text):
            raise ValueError(
                f"chunk {self.id}: offsets span {span} characters but text is "
                f"{len(self.text)} — offsets would not resolve to this text"
            )
        return self


class CandidateEntity(BaseModel):
    """An entity the model believes the text mentions.

    ``chunk_ids`` is a list because the same entity is normally mentioned in
    several chunks of one document (and overlapping chunks guarantee it). Every
    mention that produced this candidate is recorded, so stage 02 can weigh a
    candidate by how much of the document supports it.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    entity_type: NonEmptyStr
    canonical_name: NonEmptyStr = Field(
        description="Best single name for the entity. A hint for stage 02, not a decision."
    )
    surface_form: NonEmptyStr = Field(
        description="Verbatim span as it appears in the source text."
    )
    description: str = Field(
        default="", description="What the source says about the entity. May be empty."
    )
    chunk_ids: list[str] = Field(min_length=1, description="Chunks mentioning this entity.")

    @field_validator("chunk_ids")
    @classmethod
    def _unique_chunk_ids(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(value))


class CandidateRelation(BaseModel):
    """A directed, typed edge the model believes the text asserts.

    ``evidence`` is the load-bearing field: it must be a verbatim span from one
    of ``chunk_ids``, which is what makes a downstream answer citable. The
    extractor drops relations whose evidence cannot be found in the source
    rather than passing an unciteable edge to stage 02.
    """

    model_config = ConfigDict(extra="forbid")

    source_entity_id: str
    target_entity_id: str
    relation_type: NonEmptyStr
    description: str = Field(default="", description="How the source characterizes the edge.")
    evidence: NonEmptyStr = Field(description="Verbatim span asserting the relation.")
    chunk_ids: list[str] = Field(min_length=1)

    @field_validator("chunk_ids")
    @classmethod
    def _unique_chunk_ids(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(value))

    @model_validator(mode="after")
    def _no_self_loop(self) -> CandidateRelation:
        if self.source_entity_id == self.target_entity_id:
            raise ValueError(
                f"relation {self.relation_type}: source and target are the same entity "
                f"({self.source_entity_id})"
            )
        return self

    @property
    def key(self) -> str:
        return relation_key(self.source_entity_id, self.target_entity_id, self.relation_type)


class ExtractionError(BaseModel):
    """A non-fatal failure, recorded rather than raised.

    A chunk that fails extraction, or a record dropped for failing a verbatim
    check, must leave a trace: a partial result is valid output, a silent gap is
    not. Track these — the drop rate is the signal that a prompt, a schema, or a
    model tier is not working.
    """

    model_config = ConfigDict(extra="forbid")

    chunk_id: str | None = None
    stage: str = Field(description="Where it failed, e.g. 'llm_call', 'evidence_check'.")
    message: str
    payload: str | None = Field(
        default=None, description="Offending value, truncated, for debugging."
    )


class DocumentMetadata(BaseModel):
    """Document-level provenance, stamped onto every result."""

    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(description="Content-derived, so re-ingestion is idempotent.")
    uri: NonEmptyStr = Field(description="File path, URL, or connector-specific locator.")
    content_sha256: str = Field(min_length=64, max_length=64)
    mime_type: str = "text/plain"
    title: str | None = None
    source_timestamp: datetime | None = Field(
        default=None, description="From the document itself, when available."
    )
    ingested_at: datetime | None = Field(default=None, description="When this run processed it.")
    extraction_model: str | None = Field(
        default=None, description="Exact model id used, for attributing quality changes."
    )
    schema_version: str | None = Field(default=None, description="_config/ graph schema version.")


class ExtractionResult(BaseModel):
    """Stage 1's output for one document. The 01 → 02 payload contract."""

    model_config = ConfigDict(extra="forbid")

    metadata: DocumentMetadata
    chunks: list[Chunk] = Field(default_factory=list)
    entities: list[CandidateEntity] = Field(default_factory=list)
    relations: list[CandidateRelation] = Field(default_factory=list)
    errors: list[ExtractionError] = Field(default_factory=list)

    @model_validator(mode="after")
    def _referential_integrity(self) -> ExtractionResult:
        """Enforce that the result is internally consistent.

        These checks are the reason this stage can hand stage 02 candidate data
        without stage 02 having to defend against dangling references.
        """
        chunk_ids = {chunk.id for chunk in self.chunks}
        if len(chunk_ids) != len(self.chunks):
            raise ValueError("duplicate chunk ids")

        entity_ids = {entity.id for entity in self.entities}
        if len(entity_ids) != len(self.entities):
            raise ValueError("duplicate entity ids")

        for entity in self.entities:
            unknown = set(entity.chunk_ids) - chunk_ids
            if unknown:
                raise ValueError(f"entity {entity.id} references unknown chunks: {sorted(unknown)}")

        for relation in self.relations:
            for role, ref in (
                ("source", relation.source_entity_id),
                ("target", relation.target_entity_id),
            ):
                if ref not in entity_ids:
                    raise ValueError(
                        f"relation {relation.relation_type} references unknown "
                        f"{role} entity: {ref}"
                    )
            unknown = set(relation.chunk_ids) - chunk_ids
            if unknown:
                raise ValueError(
                    f"relation {relation.relation_type} references unknown "
                    f"chunks: {sorted(unknown)}"
                )
        return self

    def chunk_by_id(self, chunk_id: str) -> Chunk | None:
        return next((chunk for chunk in self.chunks if chunk.id == chunk_id), None)


# --------------------------------------------------------------------------- #
# LLM boundary: what with_structured_output() fills in
# --------------------------------------------------------------------------- #
#
# Kept deliberately narrow. The model is asked only for what it can read off
# the text in front of it — no ids, no offsets, no cross-chunk reasoning. Every
# field description below is part of the prompt the provider sends, so they are
# written as instructions to the model, not as notes to a reader.


class ExtractedEntity(BaseModel):
    """One entity as reported by the model for a single chunk."""

    model_config = ConfigDict(extra="forbid")

    name: NonEmptyStr = Field(
        description="The entity's name, normalized to its fullest form used in this text."
    )
    entity_type: NonEmptyStr = Field(
        description="The entity's type, chosen from the allowed types listed in the instructions."
    )
    surface_form: NonEmptyStr = Field(
        description=(
            "The exact substring of the provided text that names this entity. "
            "Copy it verbatim, character for character. Do not paraphrase, "
            "correct, expand, or reformat it."
        )
    )
    description: str = Field(
        default="",
        description=(
            "What this text says about the entity, in one sentence. "
            "Leave empty if the text only names it."
        ),
    )


class ExtractedRelation(BaseModel):
    """One relation as reported by the model for a single chunk."""

    model_config = ConfigDict(extra="forbid")

    source: NonEmptyStr = Field(
        description="The 'name' of the entity the relation points from, exactly as listed above."
    )
    target: NonEmptyStr = Field(
        description="The 'name' of the entity the relation points to, exactly as listed above."
    )
    relation_type: NonEmptyStr = Field(
        description=(
            "The relation's type, chosen from the allowed types listed in the instructions."
        )
    )
    description: str = Field(
        default="", description="How this text characterizes the relation, in one sentence."
    )
    evidence: NonEmptyStr = Field(
        description=(
            "The exact sentence or clause from the provided text that asserts this "
            "relation. Copy it verbatim, character for character. If no single span "
            "states the relation, omit the relation entirely rather than composing one."
        )
    )


class ChunkExtraction(BaseModel):
    """The model's full report for one chunk.

    Both lists default to empty and that is a normal, expected outcome: most
    chunks of a real corpus assert nothing extractable, and a model that feels
    obliged to fill these will invent content.
    """

    model_config = ConfigDict(extra="forbid")

    entities: list[ExtractedEntity] = Field(
        default_factory=list,
        description="Entities named in this text. Empty list if there are none.",
    )
    relations: list[ExtractedRelation] = Field(
        default_factory=list,
        description=(
            "Relations asserted by this text, between entities listed above. "
            "Empty list if the text asserts none."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def deserialize_stringified_json(cls, data):
        """Deserialize JSON-stringified fields from LLM tool calls.

        Some LLMs (like Claude) may return stringified JSON for list/dict fields
        instead of native structures. This validator handles that transparently.
        """
        if not isinstance(data, dict):
            return data

        for field_name in ("entities", "relations"):
            if field_name in data and isinstance(data[field_name], str):
                try:
                    data[field_name] = json.loads(data[field_name])
                except (json.JSONDecodeError, ValueError):
                    pass

        return data
