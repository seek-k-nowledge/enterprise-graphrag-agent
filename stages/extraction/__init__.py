"""Stage 1 — extraction: parse, chunk, and extract candidate entities/relations.

Emits ``ExtractionResult`` for stage 02. Never touches Neo4j.
"""

from stages.extraction.extractor import (
    ExtractionConfig,
    chunk_text,
    extract_document,
    normalize_text,
)
from stages.extraction.schemas import (
    CandidateEntity,
    CandidateRelation,
    Chunk,
    ChunkExtraction,
    DocumentMetadata,
    ExtractionError,
    ExtractionResult,
)

__all__ = [
    "CandidateEntity",
    "CandidateRelation",
    "Chunk",
    "ChunkExtraction",
    "DocumentMetadata",
    "ExtractionConfig",
    "ExtractionError",
    "ExtractionResult",
    "chunk_text",
    "extract_document",
    "normalize_text",
]
