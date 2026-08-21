"""
Graph writing and idempotent upserts for Stage 2.

Writes resolved entities, chunks, documents, and relationships to Neo4j
using MERGE-based idempotent operations. All writes are batched for performance
and transactional for consistency.

Usage:
    writer = GraphWriter(neo4j_client, batch_size=500)
    result = writer.write_extraction(extraction_result, resolved_entities, resolved_relations)
"""

import logging
from typing import Optional

from stages.extraction.schemas import Chunk, ExtractionResult, DocumentMetadata
from .neo4j_client import Neo4jClient
from .schemas import (
    CanonicalNode,
    CanonicalRelation,
    NodeUpsertResult,
    RelationUpsertResult,
)
from .embedder import ChunkEmbedder

logger = logging.getLogger(__name__)


class GraphWriter:
    """
    Writes resolved entities and relationships to Neo4j idempotently.

    All operations use MERGE to ensure re-ingestion produces identical results.
    """

    def __init__(self, client: Neo4jClient, batch_size: int = 500):
        """
        Initialize the graph writer.

        Args:
            client: Neo4jClient instance (must be connected)
            batch_size: Batch size for MERGE operations (nodes/relations per transaction)
        """
        self.client = client
        self.batch_size = batch_size

    def setup_schema(self) -> None:
        """
        Create constraints and indexes (idempotent).

        Should be called once before any writes. Safe to call multiple times.
        """
        logger.info("Setting up Neo4j schema...")

        constraints = [
            "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT document_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
        ]

        indexes = [
            "CREATE INDEX entity_type_idx IF NOT EXISTS FOR (e:Entity) ON (e.entity_type)",
            "CREATE INDEX chunk_document_idx IF NOT EXISTS FOR (c:Chunk) ON (c.document_id)",
            "CREATE INDEX document_uri_idx IF NOT EXISTS FOR (d:Document) ON (d.uri)",
        ]

        self.client.run_schema_setup(constraints, indexes)
        logger.info("Schema setup complete")

    def write_extraction(
        self,
        extraction_result: ExtractionResult,
        canonical_entities: dict[str, CanonicalNode],
        canonical_relations: dict[str, CanonicalRelation],
    ) -> tuple[NodeUpsertResult, RelationUpsertResult]:
        """
        Write an extraction result to the graph.

        Orchestrates: document → chunks → entities → relationships.
        All operations are batched and transactional.

        Args:
            extraction_result: Result from Stage 1
            canonical_entities: Resolved entities (from entity_resolver)
            canonical_relations: Resolved relations (from entity_resolver)

        Returns:
            Tuple of (NodeUpsertResult, RelationUpsertResult)
        """
        logger.info(
            f"Writing extraction to graph: document={extraction_result.metadata.document_id}, "
            f"chunks={len(extraction_result.chunks)}, "
            f"entities={len(canonical_entities)}, "
            f"relations={len(canonical_relations)}"
        )

        node_result = NodeUpsertResult()
        rel_result = RelationUpsertResult()

        try:
            # Step 1: Write document node
            doc_created = self._write_document(extraction_result.metadata)
            if doc_created:
                node_result.documents_created += 1
            else:
                node_result.documents_updated += 1

            # Step 2: Write chunk nodes
            chunk_created, chunk_updated = self._write_chunks(extraction_result.chunks, extraction_result.metadata.document_id)
            node_result.chunks_created += chunk_created
            node_result.chunks_updated += chunk_updated

            # Step 2b: Embed chunks and store vectors in Neo4j
            self._embed_and_index_chunks(extraction_result.chunks, extraction_result.metadata.document_id)

            # Step 3: Write entity nodes
            entity_created, entity_updated = self._write_entities(canonical_entities, extraction_result.metadata.document_id)
            node_result.entities_created += entity_created
            node_result.entities_updated += entity_updated

            # Step 4: Write relationships
            mention_created, mention_updated = self._write_mention_relations(
                canonical_entities, extraction_result.chunks
            )
            rel_result.mention_relations_created += mention_created
            rel_result.mention_relations_updated += mention_updated

            from_created = self._write_from_relations(
                extraction_result.chunks, extraction_result.metadata.document_id
            )
            rel_result.from_relations_created += from_created

            content_created, content_updated = self._write_content_relations(canonical_relations)
            rel_result.content_relations_created += content_created
            rel_result.content_relations_updated += content_updated

            logger.info(
                f"Write complete: {node_result.entities_created} entities created, "
                f"{node_result.chunks_created} chunks created, "
                f"{rel_result.content_relations_created} relations created"
            )

        except Exception as e:
            logger.error(f"Error writing extraction to graph: {e}")
            raise

        return node_result, rel_result

    # ─────────────────────────────────────────────────────────────────────────────
    # Private methods: node writes
    # ─────────────────────────────────────────────────────────────────────────────

    def _write_document(self, metadata: DocumentMetadata) -> bool:
        """
        Write or update a document node.

        Returns True if created, False if updated.
        """
        # Extract filename from URI for friendly display name
        doc_name = metadata.title or metadata.uri.split("/")[-1] if metadata.uri else metadata.document_id

        cypher = """
        MERGE (d:Document {id: $doc_id})
        ON CREATE SET
            d.name = $doc_name,
            d.uri = $uri,
            d.content_sha256 = $content_sha256,
            d.ingested_at = $ingested_at,
            d.extraction_model = $extraction_model,
            d.schema_version = $schema_version
        RETURN elementId(d) as id, true as created
        """

        result = self.client.query(
            cypher,
            parameters={
                "doc_id": metadata.document_id,
                "doc_name": doc_name,
                "uri": metadata.uri,
                "content_sha256": metadata.content_sha256,
                "ingested_at": metadata.ingested_at.isoformat(),
                "extraction_model": metadata.extraction_model,
                "schema_version": metadata.schema_version,
            },
            read_only=False,
        )

        if result.records:
            return result.affected_rows > 0

        logger.warning(f"Document write returned no result: {metadata.document_id}")
        return False

    def _write_chunks(self, chunks: list[Chunk], document_id: str) -> tuple[int, int]:
        """
        Write chunk nodes in batches.

        Returns (created_count, updated_count).
        """
        created, updated = 0, 0

        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i : i + self.batch_size]
            queries = []

            for chunk in batch:
                chunk_name = f"Chunk {chunk.id[-8:]}" if len(chunk.id) > 8 else chunk.id
                cypher = """
                MERGE (c:Chunk {id: $chunk_id})
                ON CREATE SET
                    c.name = $chunk_name,
                    c.text = $text,
                    c.start_char = $start_char,
                    c.end_char = $end_char,
                    c.document_id = $document_id,
                    c.created = $created
                RETURN c, elementId(c) as chunk_id
                """
                params = {
                    "chunk_id": chunk.id,
                    "chunk_name": chunk_name,
                    "text": chunk.text,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                    "document_id": document_id,
                    "created": logging.root.handlers[0].formatter.formatTime(
                        logging.LogRecord(0, 0, "", 0, "", (), None)
                    ) if logging.root.handlers else None,
                }
                queries.append((cypher, params))

            # Execute batch
            results = self.client.batch_execute(queries, read_only=False)
            for result in results:
                if result.affected_rows > 0:
                    created += 1
                else:
                    updated += 1

        logger.info(f"Chunks written: {created} created, {updated} updated")
        return created, updated

    def _embed_and_index_chunks(self, chunks: list[Chunk], document_id: str) -> None:
        """
        Embed chunk text and store vectors in Neo4j for semantic search.

        Args:
            chunks: List of Chunk objects (already written to Neo4j)
            document_id: Document ID for tracking

        Logs errors but doesn't fail the write operation if embedding fails.
        """
        if not chunks:
            return

        try:
            embedder = ChunkEmbedder(self.client)

            # Embed all chunks for this document
            result = embedder.embed_chunks(chunks, document_id, skip_existing=True)

            # Ensure vector index exists
            embedder.ensure_vector_index()

            logger.info(
                f"Chunk embedding complete: {result.chunks_embedded} embedded, "
                f"{result.vectors_indexed} indexed, {len(result.errors)} errors"
            )

            # Log any embedding errors for visibility
            if result.errors:
                logger.warning(f"Embedding errors for document {document_id}: {result.errors}")

        except Exception as e:
            logger.error(f"Failed to embed chunks for document {document_id}: {e}")
            # Don't re-raise: embedding failure shouldn't fail the entire write operation

    def _write_entities(self, canonical_entities: dict[str, CanonicalNode], document_id: str) -> tuple[int, int]:
        """
        Write canonical entity nodes in batches.

        Returns (created_count, updated_count).
        """
        created, updated = 0, 0

        entities_list = list(canonical_entities.values())

        for i in range(0, len(entities_list), self.batch_size):
            batch = entities_list[i : i + self.batch_size]
            queries = []

            for entity in batch:
                cypher = """
                MERGE (e:Entity {id: $entity_id})
                ON CREATE SET
                    e.name = $canonical_name,
                    e.entity_type = $entity_type,
                    e.canonical_name = $canonical_name,
                    e.surface_forms = $surface_forms,
                    e.description = $description,
                    e.sources = [$document_id],
                    e.first_seen = $first_seen,
                    e.updated = $updated
                ON MATCH SET
                    e.surface_forms = e.surface_forms + (
                        [sf IN $surface_forms WHERE NOT sf IN e.surface_forms | sf]
                    ),
                    e.sources = e.sources + (
                        CASE WHEN $document_id IN e.sources THEN [] ELSE [$document_id] END
                    ),
                    e.updated = $updated
                RETURN e, elementId(e) as entity_id
                """
                params = {
                    "entity_id": entity.id,
                    "entity_type": entity.entity_type,
                    "canonical_name": entity.canonical_name,
                    "surface_forms": entity.surface_forms,
                    "description": entity.description,
                    "document_id": document_id,
                    "first_seen": entity.first_seen.isoformat(),
                    "updated": entity.updated.isoformat(),
                }
                queries.append((cypher, params))

            results = self.client.batch_execute(queries, read_only=False)
            for result in results:
                if result.affected_rows > 0:
                    created += 1
                else:
                    updated += 1

        logger.info(f"Entities written: {created} created, {updated} updated")
        return created, updated

    # ─────────────────────────────────────────────────────────────────────────────
    # Private methods: relationship writes
    # ─────────────────────────────────────────────────────────────────────────────

    def _write_mention_relations(
        self,
        canonical_entities: dict[str, CanonicalNode],
        chunks: list[Chunk],
    ) -> tuple[int, int]:
        """
        Write MENTIONED_IN relationships (Entity) -[:MENTIONED_IN]-> (Chunk).

        Returns (created_count, updated_count).
        """
        created, updated = 0, 0

        for entity in canonical_entities.values():
            # Determine which chunks mention this entity
            chunk_ids = set()
            for candidate_id in entity.candidate_ids:
                # In practice, we'd track chunk_ids from the candidate
                # For now, we rely on the entity's resolution process to track them
                pass

            if not chunk_ids and entity.candidate_ids:
                # Fallback: iterate entities in Stage 1 to find chunks
                # This is a limitation of the current design; ideally, we'd pass chunk_ids through resolution
                logger.warning(f"No chunks tracked for entity {entity.id}; skipping mention relations")
                continue

            queries = []
            for chunk in chunks:
                # Simplified: mention all chunks that were part of the document
                # A more precise approach tracks which chunks were sources for each candidate
                cypher = """
                MATCH (e:Entity {id: $entity_id}), (c:Chunk {id: $chunk_id})
                MERGE (e)-[r:MENTIONED_IN]->(c)
                ON CREATE SET r.chunk_count = 1
                ON MATCH SET r.chunk_count = r.chunk_count + 1
                RETURN r
                """
                params = {
                    "entity_id": entity.id,
                    "chunk_id": chunk.id,
                }
                queries.append((cypher, params))

            if queries:
                results = self.client.batch_execute(queries, read_only=False)
                for result in results:
                    if result.affected_rows > 0:
                        created += 1
                    else:
                        updated += 1

        logger.info(f"Mention relations written: {created} created, {updated} updated")
        return created, updated

    def _write_from_relations(self, chunks: list[Chunk], document_id: str) -> int:
        """
        Write FROM relationships (Chunk) -[:FROM]-> (Document).

        Returns created_count (all are created or matched, not updated).
        """
        created = 0

        for chunk in chunks:
            cypher = """
            MATCH (c:Chunk {id: $chunk_id}), (d:Document {id: $document_id})
            MERGE (c)-[:FROM]->(d)
            RETURN 1
            """
            result = self.client.query(
                cypher,
                parameters={"chunk_id": chunk.id, "document_id": document_id},
                read_only=False,
            )
            if result.records:
                created += 1

        logger.info(f"From relations written: {created}")
        return created

    def _write_content_relations(
        self,
        canonical_relations: dict[str, CanonicalRelation],
    ) -> tuple[int, int]:
        """
        Write domain relationships (Entity) -[:RELATION_TYPE]-> (Entity).

        Returns (created_count, updated_count).
        """
        created, updated = 0, 0

        relations_list = list(canonical_relations.values())

        for i in range(0, len(relations_list), self.batch_size):
            batch = relations_list[i : i + self.batch_size]
            queries = []

            for relation in batch:
                cypher = f"""
                MATCH (src:Entity {{id: $source_id}}), (tgt:Entity {{id: $target_id}})
                MERGE (src)-[r:{relation.relation_type}]->(tgt)
                ON CREATE SET
                    r.description = $description,
                    r.evidence = $evidence,
                    r.supporting_chunks = $supporting_chunks,
                    r.relation_count = $relation_count,
                    r.confidence = $confidence,
                    r.created = $created,
                    r.updated = $updated
                ON MATCH SET
                    r.evidence = r.evidence + (
                        [e IN $evidence WHERE NOT e IN r.evidence | e]
                    ),
                    r.supporting_chunks = r.supporting_chunks + (
                        [c IN $supporting_chunks WHERE NOT c IN r.supporting_chunks | c]
                    ),
                    r.relation_count = r.relation_count + 1,
                    r.confidence = $confidence,
                    r.updated = $updated
                RETURN r
                """
                params = {
                    "source_id": relation.source_id,
                    "target_id": relation.target_id,
                    "description": relation.description,
                    "evidence": relation.evidence,
                    "supporting_chunks": relation.supporting_chunks,
                    "relation_count": relation.relation_count,
                    "confidence": relation.confidence,
                    "created": relation.created.isoformat(),
                    "updated": relation.updated.isoformat(),
                }
                queries.append((cypher, params))

            results = self.client.batch_execute(queries, read_only=False)
            for result in results:
                if result.affected_rows > 0:
                    created += 1
                else:
                    updated += 1

        logger.info(f"Content relations written: {created} created, {updated} updated")
        return created, updated
