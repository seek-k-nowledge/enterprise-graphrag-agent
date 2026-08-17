"""
Stage 2: Graph Indexing & Neo4j Integration

Public API for entity resolution, graph writing, and chunk embedding.

Entry point:
    result = process_extraction(
        extraction_result,
        neo4j_client,
        config=GraphIndexingConfig()
    )
    # result: GraphWriteResult with full metadata
"""

from datetime import datetime
import logging

from stages.extraction.schemas import ExtractionResult
from .neo4j_client import Neo4jClient
from .entity_resolver import EntityResolver
from .graph_writer import GraphWriter
from .embedder import ChunkEmbedder
from .schemas import (
    GraphWriteResult,
    GraphIndexingConfig,
    CanonicalNode,
    CanonicalRelation,
    ResolutionMetadata,
)

logger = logging.getLogger(__name__)

# Public API
__all__ = [
    "process_extraction",
    "create_client",
    "GraphWriteResult",
    "GraphIndexingConfig",
    "CanonicalNode",
    "CanonicalRelation",
    "Neo4jClient",
    "EntityResolver",
    "GraphWriter",
    "ChunkEmbedder",
]


def create_client(
    uri: str = "bolt://localhost:7687",
    user: str = "neo4j",
    password: str = "graphrag_dev_password",
    timeout: int = 30,
    max_retries: int = 3,
) -> Neo4jClient:
    """
    Factory function to create and connect a Neo4jClient.

    Args:
        uri: Bolt connection string
        user: Username
        password: Password
        timeout: Query timeout in seconds
        max_retries: Retry count on transient errors

    Returns:
        Connected Neo4jClient

    Raises:
        ServiceUnavailable if connection fails
    """
    client = Neo4jClient(
        uri=uri,
        user=user,
        password=password,
        timeout=timeout,
        max_retries=max_retries,
    )
    client.connect()
    return client


def process_extraction(
    extraction_result: ExtractionResult,
    neo4j_client: Neo4jClient,
    config: GraphIndexingConfig = None,
) -> GraphWriteResult:
    """
    Process an extraction result: resolve entities, write to graph, embed chunks.

    This is the main entry point for Stage 2. It orchestrates the full pipeline:
    1. Entity resolution (candidate → canonical)
    2. Relation resolution (candidate edges → canonical edges)
    3. Neo4j writes (nodes and relationships, idempotent)
    4. Chunk embedding (vectorization and storage)

    Args:
        extraction_result: ExtractionResult from Stage 1
        neo4j_client: Connected Neo4jClient instance
        config: GraphIndexingConfig (uses defaults if None)

    Returns:
        GraphWriteResult with full metadata, node/relation counts, and errors

    Raises:
        Exception on critical errors (Neo4j failure, embedding API failure)
    """
    if config is None:
        config = GraphIndexingConfig()

    logger.info(
        f"Processing extraction: document={extraction_result.metadata.document_id}, "
        f"chunks={len(extraction_result.chunks)}, "
        f"candidates={len(extraction_result.entities)}"
    )

    result = GraphWriteResult(
        document_id=extraction_result.metadata.document_id,
        document_uri=extraction_result.metadata.uri,
        extraction_model=extraction_result.metadata.extraction_model,
        schema_version=extraction_result.metadata.schema_version,
    )

    try:
        # ─────────────────────────────────────────────────────────────────────────
        # Step 1: Entity Resolution
        # ─────────────────────────────────────────────────────────────────────────
        logger.info("Step 1: Resolving entities...")
        resolver = EntityResolver(
            fuzzy_threshold=config.resolution.fuzzy_threshold,
            rules=config.resolution.rules,
            auto_merge_fuzzy=config.resolution.auto_merge_fuzzy,
        )

        resolution_result = resolver.resolve(
            extraction_result.entities,
            extraction_result.metadata.document_id,
        )

        canonical_entities = resolution_result.canonical_entities
        result.canonical_entities = canonical_entities
        result.resolution_metadata = resolution_result.metadata

        logger.info(
            f"Entity resolution complete: {len(extraction_result.entities)} candidates → "
            f"{len(canonical_entities)} canonical nodes"
        )

        # ─────────────────────────────────────────────────────────────────────────
        # Step 2: Relation Resolution
        # ─────────────────────────────────────────────────────────────────────────
        logger.info("Step 2: Resolving relations...")
        canonical_relations = resolver.resolve_relations(
            extraction_result.relations,
            resolution_result.candidate_to_canonical,
        )
        result.canonical_relations = canonical_relations

        logger.info(
            f"Relation resolution complete: {len(extraction_result.relations)} candidates → "
            f"{len(canonical_relations)} canonical relations"
        )

        # ─────────────────────────────────────────────────────────────────────────
        # Step 3: Neo4j Schema Setup (idempotent, once per session typically)
        # ─────────────────────────────────────────────────────────────────────────
        logger.info("Step 3: Ensuring Neo4j schema...")
        writer = GraphWriter(
            neo4j_client,
            batch_size=config.neo4j.batch_size,
        )
        # Schema setup is idempotent; safe to call every time
        try:
            writer.setup_schema()
        except Exception as e:
            logger.warning(f"Schema setup skipped (may already exist): {e}")

        # ─────────────────────────────────────────────────────────────────────────
        # Step 4: Write to Neo4j
        # ─────────────────────────────────────────────────────────────────────────
        logger.info("Step 4: Writing to Neo4j...")
        node_result, rel_result = writer.write_extraction(
            extraction_result,
            canonical_entities,
            canonical_relations,
        )

        result.nodes = node_result
        result.relations = rel_result

        logger.info(
            f"Neo4j write complete: {node_result.total_nodes_written} nodes, "
            f"{rel_result.total_relations_written} relations"
        )

        # ─────────────────────────────────────────────────────────────────────────
        # Step 5: Chunk Embedding
        # ─────────────────────────────────────────────────────────────────────────
        logger.info("Step 5: Embedding chunks...")
        embedder = ChunkEmbedder(
            neo4j_client=neo4j_client,
            model=config.embedding.model,
            provider=config.embedding.provider,
            batch_size=config.embedding.batch_size,
            dimensions=config.embedding.dimensions,
        )

        # Ensure vector index exists (idempotent)
        try:
            embedder.ensure_vector_index()
        except Exception as e:
            logger.warning(f"Vector index creation skipped: {e}")

        # Embed chunks
        embed_result = embedder.embed_chunks(
            extraction_result.chunks,
            extraction_result.metadata.document_id,
            skip_existing=True,
        )

        result.chunks_embedded = embed_result.chunks_embedded
        result.vectors_indexed = embed_result.vectors_indexed
        if embed_result.errors:
            result.errors.extend(embed_result.errors)

        logger.info(
            f"Embedding complete: {embed_result.chunks_embedded} chunks embedded, "
            f"{embed_result.vectors_indexed} vectors indexed"
        )

        # ─────────────────────────────────────────────────────────────────────────
        # Step 6: Optional APOC Enrichment
        # ─────────────────────────────────────────────────────────────────────────
        if config.apoc_enabled:
            logger.info("Step 6: Running APOC enrichment...")
            try:
                _run_apoc_enrichment(neo4j_client, config.apoc_algorithms)
                logger.info("APOC enrichment complete")
            except Exception as e:
                logger.warning(f"APOC enrichment skipped: {e}")
                result.errors.append({
                    "type": "apoc_enrichment_error",
                    "error": str(e),
                })

        # ─────────────────────────────────────────────────────────────────────────
        # Finalize result
        # ─────────────────────────────────────────────────────────────────────────
        result.ingested_at = datetime.utcnow()

        logger.info(
            f"Stage 2 complete: document={result.document_id}, "
            f"total_nodes={result.total_nodes_written}, "
            f"total_relations={result.total_relations_written}, "
            f"chunks_embedded={result.chunks_embedded}"
        )

        return result

    except Exception as e:
        logger.error(f"Fatal error processing extraction: {e}")
        result.errors.append({
            "type": "fatal_error",
            "error": str(e),
            "stage": "orchestration",
        })
        raise


def _run_apoc_enrichment(client: Neo4jClient, algorithms: list[str]) -> None:
    """
    Run optional APOC enrichment algorithms.

    Args:
        client: Neo4jClient instance
        algorithms: List of algorithm names (e.g., ["lpa", "pagerank"])

    Raises:
        Exception if APOC is unavailable or algorithms fail
    """
    if "lpa" in algorithms:
        logger.info("Running APOC community detection (LPA)...")
        cypher = """
        CALL apoc.algo.community.lpa()
        YIELD nodeId, community
        MATCH (n) WHERE id(n) = nodeId
        SET n.community = community
        RETURN count(n) as nodes_processed
        """
        result = client.query(cypher, read_only=False)
        if result.records:
            logger.info(f"LPA complete: {result.records[0].get('nodes_processed', 0)} nodes processed")

    if "pagerank" in algorithms:
        logger.info("Running APOC PageRank...")
        cypher = """
        CALL apoc.algo.pageRank()
        YIELD nodeId, score
        MATCH (n) WHERE id(n) = nodeId
        SET n.pagerank = score
        RETURN count(n) as nodes_processed
        """
        result = client.query(cypher, read_only=False)
        if result.records:
            logger.info(f"PageRank complete: {result.records[0].get('nodes_processed', 0)} nodes processed")
