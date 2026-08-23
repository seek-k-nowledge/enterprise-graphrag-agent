"""
Read-only graph accessor for Stage 3: interface to Stage 2's Neo4j graph.

Encapsulates all Neo4j queries, ensures read-only access, and provides
convenient methods for retrieval strategies. Handles caching, timeouts, and errors.

Usage:
    accessor = GraphAccessor(neo4j_client, cache_ttl_sec=3600)
    entity = accessor.get_entity_by_id("person:john_doe")
    relations = accessor.get_entity_relations("person:john_doe")
    subgraph = accessor.traverse_multi_hop("person:john_doe", max_hops=2)
"""

import hashlib
import logging
from typing import Optional
from datetime import datetime, timedelta

from stages.graph_indexing.neo4j_client import Neo4jClient
from .schemas import GraphEntity, GraphRelation, GraphChunk, Subgraph

logger = logging.getLogger(__name__)


class GraphAccessor:
    """
    Read-only accessor to the Stage 2 Neo4j graph.

    All methods are read-only. Caching is optional and uses TTL.
    """

    def __init__(
        self,
        neo4j_client: Neo4jClient,
        enable_caching: bool = True,
        cache_ttl_sec: int = 3600,
        query_timeout_sec: int = 30,
    ):
        """
        Initialize the graph accessor.

        Args:
            neo4j_client: Connected Neo4jClient instance
            enable_caching: Enable query result caching
            cache_ttl_sec: Cache time-to-live in seconds
            query_timeout_sec: Query timeout in seconds
        """
        self.client = neo4j_client
        self.enable_caching = enable_caching
        self.cache_ttl_sec = cache_ttl_sec
        self.query_timeout_sec = query_timeout_sec
        self.cache = {} if enable_caching else None

    def get_entity_by_id(self, entity_id: str) -> Optional[GraphEntity]:
        """
        Get an entity by its canonical ID.

        Args:
            entity_id: Entity ID (e.g., "person:john_doe")

        Returns:
            GraphEntity or None if not found
        """
        cache_key = f"entity_id:{entity_id}"
        if self.enable_caching and self._check_cache(cache_key):
            return self.cache[cache_key]["data"]

        cypher = """
        MATCH (e:Entity {id: $entity_id})
        RETURN e
        """

        try:
            result = self.client.query(
                cypher,
                parameters={"entity_id": entity_id},
                read_only=True,
            )

            if not result.records:
                return None

            entity_data = result.records[0].get("e")
            entity = self._entity_from_neo4j(entity_data)

            if self.enable_caching:
                self._cache_result(cache_key, entity)

            return entity
        except Exception as e:
            logger.error(f"Error fetching entity {entity_id}: {e}")
            return None

    def get_entity_by_name(
        self,
        name: str,
        entity_type: Optional[str] = None,
    ) -> Optional[GraphEntity]:
        """
        Get an entity by its canonical name (optionally filtered by type).

        Args:
            name: Canonical name to search for
            entity_type: Optional entity type filter

        Returns:
            GraphEntity or None if not found (returns first match if multiple)
        """
        cache_key = f"entity_name:{name}:{entity_type or '*'}"
        if self.enable_caching and self._check_cache(cache_key):
            return self.cache[cache_key]["data"]

        if entity_type:
            cypher = """
            MATCH (e:Entity {canonical_name: $name, entity_type: $type})
            RETURN e LIMIT 1
            """
            params = {"name": name, "type": entity_type}
        else:
            cypher = """
            MATCH (e:Entity {canonical_name: $name})
            RETURN e LIMIT 1
            """
            params = {"name": name}

        try:
            result = self.client.query(cypher, parameters=params, read_only=True)

            if not result.records:
                return None

            entity_data = result.records[0].get("e")
            entity = self._entity_from_neo4j(entity_data)

            if self.enable_caching:
                self._cache_result(cache_key, entity)

            return entity
        except Exception as e:
            logger.error(f"Error fetching entity by name {name}: {e}")
            return None

    def search_entities(
        self,
        query_text: str,
        entity_type: Optional[str] = None,
        limit: int = 10,
    ) -> list[GraphEntity]:
        """
        Search for entities by text (canonical name or surface forms).

        Uses CONTAINS for simple substring matching. For semantic search, use
        search_entities_by_embedding().

        Args:
            query_text: Text to search for
            entity_type: Optional filter by entity type
            limit: Maximum results to return

        Returns:
            List of GraphEntity objects
        """
        cache_key = f"search_entities:{query_text}:{entity_type or '*'}:{limit}"
        if self.enable_caching and self._check_cache(cache_key):
            return self.cache[cache_key]["data"]

        query_lower = query_text.lower()

        if entity_type:
            cypher = """
            MATCH (e:Entity {entity_type: $type})
            WHERE toLower(e.canonical_name) CONTAINS $query
               OR any(sf IN e.surface_forms WHERE toLower(sf) CONTAINS $query)
            RETURN e
            LIMIT $limit
            """
            params = {"query": query_lower, "type": entity_type, "limit": limit}
        else:
            cypher = """
            MATCH (e:Entity)
            WHERE toLower(e.canonical_name) CONTAINS $query
               OR any(sf IN e.surface_forms WHERE toLower(sf) CONTAINS $query)
            RETURN e
            LIMIT $limit
            """
            params = {"query": query_lower, "limit": limit}

        try:
            result = self.client.query(cypher, parameters=params, read_only=True)
            entities = [
                self._entity_from_neo4j(r.get("e")) for r in result.records
            ]

            if self.enable_caching:
                self._cache_result(cache_key, entities)

            return entities
        except Exception as e:
            logger.error(f"Error searching entities with '{query_text}': {e}")
            return []

    def get_entity_relations(
        self,
        entity_id: str,
        relation_types: Optional[list[str]] = None,
        limit: int = 100,
    ) -> list[GraphRelation]:
        """
        Get all relations involving an entity (both incoming and outgoing).

        Args:
            entity_id: Entity ID
            relation_types: Optional filter by relation type (e.g., ["WORKS_AT"])
            limit: Maximum relations to return

        Returns:
            List of GraphRelation objects
        """
        cache_key = f"entity_relations:{entity_id}:{':'.join(relation_types or [])}:{limit}"
        if self.enable_caching and self._check_cache(cache_key):
            return self.cache[cache_key]["data"]

        if relation_types:
            type_str = "|".join(relation_types)
            cypher = f"""
            MATCH (e:Entity {{id: $entity_id}})-[r:{type_str}]-(target)
            RETURN r, target.id as target_id, target.entity_type as target_type
            LIMIT $limit
            """
        else:
            cypher = """
            MATCH (e:Entity {id: $entity_id})-[r]-(target)
            RETURN r, target.id as target_id, target.entity_type as target_type
            LIMIT $limit
            """

        try:
            result = self.client.query(
                cypher,
                parameters={"entity_id": entity_id, "limit": limit},
                read_only=True,
            )

            relations = [
                self._relation_from_neo4j(r.get("r"), r.get("target_id"))
                for r in result.records
            ]

            if self.enable_caching:
                self._cache_result(cache_key, relations)

            return relations
        except Exception as e:
            logger.error(f"Error fetching relations for {entity_id}: {e}")
            return []

    def traverse_multi_hop(
        self,
        start_entity_id: str,
        max_hops: int = 2,
    ) -> Subgraph:
        """
        Traverse the graph from a starting entity up to max_hops away.

        Returns a subgraph containing all reachable entities, relations, and
        their supporting chunks.

        Args:
            start_entity_id: Starting entity ID
            max_hops: Maximum hops to traverse (1-5)

        Returns:
            Subgraph object with entities, relations, and chunks
        """
        max_hops = max(1, min(5, max_hops))  # Clamp to [1, 5]
        cache_key = f"traverse_multi_hop:{start_entity_id}:{max_hops}"
        if self.enable_caching and self._check_cache(cache_key):
            return self.cache[cache_key]["data"]

        subgraph = Subgraph()

        try:
            # Get starting entity
            start_entity = self.get_entity_by_id(start_entity_id)
            if not start_entity:
                logger.warning(f"Start entity not found: {start_entity_id}")
                return subgraph

            subgraph.entities[start_entity_id] = start_entity

            # Traverse with APOC or raw Cypher
            cypher = f"""
            MATCH (start:Entity {{id: $start_id}})
            CALL apoc.path.expandConfig(start, {{relationshipFilter: "", maxLevel: {max_hops}, uniqueness: 'NODE_GLOBAL'}})
            YIELD path
            WITH nodes(path) as entities, relationships(path) as rels
            UNWIND entities as e
            UNWIND rels as r
            WITH DISTINCT e, r
            MATCH (e)-[r]->(target)
            OPTIONAL MATCH (e)-[:MENTIONED_IN]->(c:Chunk)
            RETURN DISTINCT e, r, target, c
            LIMIT 1000
            """

            result = self.client.query(
                cypher,
                parameters={"start_id": start_entity_id},
                read_only=True,
            )

            # Collect entities, relations, chunks from results
            for record in result.records:
                entity_data = record.get("e")
                if entity_data:
                    entity = self._entity_from_neo4j(entity_data)
                    subgraph.entities[entity.id] = entity

                target_data = record.get("target")
                if target_data:
                    target = self._entity_from_neo4j(target_data)
                    subgraph.entities[target.id] = target

                rel_data = record.get("r")
                if rel_data:
                    rel = self._relation_from_neo4j(
                        rel_data,
                        record.get("target", {}).get("id"),
                    )
                    rel_key = f"{rel.source_id}:{rel.target_id}:{rel.relation_type}"
                    if rel_key not in subgraph.relations:
                        subgraph.relations[rel_key] = rel

                chunk_data = record.get("c")
                if chunk_data:
                    chunk = self._chunk_from_neo4j(chunk_data)
                    subgraph.chunks[chunk.id] = chunk

            logger.info(
                f"Traversal complete: {len(subgraph.entities)} entities, "
                f"{len(subgraph.relations)} relations, {len(subgraph.chunks)} chunks"
            )

            if self.enable_caching:
                self._cache_result(cache_key, subgraph)

            return subgraph

        except Exception as e:
            logger.error(f"Error traversing from {start_entity_id}: {e}")
            return subgraph

    def search_chunks_by_embedding(
        self,
        query_embedding: list[float],
        top_k: int = 10,
    ) -> list[GraphChunk]:
        """
        Search chunks by vector embedding similarity.

        Args:
            query_embedding: Query vector (must match dimension)
            top_k: Number of results to return

        Returns:
            List of GraphChunk objects, ranked by similarity
        """
        cache_key = f"chunk_search:{hashlib.md5(str(query_embedding[:100]).encode()).hexdigest()}:{top_k}"
        if self.enable_caching and self._check_cache(cache_key):
            return self.cache[cache_key]["data"]

        cypher = """
        MATCH (c:Chunk)
        WHERE c.embedding IS NOT NULL
        WITH c, vector.similarity.cosine(c.embedding, $embedding) as similarity
        RETURN c, similarity
        ORDER BY similarity DESC
        LIMIT $limit
        """

        try:
            result = self.client.query(
                cypher,
                parameters={"embedding": query_embedding, "limit": top_k},
                read_only=True,
            )

            chunks = []
            for r in result.records:
                chunk = self._chunk_from_neo4j(r.get("c"))
                similarity = r.get("similarity")
                if similarity is not None:
                    chunk.similarity_score = float(similarity)
                chunks.append(chunk)

            if self.enable_caching:
                self._cache_result(cache_key, chunks)

            return chunks
        except Exception as e:
            logger.error(f"Error searching chunks by embedding: {e}")
            return []

    def get_chunks_for_entity(
        self,
        entity_id: str,
    ) -> list[GraphChunk]:
        """
        Get all chunks mentioning a specific entity.

        Args:
            entity_id: Entity ID

        Returns:
            List of GraphChunk objects
        """
        cache_key = f"entity_chunks:{entity_id}"
        if self.enable_caching and self._check_cache(cache_key):
            return self.cache[cache_key]["data"]

        cypher = """
        MATCH (e:Entity {id: $entity_id})-[:MENTIONED_IN]->(c:Chunk)
        RETURN c
        """

        try:
            result = self.client.query(
                cypher,
                parameters={"entity_id": entity_id},
                read_only=True,
            )

            chunks = [
                self._chunk_from_neo4j(r.get("c")) for r in result.records
            ]

            if self.enable_caching:
                self._cache_result(cache_key, chunks)

            return chunks
        except Exception as e:
            logger.error(f"Error fetching chunks for entity {entity_id}: {e}")
            return []

    def get_document_chunks(self, document_id: str) -> list[GraphChunk]:
        """
        Get all chunks from a specific document.

        Args:
            document_id: Document ID

        Returns:
            List of GraphChunk objects
        """
        cache_key = f"doc_chunks:{document_id}"
        if self.enable_caching and self._check_cache(cache_key):
            return self.cache[cache_key]["data"]

        cypher = """
        MATCH (c:Chunk {document_id: $document_id})
        RETURN c
        ORDER BY c.start_char ASC
        """

        try:
            result = self.client.query(
                cypher,
                parameters={"document_id": document_id},
                read_only=True,
            )

            chunks = [
                self._chunk_from_neo4j(r.get("c")) for r in result.records
            ]

            if self.enable_caching:
                self._cache_result(cache_key, chunks)

            return chunks
        except Exception as e:
            logger.error(f"Error fetching chunks for document {document_id}: {e}")
            return []

    # ─────────────────────────────────────────────────────────────────────────────
    # Private methods: Neo4j → Schema conversion
    # ─────────────────────────────────────────────────────────────────────────────

    def _entity_from_neo4j(self, neo4j_node: dict) -> GraphEntity:
        """Convert a Neo4j Entity node to GraphEntity."""
        return GraphEntity(
            id=neo4j_node.get("id", ""),
            entity_type=neo4j_node.get("entity_type", ""),
            canonical_name=neo4j_node.get("canonical_name", ""),
            surface_forms=neo4j_node.get("surface_forms", []),
            description=neo4j_node.get("description", ""),
            sources=neo4j_node.get("sources", []),
        )

    def _relation_from_neo4j(
        self,
        neo4j_rel: dict,
        target_id: Optional[str] = None,
    ) -> GraphRelation:
        """Convert a Neo4j Relation to GraphRelation."""
        # Neo4j relations require manual source extraction
        source_id = neo4j_rel.get("source_id", "")
        if not source_id and hasattr(neo4j_rel, "start_node"):
            source_id = neo4j_rel.start_node.get("id", "")

        if not target_id:
            target_id = neo4j_rel.get("target_id", "")
            if not target_id and hasattr(neo4j_rel, "end_node"):
                target_id = neo4j_rel.end_node.get("id", "")

        return GraphRelation(
            source_id=source_id,
            target_id=target_id or "",
            relation_type=neo4j_rel.get("type", ""),
            description=neo4j_rel.get("description", ""),
            evidence=neo4j_rel.get("evidence", []),
            supporting_chunks=neo4j_rel.get("supporting_chunks", []),
            confidence=neo4j_rel.get("confidence", 0.5),
        )

    def _chunk_from_neo4j(self, neo4j_node: dict) -> GraphChunk:
        """Convert a Neo4j Chunk node to GraphChunk."""
        return GraphChunk(
            id=neo4j_node.get("id", ""),
            text=neo4j_node.get("text", ""),
            start_char=neo4j_node.get("start_char", 0),
            end_char=neo4j_node.get("end_char", 0),
            document_id=neo4j_node.get("document_id", ""),
            embedding=neo4j_node.get("embedding", None),
        )

    # ─────────────────────────────────────────────────────────────────────────────
    # Caching utilities
    # ─────────────────────────────────────────────────────────────────────────────

    def _check_cache(self, key: str) -> bool:
        """Check if a key exists in cache and is not expired."""
        if not self.cache or key not in self.cache:
            return False

        entry = self.cache[key]
        if datetime.utcnow() > entry["expires"]:
            del self.cache[key]
            return False

        return True

    def _cache_result(self, key: str, data: any) -> None:
        """Store a result in cache with TTL."""
        if not self.cache:
            return

        self.cache[key] = {
            "data": data,
            "expires": datetime.utcnow() + timedelta(seconds=self.cache_ttl_sec),
        }

    def clear_cache(self) -> None:
        """Clear the entire cache."""
        if self.cache:
            self.cache.clear()
            logger.info("Query cache cleared")
