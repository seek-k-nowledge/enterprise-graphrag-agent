"""
Graph traversal retrieval strategy: multi-hop Cypher-based traversal.

Best for relational questions like "How does X relate to Y?" or "Path from A to B?"
"""

import logging
import re
from typing import Optional

from ..graph_accessor import GraphAccessor
from ..schemas import Subgraph, RetrievalStrategyResult
from .base import BaseRetriever

logger = logging.getLogger(__name__)


class GraphTraversalRetriever(BaseRetriever):
    """
    Multi-hop graph traversal for relational questions.

    Strategy:
    1. Extract named entities from query (simple NER)
    2. Find corresponding nodes in the graph
    3. Traverse paths between them (1-3 hops)
    4. Return subgraph of traversed nodes, edges, and chunks
    """

    def __init__(
        self,
        graph_accessor: GraphAccessor,
        max_hops: int = 2,
        timeout_sec: int = 30,
    ):
        """
        Initialize graph traversal retriever.

        Args:
            graph_accessor: GraphAccessor instance
            max_hops: Maximum hops to traverse (1-5, clamped)
            timeout_sec: Query timeout in seconds
        """
        super().__init__(graph_accessor, timeout_sec)
        self.max_hops = max(1, min(5, max_hops))

    def retrieve(self, query: str) -> RetrievalStrategyResult:
        """
        Retrieve by traversing multi-hop paths in the graph.

        Args:
            query: User query text

        Returns:
            RetrievalStrategyResult with traversed subgraph
        """
        logger.info(f"GraphTraversalRetriever: {query}")

        try:
            # Step 1: Extract entity candidates from query
            entities = self._extract_entities_from_query(query)
            if not entities:
                return self._create_result(
                    explanation="No entities found in query",
                    error="Entity extraction failed",
                )

            logger.debug(f"Extracted entity candidates: {entities}")

            # Step 2: Resolve candidates to actual nodes
            resolved_entities = self._resolve_entities(entities)
            if not resolved_entities:
                return self._create_result(
                    explanation=f"Could not resolve any of {len(entities)} entity candidates to graph nodes",
                    error="Entity resolution failed",
                )

            logger.debug(f"Resolved to {len(resolved_entities)} entities")

            # Step 3: Traverse from each resolved entity
            subgraph = Subgraph()
            for entity in resolved_entities:
                traversed = self.graph_accessor.traverse_multi_hop(
                    entity.id, self.max_hops
                )
                self._merge_subgraph(subgraph, traversed)

            if not subgraph.entities:
                return self._create_result(
                    explanation="Traversal found no connected entities",
                    error="Traversal returned empty subgraph",
                )

            confidence = self._compute_confidence(subgraph)

            return self._create_result(
                subgraph=subgraph,
                confidence=confidence,
                explanation=f"Traversed {len(subgraph.entities)} entities, "
                f"{len(subgraph.relations)} relations from {len(resolved_entities)} starting points",
            )

        except Exception as e:
            logger.error(f"GraphTraversalRetriever error: {e}")
            return self._create_result(
                error=str(e),
                explanation="Graph traversal failed",
            )

    # ─────────────────────────────────────────────────────────────────────────────
    # Private methods
    # ─────────────────────────────────────────────────────────────────────────────

    def _extract_entities_from_query(self, query: str) -> list[str]:
        """
        Simple entity extraction: look for capitalized words and quoted phrases.

        A proper implementation would use NER. For now, extract:
        - Capitalized sequences (e.g., "John Doe")
        - Quoted phrases (e.g., "Acme Corp")

        Returns:
            List of entity name candidates
        """
        entities = []

        # Extract quoted phrases
        quoted = re.findall(r'"([^"]+)"', query)
        entities.extend(quoted)

        # Extract capitalized sequences
        # Pattern: capital letter followed by optional lowercase, then another capital
        # This is a simple heuristic
        capitalized = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', query)
        entities.extend(capitalized)

        # Remove duplicates and common stop words
        stop_words = {"The", "A", "An", "This", "That", "Is", "Are", "Was", "Were"}
        entities = list(set(e for e in entities if e not in stop_words))

        return entities[:5]  # Limit to top 5 to avoid explosion

    def _resolve_entities(self, candidates: list[str]):
        """
        Resolve entity name candidates to actual graph nodes.

        Tries exact match first, then fuzzy search.

        Returns:
            List of resolved GraphEntity objects
        """
        resolved = []

        for candidate in candidates:
            # Try exact match
            entity = self.graph_accessor.get_entity_by_name(candidate)
            if entity:
                resolved.append(entity)
                continue

            # Try search with substring
            results = self.graph_accessor.search_entities(candidate, limit=1)
            if results:
                resolved.append(results[0])

        return resolved

    def _merge_subgraph(self, target: Subgraph, source: Subgraph) -> None:
        """Merge source subgraph into target."""
        target.entities.update(source.entities)
        target.relations.update(source.relations)
        target.chunks.update(source.chunks)

    def _compute_confidence(self, subgraph: Subgraph) -> float:
        """
        Compute confidence based on subgraph size and connectivity.

        Heuristic: larger and more connected graphs have higher confidence.
        Range: [0.3, 0.95]
        """
        if not subgraph.entities or not subgraph.relations:
            return 0.3

        entity_count = len(subgraph.entities)
        relation_count = len(subgraph.relations)
        chunk_count = len(subgraph.chunks)

        # More entities and relations = higher confidence
        # Capped at 0.95 (never 100% sure without verification)
        connectivity = relation_count / max(entity_count, 1)
        coverage = chunk_count / max(entity_count, 1)

        confidence = min(0.95, 0.3 + (connectivity * 0.3) + (coverage * 0.35))
        return confidence
