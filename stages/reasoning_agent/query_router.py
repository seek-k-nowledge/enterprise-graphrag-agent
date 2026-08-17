"""
Query classification and routing for Stage 3.

Routes incoming queries to appropriate retrieval strategies based on linguistic patterns
and semantic understanding.
"""

import logging
import re

from .schemas import RouterDecision

logger = logging.getLogger(__name__)


class QueryRouter:
    """
    Classifies queries and decides which retrieval strategies to use.

    Routes based on:
    - Linguistic patterns (how many, describe, list, etc.)
    - Query structure (subject-verb-object relationships)
    - Named entity presence
    - Question type (factual, descriptive, relational, structural)
    """

    def __init__(self, available_strategies: list[str] = None):
        """
        Initialize the router.

        Args:
            available_strategies: List of available strategies to choose from
                (default: all three)
        """
        self.available_strategies = available_strategies or [
            "graph_traversal",
            "vector_search",
            "cypher_direct",
        ]

    def route(self, query: str) -> RouterDecision:
        """
        Route a query to retrieval strategies.

        Args:
            query: User query text

        Returns:
            RouterDecision with classification and strategy list
        """
        logger.info(f"Routing query: {query}")

        # Classify query type
        query_type = self._classify_query_type(query)
        logger.debug(f"Query type: {query_type}")

        # Extract seed entities
        seed_entities = self._extract_entities(query)
        logger.debug(f"Seed entities: {seed_entities}")

        # Select strategies based on query type
        strategies = self._select_strategies(query_type)
        logger.debug(f"Selected strategies: {strategies}")

        # Determine strategy-specific parameters
        parameters = self._extract_parameters(query, query_type)

        decision = RouterDecision(
            query_classification=query_type,
            strategies=strategies,
            seed_entities=seed_entities,
            parameters=parameters,
            reasoning=f"Query type '{query_type}' routes to {strategies}",
        )

        logger.info(f"Router decision: {decision.query_classification} → {decision.strategies}")
        return decision

    # ─────────────────────────────────────────────────────────────────────────────
    # Private methods: Classification
    # ─────────────────────────────────────────────────────────────────────────────

    def _classify_query_type(self, query: str) -> str:
        """
        Classify query into one of: multi_hop, semantic, aggregate, hybrid.

        Args:
            query: Query text

        Returns:
            Query type string
        """
        query_lower = query.lower()

        # Aggregate/structural: count, top-K, comparisons
        if self._is_aggregate_query(query_lower):
            return "aggregate"

        # Semantic/descriptive: describe, what is, similar to
        if self._is_semantic_query(query_lower):
            return "semantic"

        # Multi-hop/relational: how relates, path, connection
        if self._is_multi_hop_query(query_lower):
            return "multi_hop"

        # Default to hybrid (run all strategies)
        return "hybrid"

    def _is_aggregate_query(self, query_lower: str) -> bool:
        """Check if query is structural/aggregate."""
        aggregate_patterns = [
            r'how many',
            r'count',
            r'number of',
            r'total',
            r'top \d+',
            r'most',
            r'highest',
            r'lowest',
            r'best',
            r'worst',
            r'average',
            r'sum',
            r'which.*has the',
        ]
        return any(re.search(pattern, query_lower) for pattern in aggregate_patterns)

    def _is_semantic_query(self, query_lower: str) -> bool:
        """Check if query is semantic/descriptive."""
        semantic_patterns = [
            r'^describe ',
            r'what is',
            r'what are',
            r'tell me about',
            r'explain',
            r'definition',
            r'information about',
            r'background',
            r'details',
            r'similar to',
            r'related to',
        ]
        return any(re.search(pattern, query_lower) for pattern in semantic_patterns)

    def _is_multi_hop_query(self, query_lower: str) -> bool:
        """Check if query is relational/multi-hop."""
        multi_hop_patterns = [
            r'how does .* relate to',
            r'how is .* connected to',
            r'path from',
            r'relationship between',
            r'does .* work at',
            r'is .* part of',
            r'associated with',
        ]
        return any(re.search(pattern, query_lower) for pattern in multi_hop_patterns)

    # ─────────────────────────────────────────────────────────────────────────────
    # Private methods: Entity extraction
    # ─────────────────────────────────────────────────────────────────────────────

    def _extract_entities(self, query: str) -> list[str]:
        """
        Extract named entity candidates from query.

        Simple heuristics: capitalized words and quoted phrases.

        Returns:
            List of entity name candidates
        """
        entities = []

        # Quoted phrases
        quoted = re.findall(r'"([^"]+)"', query)
        entities.extend(quoted)

        # Capitalized sequences
        capitalized = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', query)
        entities.extend(capitalized)

        # Remove duplicates and common stop words
        stop_words = {"The", "A", "An", "This", "That", "Is", "Are"}
        entities = list(set(e for e in entities if e not in stop_words))

        return entities[:5]  # Limit to top 5

    # ─────────────────────────────────────────────────────────────────────────────
    # Private methods: Strategy selection
    # ─────────────────────────────────────────────────────────────────────────────

    def _select_strategies(self, query_type: str) -> list[str]:
        """
        Select retrieval strategies based on query type.

        Args:
            query_type: Classified query type

        Returns:
            Ordered list of strategies to execute
        """
        if query_type == "aggregate":
            # Structural queries: cypher_direct first, fallback to traversal
            return [s for s in ["cypher_direct", "graph_traversal"] if s in self.available_strategies]

        elif query_type == "semantic":
            # Semantic queries: vector_search first, traversal for context
            return [s for s in ["vector_search", "graph_traversal"] if s in self.available_strategies]

        elif query_type == "multi_hop":
            # Relational queries: graph_traversal first, vector as fallback
            return [s for s in ["graph_traversal", "vector_search"] if s in self.available_strategies]

        else:  # hybrid
            # Default: all strategies in order of versatility
            return [s for s in ["graph_traversal", "vector_search", "cypher_direct"] if s in self.available_strategies]

    # ─────────────────────────────────────────────────────────────────────────────
    # Private methods: Parameter extraction
    # ─────────────────────────────────────────────────────────────────────────────

    def _extract_parameters(self, query: str, query_type: str) -> dict:
        """
        Extract strategy-specific parameters from query.

        Args:
            query: Query text
            query_type: Classified query type

        Returns:
            Dict of parameters keyed by strategy name
        """
        params = {}

        # Extract max_hops for graph_traversal
        hops_match = re.search(r'(\d+)\s*(?:hop|step)', query.lower())
        max_hops = int(hops_match.group(1)) if hops_match else 2
        params["max_hops"] = max(1, min(5, max_hops))

        # Extract top-k for vector_search
        topk_match = re.search(r'top\s+(\d+)', query.lower())
        top_k = int(topk_match.group(1)) if topk_match else 10
        params["top_k"] = max(1, min(100, top_k))

        # Extract limit for cypher_direct
        limit_match = re.search(r'limit\s+(\d+)|(?:top|first)\s+(\d+)', query.lower())
        if limit_match:
            limit = int(limit_match.group(1) or limit_match.group(2))
            params["limit"] = max(1, min(1000, limit))
        else:
            params["limit"] = 10

        return params
