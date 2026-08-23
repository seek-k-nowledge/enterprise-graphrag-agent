"""
Direct Cypher retrieval strategy: structural and aggregate queries.

Best for questions like "How many X?" or "Top 10 Y by Z?"
"""

import logging
import re

from ..graph_accessor import GraphAccessor
from ..schemas import Subgraph, RetrievalStrategyResult
from .base import BaseRetriever

logger = logging.getLogger(__name__)


class CypherDirectRetriever(BaseRetriever):
    """
    Direct Cypher queries for structural and aggregate questions.

    Strategy:
    1. Detect query intent (count, top-K, list, etc.)
    2. Author Cypher to answer the specific question
    3. Execute and format results
    4. Return result as structured answer

    Note: This is a simplified version. A production system would:
    - Use an LLM agent to author Cypher (safer, more flexible)
    - Validate Cypher before execution (prevent injection)
    - Have a library of common patterns

    For MVP, we implement pattern matching for common cases.
    """

    def __init__(self, graph_accessor: GraphAccessor, timeout_sec: int = 30):
        """
        Initialize direct Cypher retriever.

        Args:
            graph_accessor: GraphAccessor instance
            timeout_sec: Query timeout in seconds
        """
        super().__init__(graph_accessor, timeout_sec)

    def retrieve(self, query: str) -> RetrievalStrategyResult:
        """
        Retrieve via direct Cypher pattern matching.

        Args:
            query: User query text

        Returns:
            RetrievalStrategyResult with structured query results
        """
        logger.info(f"CypherDirectRetriever: {query}")

        try:
            # Detect query intent
            intent = self._detect_intent(query)
            if not intent:
                return self._create_result(
                    error="Query intent not recognized",
                    explanation="Could not determine structural query pattern",
                )

            logger.debug(f"Detected intent: {intent}")

            # Author Cypher for the intent
            cypher, params = self._author_cypher(query, intent)
            if not cypher:
                return self._create_result(
                    error="Cypher authoring failed",
                    explanation="Could not generate Cypher for this query",
                )

            logger.debug(f"Authored Cypher: {cypher[:100]}...")

            # Execute Cypher via Neo4j
            result = self.graph_accessor.client.query(
                cypher, parameters=params, read_only=True
            )

            if not result.records:
                return self._create_result(
                    explanation="Query returned no results",
                    error="Empty result set",
                )

            # Format result as explanation
            explanation = self._format_result(result.records, intent)
            confidence = 0.8  # Structured queries are high confidence

            # Build minimal subgraph from result (for consistency)
            subgraph = Subgraph()

            return self._create_result(
                subgraph=subgraph,
                confidence=confidence,
                explanation=explanation,
            )

        except Exception as e:
            logger.error(f"CypherDirectRetriever error: {e}")
            return self._create_result(
                error=str(e),
                explanation="Cypher query execution failed",
            )

    # ─────────────────────────────────────────────────────────────────────────────
    # Private methods
    # ─────────────────────────────────────────────────────────────────────────────

    def _detect_intent(self, query: str) -> str:
        """
        Detect query intent from linguistic patterns.

        Returns:
            Intent type: "count", "top_k", "list", "has_relation", or None
        """
        query_lower = query.lower()

        # Count queries: "how many", "count", "number of"
        if re.search(r'(how many|count|number of)', query_lower):
            return "count"

        # Top-K queries: "top 10", "most X", "best"
        if re.search(r'(top \d+|most|best|highest|lowest)', query_lower):
            return "top_k"

        # List queries: "list", "all", "show"
        if re.search(r'(list|all|show|which|get)', query_lower):
            return "list"

        # Relation queries: "does X work at Y", "related to"
        if re.search(r'(does|is|related|connected|associated)', query_lower):
            return "has_relation"

        return None

    def _author_cypher(self, query: str, intent: str) -> tuple[str, dict]:
        """
        Author Cypher query based on intent and query text.

        For MVP, this is pattern-based. Production would use an LLM agent.

        Returns:
            Tuple of (cypher_string, parameters_dict)
        """
        if intent == "count":
            # Example: "How many people work at Acme?"
            # Extract entity name (simple heuristic)
            match = re.search(r'(?:at|for|in|with)\s+([A-Z][a-zA-Z\s]+?)(?:\?|$)', query)
            if not match:
                return None, None

            entity_name = match.group(1).strip()
            cypher = """
            MATCH (org:Entity {entity_type: "Organization", canonical_name: $entity_name})
            MATCH (person:Entity {entity_type: "Person"})-[:WORKS_AT]->(org)
            RETURN count(distinct person) as count
            """
            return cypher, {"entity_name": entity_name}

        elif intent == "top_k":
            # Example: "Top 10 companies by employee count"
            match = re.search(r'top\s+(\d+)', query.lower())
            k = int(match.group(1)) if match else 10
            k = min(k, 100)  # Cap at 100

            # Simplified: just list top entities
            cypher = """
            MATCH (e:Entity)
            RETURN e.canonical_name as name, e.entity_type as type,
                   size((e)-[]->()) as relation_count
            ORDER BY relation_count DESC
            LIMIT $limit
            """
            return cypher, {"limit": k}

        elif intent == "list":
            # Example: "List all organizations"
            # Extract entity type from query
            types = ["Person", "Organization", "Location", "Event"]
            entity_type = None
            for t in types:
                if t.lower() in query.lower():
                    entity_type = t
                    break

            if not entity_type:
                entity_type = "Entity"  # Default

            cypher = """
            MATCH (e:Entity {entity_type: $entity_type})
            RETURN e.canonical_name as name
            ORDER BY name ASC
            LIMIT 50
            """
            return cypher, {"entity_type": entity_type}

        elif intent == "has_relation":
            # Example: "Does John work at Acme?"
            # Extract two entity names
            match = re.search(r'([A-Z][a-zA-Z\s]+?)\s+(?:works at|at|for)\s+([A-Z][a-zA-Z\s]+?)(?:\?|$)', query)
            if not match:
                return None, None

            entity_a = match.group(1).strip()
            entity_b = match.group(2).strip()

            cypher = """
            MATCH (a:Entity {canonical_name: $entity_a})
            MATCH (b:Entity {canonical_name: $entity_b})
            MATCH path = (a)-[r]-(b)
            RETURN type(r) as relation_type, r.confidence as confidence
            """
            return cypher, {"entity_a": entity_a, "entity_b": entity_b}

        return None, None

    def _format_result(self, records: list, intent: str) -> str:
        """
        Format Cypher result as human-readable explanation.

        Args:
            records: List of Neo4j record dicts
            intent: Query intent

        Returns:
            Formatted explanation string
        """
        if not records:
            return "Query returned no results."

        if intent == "count":
            count = records[0].get("count", 0)
            return f"Found {count} results."

        elif intent == "top_k":
            lines = []
            for i, record in enumerate(records[:10], 1):
                name = record.get("name", "Unknown")
                type_name = record.get("type", "Entity")
                lines.append(f"{i}. {name} ({type_name})")
            return "Top results:\n" + "\n".join(lines)

        elif intent == "list":
            names = [r.get("name", "Unknown") for r in records[:20]]
            return f"Found: {', '.join(names)}"

        elif intent == "has_relation":
            if records:
                rel_type = records[0].get("relation_type", "Unknown")
                confidence = records[0].get("confidence", 0.5)
                try:
                    confidence_float = float(confidence)
                    return f"Related by: {rel_type} (confidence: {confidence_float:.2f})"
                except (ValueError, TypeError):
                    return f"Related by: {rel_type} (confidence: {confidence})"
            return "No relation found."

        return f"Results: {len(records)} records"
