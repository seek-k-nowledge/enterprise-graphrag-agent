"""
Retrieval orchestration agent for multi-hop traversal.

Executes retrieval strategies and aggregates results.
"""

import logging
from typing import Optional

from ..graph_accessor import GraphAccessor
from ..retrieval import (
    GraphTraversalRetriever,
    VectorSearchRetriever,
    CypherDirectRetriever,
)
from ..schemas import RetrievalResult, RetrievalStrategyResult, ReasoningStep

logger = logging.getLogger(__name__)


class RetrievalAgent:
    """
    Orchestrates retrieval strategies and aggregates results.

    Executes selected strategies in sequence or parallel, merges results,
    and ranks by confidence.
    """

    def __init__(
        self,
        graph_accessor: GraphAccessor,
        max_hops: int = 2,
        top_k: int = 20,
        timeout_sec: int = 30,
    ):
        """
        Initialize retrieval agent.

        Args:
            graph_accessor: GraphAccessor instance
            max_hops: Default max hops for traversal
            top_k: Default top-k for vector search (default 20 for broad coverage)
            timeout_sec: Query timeout in seconds
        """
        self.graph_accessor = graph_accessor
        self.max_hops = max_hops
        self.top_k = top_k
        self.timeout_sec = timeout_sec

        # Initialize retrievers
        self.traversal_retriever = GraphTraversalRetriever(
            graph_accessor, max_hops=max_hops, timeout_sec=timeout_sec
        )
        self.vector_retriever = VectorSearchRetriever(
            graph_accessor, top_k=top_k, timeout_sec=timeout_sec
        )
        self.cypher_retriever = CypherDirectRetriever(
            graph_accessor, timeout_sec=timeout_sec
        )

    def retrieve(
        self,
        query: str,
        strategies: list[str],
        parameters: Optional[dict] = None,
    ) -> tuple[RetrievalResult, ReasoningStep]:
        """
        Execute retrieval strategies and aggregate results.

        Args:
            query: User query text
            strategies: List of strategy names to execute
            parameters: Optional strategy-specific parameters

        Returns:
            Tuple of (RetrievalResult, ReasoningStep for audit trail)
        """
        if parameters is None:
            parameters = {}

        logger.info(f"RetrievalAgent executing strategies: {strategies}")

        result = RetrievalResult(query=query)
        step = ReasoningStep(
            step_type="retrieval",
            agent="RetrievalAgent",
            input={"query": query, "strategies": strategies, "parameters": parameters},
        )

        try:
            # Execute each strategy
            strategy_results = []

            if "graph_traversal" in strategies:
                max_hops = parameters.get("max_hops", self.max_hops)
                self.traversal_retriever.max_hops = max_hops
                res = self.traversal_retriever.retrieve(query)
                strategy_results.append(res)

            if "vector_search" in strategies:
                top_k = parameters.get("top_k", self.top_k)
                self.vector_retriever.top_k = top_k
                res = self.vector_retriever.retrieve(query)
                strategy_results.append(res)

            if "cypher_direct" in strategies:
                res = self.cypher_retriever.retrieve(query)
                strategy_results.append(res)

            # Rank by confidence (descending)
            strategy_results.sort(key=lambda r: r.confidence, reverse=True)

            result.strategies_executed = [r.strategy for r in strategy_results]
            result.strategy_results = strategy_results

            # Merge successful results
            for r in strategy_results:
                if not r.error:
                    self._merge_subgraph(result.merged_subgraph, r.subgraph)

            result.total_entities = len(result.merged_subgraph.entities)
            result.total_relations = len(result.merged_subgraph.relations)
            result.total_chunks = len(result.merged_subgraph.chunks)

            step.output = {
                "strategies_executed": result.strategies_executed,
                "total_entities": result.total_entities,
                "total_relations": result.total_relations,
                "total_chunks": result.total_chunks,
                "top_confidence": strategy_results[0].confidence if strategy_results else 0,
            }
            step.reasoning = f"Executed {len(strategy_results)} strategies, "
            step.reasoning += f"merged {len(result.merged_subgraph.entities)} entities"

            logger.info(f"Retrieval complete: {step.output}")

            return result, step

        except Exception as e:
            logger.error(f"Retrieval agent error: {e}")
            step.output = {"error": str(e)}
            raise

    # ─────────────────────────────────────────────────────────────────────────────
    # Private methods
    # ─────────────────────────────────────────────────────────────────────────────

    def _merge_subgraph(self, target, source) -> None:
        """Merge source subgraph into target."""
        target.entities.update(source.entities)
        target.relations.update(source.relations)
        target.chunks.update(source.chunks)
