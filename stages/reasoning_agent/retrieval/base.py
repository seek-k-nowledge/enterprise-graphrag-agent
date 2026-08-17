"""
Abstract base class for retrieval strategies.
"""

from abc import ABC, abstractmethod
from typing import Optional
import logging

from ..graph_accessor import GraphAccessor
from ..schemas import Subgraph, RetrievalStrategyResult

logger = logging.getLogger(__name__)


class BaseRetriever(ABC):
    """
    Abstract base class for retrieval strategies.

    Each strategy implements a different approach to querying the graph:
    - Graph traversal: relational/multi-hop questions
    - Vector search: semantic/descriptive questions
    - Direct Cypher: structural/aggregate questions
    """

    def __init__(self, graph_accessor: GraphAccessor, timeout_sec: int = 30):
        """
        Initialize the retriever.

        Args:
            graph_accessor: GraphAccessor instance (read-only interface to Stage 2 graph)
            timeout_sec: Query timeout in seconds
        """
        self.graph_accessor = graph_accessor
        self.timeout_sec = timeout_sec

    @abstractmethod
    def retrieve(self, query: str) -> RetrievalStrategyResult:
        """
        Execute retrieval and return results.

        Args:
            query: User query text

        Returns:
            RetrievalStrategyResult with subgraph, confidence, explanation, and optional error
        """
        pass

    def _create_result(
        self,
        subgraph: Optional[Subgraph] = None,
        confidence: float = 0.5,
        explanation: str = "",
        error: Optional[str] = None,
    ) -> RetrievalStrategyResult:
        """
        Create a standardized RetrievalStrategyResult.

        Args:
            subgraph: Retrieved subgraph (or empty if error)
            confidence: Confidence score (0-1)
            explanation: Why this strategy succeeded/failed
            error: Error message if retrieval failed

        Returns:
            RetrievalStrategyResult
        """
        if subgraph is None:
            subgraph = Subgraph()

        return RetrievalStrategyResult(
            strategy=self.__class__.__name__,
            subgraph=subgraph,
            confidence=confidence,
            explanation=explanation,
            error=error,
        )
