"""
Retrieval strategies for Stage 3.

Strategies:
- graph_traversal: Multi-hop Cypher-based traversal for relational questions
- vector_search: Embedding-based semantic search for descriptive questions
- cypher_direct: Direct Cypher for structural/aggregate queries
"""

from .base import BaseRetriever
from .graph_traversal import GraphTraversalRetriever
from .vector_search import VectorSearchRetriever
from .cypher_direct import CypherDirectRetriever

__all__ = [
    "BaseRetriever",
    "GraphTraversalRetriever",
    "VectorSearchRetriever",
    "CypherDirectRetriever",
]
