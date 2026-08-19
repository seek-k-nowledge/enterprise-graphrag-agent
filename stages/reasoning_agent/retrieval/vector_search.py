"""
Vector search retrieval strategy: embedding-based semantic search.

Best for descriptive questions like "Describe X" or "What is known about Y?"
"""

import logging
import os

from ..graph_accessor import GraphAccessor
from ..schemas import Subgraph, RetrievalStrategyResult
from .base import BaseRetriever

logger = logging.getLogger(__name__)


class VectorSearchRetriever(BaseRetriever):
    """
    Semantic search via embedding similarity.

    Strategy:
    1. Embed the query using the same embedding model as chunks
    2. Search vector index for top-K similar chunks
    3. Retrieve source entities and relations for those chunks
    4. Return subgraph with chunks + their context
    """

    def __init__(
        self,
        graph_accessor: GraphAccessor,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_provider: str = "huggingface",
        top_k: int = 10,
        timeout_sec: int = 30,
    ):
        """
        Initialize vector search retriever.

        Args:
            graph_accessor: GraphAccessor instance
            embedding_model: Model ID for embedding (currently using HuggingFace)
            embedding_provider: Provider (currently "huggingface")
            top_k: Top-K chunks to retrieve
            timeout_sec: Query timeout in seconds
        """
        super().__init__(graph_accessor, timeout_sec)
        self.embedding_model = embedding_model
        self.embedding_provider = embedding_provider
        self.top_k = max(1, min(100, top_k))
        self.embeddings = None
        self._initialize_embeddings()

    def _initialize_embeddings(self) -> None:
        """Initialize the embedding model using HuggingFace."""
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings

            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            logger.info(f"Initialized HuggingFace embeddings: sentence-transformers/all-MiniLM-L6-v2")
        except Exception as e:
            logger.warning(f"Failed to initialize embeddings: {e}")

    def retrieve(self, query: str) -> RetrievalStrategyResult:
        """
        Retrieve by semantic similarity to chunks.

        Args:
            query: User query text

        Returns:
            RetrievalStrategyResult with semantically similar chunks and their context
        """
        logger.info(f"VectorSearchRetriever: {query}")

        try:
            if not self.embeddings:
                return self._create_result(
                    error="Embeddings not initialized",
                    explanation="Vector search unavailable",
                )

            # Step 1: Embed the query
            query_embedding = self._embed_query(query)
            if not query_embedding:
                return self._create_result(
                    error="Query embedding failed",
                    explanation="Could not embed query",
                )

            logger.debug(f"Query embedded: dimension={len(query_embedding)}")

            # Step 2: Search chunks by similarity
            chunks = self.graph_accessor.search_chunks_by_embedding(
                query_embedding, self.top_k
            )
            if not chunks:
                return self._create_result(
                    explanation="No similar chunks found",
                    error="Vector search returned empty",
                )

            logger.debug(f"Found {len(chunks)} similar chunks")

            # Step 3: Build subgraph from chunks and their context
            subgraph = self._build_subgraph_from_chunks(chunks)

            if not subgraph.chunks:
                return self._create_result(
                    error="Failed to build subgraph",
                    explanation="No chunks in result",
                )

            confidence = self._compute_confidence(chunks, len(subgraph.entities))

            return self._create_result(
                subgraph=subgraph,
                confidence=confidence,
                explanation=f"Retrieved {len(chunks)} similar chunks, "
                f"{len(subgraph.entities)} entities, {len(subgraph.relations)} relations",
            )

        except Exception as e:
            logger.error(f"VectorSearchRetriever error: {e}")
            return self._create_result(
                error=str(e),
                explanation="Vector search failed",
            )

    # ─────────────────────────────────────────────────────────────────────────────
    # Private methods
    # ─────────────────────────────────────────────────────────────────────────────

    def _embed_query(self, query: str) -> list[float]:
        """Embed a query string."""
        try:
            embeddings = self.embeddings.embed_query(query)
            return embeddings
        except Exception as e:
            logger.error(f"Query embedding failed: {e}")
            return None

    def _build_subgraph_from_chunks(self, chunks) -> Subgraph:
        """
        Build a subgraph from chunks.

        For each chunk, retrieve:
        - The chunk itself
        - All entities that mention this chunk
        - Relations between those entities
        """
        subgraph = Subgraph()

        # Add chunks
        for chunk in chunks:
            subgraph.chunks[chunk.id] = chunk

        # For each chunk, find entities mentioning it
        # This is a simplified approach; ideally we'd use a Cypher query
        # to get all entities -> chunk -> entity paths
        for chunk in chunks:
            # In practice, Stage 2 would store MENTIONED_IN relationships
            # For now, we retrieve chunks and note they need entity context
            # A full implementation would query Neo4j for entities by chunk
            pass

        return subgraph

    def _compute_confidence(
        self,
        chunks,
        entity_count: int,
    ) -> float:
        """
        Compute confidence based on chunk similarity and context coverage.

        Heuristic: more chunks + more entities = higher confidence.
        Range: [0.3, 0.9]
        """
        chunk_count = len(chunks)

        # Confidence increases with number of relevant chunks and entities
        chunk_factor = min(chunk_count / 10, 1.0) * 0.5  # Up to 0.5
        entity_factor = min(entity_count / 5, 1.0) * 0.4  # Up to 0.4

        confidence = 0.3 + chunk_factor + entity_factor
        return min(0.9, confidence)
