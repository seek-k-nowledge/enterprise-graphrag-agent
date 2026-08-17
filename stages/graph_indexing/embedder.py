"""
Chunk embedding and vector index management for Stage 2.

Embeds chunk text using a configured provider (Anthropic or OpenAI) and stores
vectors in Neo4j for semantic search. Supports batching, retry, and error handling.

Usage:
    embedder = ChunkEmbedder(
        neo4j_client=client,
        model="text-embedding-3-large",
        provider="openai",
        batch_size=100,
        dimensions=1536
    )
    result = embedder.embed_chunks(chunks)
"""

import logging
from typing import Optional
import os

from stages.extraction.schemas import Chunk
from .neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


class ChunkEmbedder:
    """
    Embeds chunk text and stores vectors in Neo4j.

    Supports Anthropic and OpenAI embedding models via LangChain.
    All embeddings are deterministic given the same model and input.
    """

    def __init__(
        self,
        neo4j_client: Neo4jClient,
        model: str = "text-embedding-3-large",
        provider: str = "openai",
        batch_size: int = 100,
        dimensions: int = 1536,
        max_retries: int = 3,
    ):
        """
        Initialize the embedder.

        Args:
            neo4j_client: Connected Neo4jClient instance
            model: Model ID (e.g., text-embedding-3-large, claude-embedding-20250115)
            provider: "openai" or "anthropic"
            batch_size: Chunks per embedding API call
            dimensions: Expected vector dimension (validation only)
            max_retries: Retry count on transient failures
        """
        self.client = neo4j_client
        self.model = model
        self.provider = provider
        self.batch_size = batch_size
        self.dimensions = dimensions
        self.max_retries = max_retries
        self.embeddings = None
        self._initialize_embeddings()

    def _initialize_embeddings(self) -> None:
        """Initialize the embedding model via LangChain."""
        try:
            if self.provider.lower() == "openai":
                from langchain_openai import OpenAIEmbeddings

                self.embeddings = OpenAIEmbeddings(
                    model=self.model,
                    api_key=os.getenv("OPENAI_API_KEY"),
                )
                logger.info(f"Initialized OpenAI embeddings: {self.model}")

            elif self.provider.lower() == "anthropic":
                from langchain_anthropic import AnthropicEmbeddings

                self.embeddings = AnthropicEmbeddings(
                    model=self.model,
                    api_key=os.getenv("ANTHROPIC_API_KEY"),
                )
                logger.info(f"Initialized Anthropic embeddings: {self.model}")

            else:
                raise ValueError(
                    f"Unknown embedding provider: {self.provider}. "
                    f"Expected 'openai' or 'anthropic'"
                )
        except ImportError as e:
            logger.error(f"Failed to import embedding provider: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            raise

    def embed_chunks(
        self,
        chunks: list[Chunk],
        document_id: str,
        skip_existing: bool = True,
    ) -> "EmbedResult":
        """
        Embed a list of chunks and store vectors in Neo4j.

        Args:
            chunks: List of Chunk objects (from Stage 1)
            document_id: Document ID for logging and tracking
            skip_existing: If True, skip chunks that already have embeddings

        Returns:
            EmbedResult with counts and error list
        """
        logger.info(
            f"Embedding {len(chunks)} chunks for document {document_id} "
            f"using {self.provider}/{self.model}"
        )

        result = EmbedResult(
            document_id=document_id,
            total_chunks=len(chunks),
            model=self.model,
            provider=self.provider,
            dimensions=self.dimensions,
        )

        if not chunks:
            return result

        # Filter out chunks that already have embeddings (if skip_existing=True)
        chunks_to_embed = chunks
        if skip_existing:
            chunks_to_embed = self._filter_unembedded_chunks(chunks)
            result.skipped_existing = len(chunks) - len(chunks_to_embed)

        if not chunks_to_embed:
            logger.info(f"All chunks already embedded; skipping")
            return result

        # Batch embed chunks
        for i in range(0, len(chunks_to_embed), self.batch_size):
            batch = chunks_to_embed[i : i + self.batch_size]
            self._embed_batch(batch, result)

        logger.info(
            f"Embedding complete: {result.chunks_embedded} embedded, "
            f"{result.vectors_indexed} indexed, {len(result.errors)} errors"
        )

        return result

    # ─────────────────────────────────────────────────────────────────────────────
    # Private methods
    # ─────────────────────────────────────────────────────────────────────────────

    def _filter_unembedded_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """
        Filter out chunks that already have embeddings in Neo4j.

        Returns only chunks without embeddings, identified by querying for
        chunks with missing 'embedding' property.
        """
        if not chunks:
            return []

        chunk_ids = [c.id for c in chunks]
        cypher = """
        MATCH (c:Chunk)
        WHERE c.id IN $chunk_ids AND (c.embedding IS NULL OR c.embedding = [])
        RETURN c.id as chunk_id
        """

        result = self.client.query(
            cypher,
            parameters={"chunk_ids": chunk_ids},
            read_only=True,
        )

        unembedded_ids = {r["chunk_id"] for r in result.records}
        return [c for c in chunks if c.id in unembedded_ids]

    def _embed_batch(self, batch: list[Chunk], result: "EmbedResult") -> None:
        """
        Embed a batch of chunks and store vectors in Neo4j.

        Updates result.chunks_embedded, result.vectors_indexed, and result.errors.
        """
        texts = [chunk.text for chunk in batch]
        chunk_ids = [chunk.id for chunk in batch]

        try:
            # Call embedding API
            logger.debug(f"Embedding batch of {len(texts)} chunks")
            vectors = self._call_embedding_api(texts)

            if len(vectors) != len(texts):
                raise ValueError(
                    f"Expected {len(texts)} vectors, got {len(vectors)}"
                )

            # Validate vector dimensions
            for i, vec in enumerate(vectors):
                if len(vec) != self.dimensions:
                    error = {
                        "type": "dimension_mismatch",
                        "chunk_id": chunk_ids[i],
                        "expected_dimensions": self.dimensions,
                        "actual_dimensions": len(vec),
                    }
                    result.errors.append(error)
                    logger.warning(f"Vector dimension mismatch: {error}")
                    # Replace with zero vector to allow write to proceed
                    vectors[i] = [0.0] * self.dimensions

            # Store vectors in Neo4j
            indexed = self._store_vectors_in_neo4j(chunk_ids, vectors)
            result.chunks_embedded += len(texts)
            result.vectors_indexed += indexed

        except Exception as e:
            logger.error(f"Error embedding batch: {e}")
            error = {
                "type": "embedding_error",
                "chunk_ids": chunk_ids,
                "error": str(e),
            }
            result.errors.append(error)

    def _call_embedding_api(self, texts: list[str]) -> list[list[float]]:
        """
        Call the embedding API and return vectors.

        Handles retries on transient errors.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors (one per text)

        Raises:
            Exception if all retries fail
        """
        if not self.embeddings:
            raise RuntimeError("Embeddings not initialized")

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug(f"Embedding batch (attempt {attempt}/{self.max_retries})")
                vectors = self.embeddings.embed_documents(texts)
                logger.debug(f"Embedding succeeded: {len(vectors)} vectors")
                return vectors
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Embedding error on attempt {attempt}/{self.max_retries}: {e}"
                )
                if attempt < self.max_retries:
                    import time

                    time.sleep(2 ** attempt)  # Exponential backoff

        if last_error:
            raise last_error
        raise RuntimeError("Embedding failed after all retries")

    def _store_vectors_in_neo4j(
        self,
        chunk_ids: list[str],
        vectors: list[list[float]],
    ) -> int:
        """
        Store embedding vectors in Neo4j by updating chunk nodes.

        Args:
            chunk_ids: List of chunk IDs
            vectors: List of embedding vectors

        Returns:
            Count of chunks successfully indexed
        """
        if len(chunk_ids) != len(vectors):
            raise ValueError(
                f"Mismatch: {len(chunk_ids)} chunk IDs but {len(vectors)} vectors"
            )

        indexed = 0

        # Use UNWIND for batch update efficiency
        cypher = """
        UNWIND $chunks as chunk_data
        MATCH (c:Chunk {id: chunk_data.chunk_id})
        SET c.embedding = chunk_data.vector
        RETURN c.id as chunk_id
        """

        chunk_data = [
            {"chunk_id": chunk_id, "vector": vector}
            for chunk_id, vector in zip(chunk_ids, vectors)
        ]

        try:
            result = self.client.query(
                cypher,
                parameters={"chunks": chunk_data},
                read_only=False,
            )
            indexed = len(result.records)
            logger.debug(f"Stored {indexed} vectors in Neo4j")
        except Exception as e:
            logger.error(f"Error storing vectors in Neo4j: {e}")
            raise

        return indexed

    def ensure_vector_index(self) -> None:
        """
        Ensure vector index exists on Chunk.embedding.

        Creates the index if it doesn't exist. Neo4j 5.16+ required.
        For earlier versions, this is a no-op.
        """
        try:
            cypher = """
            CREATE VECTOR INDEX chunk_embedding_idx
            IF NOT EXISTS
            FOR (c:Chunk) ON (c.embedding)
            OPTIONS {indexConfig: {vector: {dimensions: $dimensions, similarity_function: "cosine"}}}
            """

            self.client.query(
                cypher,
                parameters={"dimensions": self.dimensions},
                read_only=False,
            )
            logger.info(f"Vector index ensured on Chunk.embedding (dimension: {self.dimensions})")
        except Exception as e:
            # Vector indexes may not be supported in all Neo4j versions
            logger.warning(f"Could not create vector index: {e}. Proceeding without it.")


class EmbedResult:
    """Result of chunk embedding operation."""

    def __init__(
        self,
        document_id: str,
        total_chunks: int,
        model: str,
        provider: str,
        dimensions: int,
    ):
        self.document_id = document_id
        self.total_chunks = total_chunks
        self.model = model
        self.provider = provider
        self.dimensions = dimensions
        self.chunks_embedded = 0
        self.vectors_indexed = 0
        self.skipped_existing = 0
        self.errors = []

    def to_dict(self) -> dict:
        """Convert to dict for logging or serialization."""
        return {
            "document_id": self.document_id,
            "total_chunks": self.total_chunks,
            "chunks_embedded": self.chunks_embedded,
            "vectors_indexed": self.vectors_indexed,
            "skipped_existing": self.skipped_existing,
            "model": self.model,
            "provider": self.provider,
            "dimensions": self.dimensions,
            "errors": self.errors,
        }

    def __repr__(self) -> str:
        return (
            f"EmbedResult(document_id={self.document_id}, "
            f"chunks_embedded={self.chunks_embedded}/{self.total_chunks}, "
            f"vectors_indexed={self.vectors_indexed}, "
            f"errors={len(self.errors)})"
        )
