"""
Stage 3: Reasoning Agent & Query Orchestration

Public API for answering queries against the Stage 2 knowledge graph.

Entry point:
    answer = answer_query(
        query_payload,
        graph_accessor,
        config=ReasoningConfig()
    )
    # answer: AnswerPayload with grounded answer and citations
"""

import logging
from typing import Optional

from stages.graph_indexing.neo4j_client import Neo4jClient
from .schemas import (
    QueryPayload,
    AnswerPayload,
    ReasoningConfig,
)
from .graph_accessor import GraphAccessor
from .agent_graph import AgentGraph

logger = logging.getLogger(__name__)

# Public API
__all__ = [
    "answer_query",
    "create_graph_accessor",
    "AnswerPayload",
    "QueryPayload",
    "ReasoningConfig",
    "GraphAccessor",
    "AgentGraph",
]


def create_graph_accessor(
    neo4j_client: Neo4jClient,
    enable_caching: bool = True,
    cache_ttl_sec: int = 3600,
    query_timeout_sec: int = 30,
) -> GraphAccessor:
    """
    Factory function to create a GraphAccessor.

    Args:
        neo4j_client: Connected Neo4jClient from Stage 2
        enable_caching: Enable query result caching
        cache_ttl_sec: Cache TTL in seconds
        query_timeout_sec: Query timeout in seconds

    Returns:
        Initialized GraphAccessor
    """
    accessor = GraphAccessor(
        neo4j_client=neo4j_client,
        enable_caching=enable_caching,
        cache_ttl_sec=cache_ttl_sec,
        query_timeout_sec=query_timeout_sec,
    )
    logger.info("Created GraphAccessor")
    return accessor


def answer_query(
    query_payload: QueryPayload,
    graph_accessor: GraphAccessor,
    config: Optional[ReasoningConfig] = None,
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
) -> AnswerPayload:
    """
    Answer a user query using the agent swarm.

    This is the main entry point for Stage 3. Orchestrates:
    1. Query routing (classify intent, select strategies)
    2. Retrieval (execute strategies, aggregate results)
    3. Synthesis (ground answer with citations)
    4. Verification (check consistency, validate citations)

    Args:
        query_payload: QueryPayload with user query
        graph_accessor: GraphAccessor instance (read-only interface to Stage 2)
        config: ReasoningConfig (uses defaults if None)
        provider: LLM provider override (cerebras, groq, anthropic)
        api_key: API key for the specified provider

    Returns:
        AnswerPayload with grounded answer, citations, and full audit trail

    Raises:
        Exception on critical failures (re-raised after logging)
    """
    if config is None:
        config = ReasoningConfig()

    # Override provider/api_key if provided
    if provider is not None:
        config.llm_provider = provider
    if api_key is not None:
        config.llm_api_key = api_key

    logger.info(
        f"Answering query: '{query_payload.text}' "
        f"(session={query_payload.session_id})"
    )

    try:
        # Create agent graph
        agent_graph = AgentGraph(
            graph_accessor=graph_accessor,
            config=config,
            enable_checkpointing=config.enable_checkpointing,
        )

        # Execute agent swarm
        answer = agent_graph.execute(
            query_payload=query_payload,
            session_id=query_payload.session_id,
        )

        logger.info(
            f"Query answered: {len(answer.answer_text)} chars, "
            f"{len(answer.citations)} citations, "
            f"confidence={answer.confidence:.2f}, "
            f"latency={answer.latency_ms}ms"
        )

        return answer

    except Exception as e:
        logger.error(f"Query answering failed: {e}")
        raise


# Convenience: direct query answering from text
def answer_query_text(
    query_text: str,
    graph_accessor: GraphAccessor,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    config: Optional[ReasoningConfig] = None,
) -> AnswerPayload:
    """
    Answer a query from plain text (convenience wrapper).

    Args:
        query_text: User query as string
        graph_accessor: GraphAccessor instance
        user_id: Optional user identifier
        session_id: Optional session identifier
        config: ReasoningConfig

    Returns:
        AnswerPayload with grounded answer
    """
    payload = QueryPayload(
        text=query_text,
        user_id=user_id,
        session_id=session_id,
    )
    return answer_query(payload, graph_accessor, config)
