"""
LangGraph state machine orchestration for Stage 3 agent swarm.

Coordinates agents (router, retrieval, synthesis, verification) through
a stateful graph with checkpointing for long-running queries.
"""

import logging
from datetime import datetime
from typing import Optional

try:
    from langgraph.graph import StateGraph
    from langgraph.checkpoint.memory import MemorySaver
except ImportError:
    StateGraph = None
    MemorySaver = None

from .schemas import (
    QueryPayload,
    AgentState,
    AnswerPayload,
    ReasoningConfig,
    TokenUsage,
)
from .graph_accessor import GraphAccessor
from .query_router import QueryRouter
from .agents import (
    RouterAgent,
    RetrievalAgent,
    SynthesisAgent,
    VerificationAgent,
)

logger = logging.getLogger(__name__)


class AgentGraph:
    """
    LangGraph-based agent orchestration for query answering.

    Workflow:
    1. Route query → determine strategy (RouterAgent or QueryRouter)
    2. Retrieve → execute strategies (RetrievalAgent)
    3. Synthesize → write answer with citations (SynthesisAgent)
    4. Verify → check consistency and citations (VerificationAgent)
    5. Return → AnswerPayload
    """

    def __init__(
        self,
        graph_accessor: GraphAccessor,
        config: ReasoningConfig = None,
        enable_checkpointing: bool = True,
    ):
        """
        Initialize the agent graph.

        Args:
            graph_accessor: GraphAccessor instance
            config: ReasoningConfig (uses defaults if None)
            enable_checkpointing: Enable LangGraph checkpointing
        """
        self.graph_accessor = graph_accessor
        self.config = config or ReasoningConfig()
        self.enable_checkpointing = enable_checkpointing

        # Initialize agents
        self.router_agent = RouterAgent(
            model=self.config.query_classifier_model,
            available_strategies=self.config.retrieval_strategies,
        )
        self.query_router = QueryRouter(self.config.retrieval_strategies)
        self.retrieval_agent = RetrievalAgent(
            graph_accessor,
            max_hops=self.config.max_traversal_hops,
            top_k=self.config.vector_search_top_k,
        )
        self.synthesis_agent = SynthesisAgent(
            model=self.config.synthesis_model,
        )
        self.verification_agent = VerificationAgent(
            model=self.config.synthesis_model,
        )

        # Initialize graph
        self.graph = None
        self.compiled_graph = None
        self._build_graph()

    def _build_graph(self) -> None:
        """Build the LangGraph state machine."""
        if StateGraph is None:
            logger.warning("LangGraph not available; using sequential execution")
            return

        # Create state graph
        self.graph = StateGraph(AgentState)

        # Add nodes (agents)
        self.graph.add_node("route", self._node_route)
        self.graph.add_node("retrieve", self._node_retrieve)
        self.graph.add_node("synthesize", self._node_synthesize)
        self.graph.add_node("verify", self._node_verify)
        self.graph.add_node("return", self._node_return)

        # Add edges
        self.graph.add_edge("route", "retrieve")
        self.graph.add_edge("retrieve", "synthesize")
        self.graph.add_edge("synthesize", "verify")
        self.graph.add_edge("verify", "return")

        # Set entry point
        self.graph.set_entry_point("route")

        # Compile with checkpointing if enabled
        if self.enable_checkpointing:
            try:
                checkpointer = MemorySaver()
                self.compiled_graph = self.graph.compile(checkpointer=checkpointer)
                logger.info("Compiled graph with checkpointing enabled")
            except Exception as e:
                logger.warning(f"Could not enable checkpointing: {e}")
                self.compiled_graph = self.graph.compile()
        else:
            self.compiled_graph = self.graph.compile()

    def execute(
        self,
        query_payload: QueryPayload,
        session_id: Optional[str] = None,
    ) -> AnswerPayload:
        """
        Execute the agent graph to answer a query.

        Args:
            query_payload: QueryPayload with user query
            session_id: Optional session ID for checkpoint resumption

        Returns:
            AnswerPayload with grounded answer

        Raises:
            Exception on critical failures
        """
        logger.info(f"Executing agent graph for query: {query_payload.text}")

        # Create initial state
        state = AgentState(
            query=query_payload,
        )

        try:
            if self.compiled_graph:
                # LangGraph execution with checkpointing
                final_state = self.compiled_graph.invoke(
                    state,
                    {"configurable": {"thread_id": session_id or "default"}},
                )
            else:
                # Sequential execution without LangGraph
                final_state = self._sequential_execute(state)

            # Extract answer
            if hasattr(final_state, "final_answer"):
                return final_state.final_answer

            # Construct answer from state
            return self._construct_answer(final_state)

        except Exception as e:
            logger.error(f"Agent graph execution failed: {e}")
            raise

    def _sequential_execute(self, state: AgentState) -> AgentState:
        """Fallback: sequential execution without LangGraph."""
        logger.info("Using sequential execution (LangGraph unavailable)")

        try:
            # Route
            state = self._node_route(state)
            # Retrieve
            state = self._node_retrieve(state)
            # Synthesize
            state = self._node_synthesize(state)
            # Verify
            state = self._node_verify(state)
            # Return
            state = self._node_return(state)
            return state
        except Exception as e:
            logger.error(f"Sequential execution failed: {e}")
            raise

    # ─────────────────────────────────────────────────────────────────────────────
    # Node implementations
    # ─────────────────────────────────────────────────────────────────────────────

    def _node_route(self, state: AgentState) -> AgentState:
        """Route node: classify query and select strategies."""
        logger.info("Node: route")
        try:
            # Try LLM-based routing first
            try:
                decision, step = self.router_agent.route(state.query.text)
            except Exception:
                # Fallback to rule-based
                logger.debug("Falling back to rule-based routing")
                decision = self.query_router.route(state.query.text)
                step = None

            state.query_classification = decision.query_classification
            state.preferred_strategies = decision.strategies
            if step:
                state.reasoning_steps.append(step)

            return state
        except Exception as e:
            logger.error(f"Route node error: {e}")
            raise

    def _node_retrieve(self, state: AgentState) -> AgentState:
        """Retrieve node: execute selected strategies."""
        logger.info("Node: retrieve")
        try:
            retrieval_result, step = self.retrieval_agent.retrieve(
                state.query.text,
                state.preferred_strategies,
                {"max_hops": state.query.max_hops, "top_k": state.query.top_k},
            )

            state.retrieval_results = retrieval_result
            state.reasoning_steps.append(step)

            return state
        except Exception as e:
            logger.error(f"Retrieve node error: {e}")
            raise

    def _node_synthesize(self, state: AgentState) -> AgentState:
        """Synthesize node: ground answer with citations."""
        logger.info("Node: synthesize")
        try:
            if not state.retrieval_results or not state.retrieval_results.merged_subgraph.entities:
                state.draft_answer = "No information found for your query."
                state.citations = []
                return state

            synthesis_output, step = self.synthesis_agent.synthesize(
                state.query.text,
                state.retrieval_results.merged_subgraph,
            )

            state.draft_answer = synthesis_output.answer_text
            state.citations = synthesis_output.citations
            state.reasoning_steps.append(step)

            return state
        except Exception as e:
            logger.error(f"Synthesize node error: {e}")
            raise

    def _node_verify(self, state: AgentState) -> AgentState:
        """Verify node: check consistency and citations."""
        logger.info("Node: verify")
        try:
            if not state.draft_answer:
                return state

            verification_output, step = self.verification_agent.verify(
                state.draft_answer,
                state.citations,
                state.retrieval_results.merged_subgraph if state.retrieval_results else None,
            )

            state.verification_feedback = verification_output.model_dump()
            state.reasoning_steps.append(step)

            return state
        except Exception as e:
            logger.error(f"Verify node error: {e}")
            raise

    def _node_return(self, state: AgentState) -> AgentState:
        """Return node: construct final AnswerPayload."""
        logger.info("Node: return")
        try:
            answer = self._construct_answer(state)
            state.final_answer = answer
            return state
        except Exception as e:
            logger.error(f"Return node error: {e}")
            raise

    # ─────────────────────────────────────────────────────────────────────────────
    # Answer construction
    # ─────────────────────────────────────────────────────────────────────────────

    def _construct_answer(self, state: AgentState) -> AnswerPayload:
        """Construct final AnswerPayload from agent state."""
        # Calculate confidence
        confidence = 0.5
        if state.retrieval_results:
            # Base confidence from retrieval
            if state.retrieval_results.strategy_results:
                confidence = state.retrieval_results.strategy_results[0].confidence

        # Adjust for verification
        if state.verification_feedback:
            confidence += state.verification_feedback.get("confidence_adjustment", 0.0)
            confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]

        # Build retrieval trace
        trace = None
        if state.retrieval_results:
            trace = {
                "query_classification": state.query_classification,
                "retrieval_strategies": state.retrieval_results.strategies_executed,
                "subgraph_stats": {
                    "nodes_touched": state.retrieval_results.total_entities,
                    "edges_touched": state.retrieval_results.total_relations,
                    "chunks_used": state.retrieval_results.total_chunks,
                },
                "reasoning_steps": [s.dict() for s in state.reasoning_steps],
                "errors": state.errors,
            }

        # Construct answer payload
        answer = AnswerPayload(
            query=state.query.text,
            answer_text=state.draft_answer or "No answer generated.",
            citations=state.citations,
            confidence=confidence,
            token_usage=state.token_usage,
            user_id=state.query.user_id,
            session_id=state.query.session_id,
            gaps=(
                state.verification_feedback.get("gaps", [])
                if state.verification_feedback
                else []
            ),
        )

        # Calculate latency
        if state.started_at:
            latency = (datetime.utcnow() - state.started_at).total_seconds() * 1000
            answer.latency_ms = int(latency)

        return answer
