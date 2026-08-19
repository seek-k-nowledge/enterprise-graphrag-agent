"""
Router agent: LLM-based query classification.

Classifies queries into types and selects appropriate retrieval strategies.
"""

import logging
import os

from ..schemas import RouterDecision, ReasoningStep

logger = logging.getLogger(__name__)


class RouterAgent:
    """
    LLM-based query router for classification and strategy selection.

    Uses Claude to understand query intent and select retrieval strategies.
    """

    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        available_strategies: list[str] = None,
    ):
        """
        Initialize router agent.

        Args:
            model: Groq model ID for routing (default: llama-3.3-70b-versatile)
            available_strategies: List of available retrieval strategies
        """
        self.model = model
        self.available_strategies = available_strategies or [
            "graph_traversal",
            "vector_search",
            "cypher_direct",
        ]
        self.client = None
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize Groq client."""
        try:
            from groq import Groq

            self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            logger.info(f"Initialized Groq client for {self.model}")
        except Exception as e:
            logger.warning(f"Failed to initialize Groq client: {e}")

    def route(self, query: str) -> tuple[RouterDecision, ReasoningStep]:
        """
        Classify query and select retrieval strategies.

        Args:
            query: User query text

        Returns:
            Tuple of (RouterDecision, ReasoningStep for audit trail)
        """
        logger.info(f"RouterAgent routing query: {query}")

        step = ReasoningStep(
            step_type="query_classification",
            agent="RouterAgent",
            input={"query": query},
        )

        try:
            if not self.client:
                # Fallback to rule-based routing
                from ..query_router import QueryRouter

                router = QueryRouter(self.available_strategies)
                decision = router.route(query)
                step.output = {
                    "classification": decision.query_classification,
                    "strategies": decision.strategies,
                    "method": "rule_based",
                }
            else:
                # LLM-based routing
                decision = self._llm_route(query)
                step.output = {
                    "classification": decision.query_classification,
                    "strategies": decision.strategies,
                    "seed_entities": decision.seed_entities,
                    "method": "llm_based",
                }

            step.reasoning = decision.reasoning

            logger.info(f"Route decision: {decision.query_classification} → {decision.strategies}")
            return decision, step

        except Exception as e:
            logger.error(f"Router agent error: {e}")
            step.output = {"error": str(e)}
            raise

    # ─────────────────────────────────────────────────────────────────────────────
    # Private methods
    # ─────────────────────────────────────────────────────────────────────────────

    def _llm_route(self, query: str) -> RouterDecision:
        """Use LLM to classify query and select strategies."""
        if not self.client:
            raise RuntimeError("LLM client not initialized")

        prompt = f"""Analyze this query and classify it. Respond in JSON format.

Query: "{query}"

Classify into ONE of:
- multi_hop: relational questions (how X relates to Y, paths, connections)
- semantic: descriptive questions (describe X, what is Y, properties)
- aggregate: structural questions (count, top-K, comparisons)
- hybrid: mixed or unclear (run all strategies)

Also:
1. Extract any named entities mentioned (list up to 3)
2. Select retrieval strategies in priority order (subset of: graph_traversal, vector_search, cypher_direct)

Response format:
{{
  "classification": "...",
  "entities": ["...", "..."],
  "strategies": ["...", "..."],
  "reasoning": "..."
}}"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )

            response_text = response.content[0].text

            # Parse JSON response
            import json
            import re

            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if not json_match:
                logger.warning("Could not parse JSON response from router")
                raise ValueError("Invalid response format")

            json_str = json_match.group(0)
            parsed = json.loads(json_str)

            # Build decision
            decision = RouterDecision(
                query_classification=parsed.get("classification", "hybrid"),
                seed_entities=parsed.get("entities", []),
                strategies=[
                    s for s in parsed.get("strategies", [])
                    if s in self.available_strategies
                ],
                reasoning=parsed.get("reasoning", ""),
            )

            # Ensure at least one strategy
            if not decision.strategies:
                decision.strategies = self.available_strategies[:2]

            return decision

        except Exception as e:
            logger.error(f"LLM routing failed: {e}")
            raise
