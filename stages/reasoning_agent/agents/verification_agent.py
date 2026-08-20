"""
Verification agent: check consistency and validate citations.

Ensures answer is logically consistent, citations are valid, and identifies gaps.
"""

import logging
import os

from langchain_groq import ChatGroq
from ..schemas import VerificationOutput, ReasoningStep

logger = logging.getLogger(__name__)


class VerificationAgent:
    """
    Verifies answers for consistency and citation validity.

    Checks:
    - Every claim has a citation
    - Citations point to valid graph elements
    - Claims are logically consistent
    - Identifies gaps or limitations
    """

    def __init__(self, model: str = "llama-3.3-70b-versatile"):
        """
        Initialize verification agent.

        Args:
            model: Groq model ID for verification (default: llama-3.3-70b-versatile)
        """
        self.model = model
        self.llm = None
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize ChatGroq LLM client."""
        try:
            self.llm = ChatGroq(
                model_name=self.model,
                groq_api_key=os.getenv("GROQ_API_KEY"),
                temperature=0.0,
            )
            logger.info(f"Initialized ChatGroq for {self.model}")
        except Exception as e:
            logger.warning(f"Failed to initialize ChatGroq: {e}")

    def verify(
        self,
        answer_text: str,
        citations,
        subgraph,
    ) -> tuple[VerificationOutput, ReasoningStep]:
        """
        Verify answer consistency and citation validity.

        Args:
            answer_text: Synthesized answer
            citations: List of Citation objects
            subgraph: Retrieved subgraph for validation

        Returns:
            Tuple of (VerificationOutput, ReasoningStep for audit trail)
        """
        logger.info("VerificationAgent verifying answer")

        step = ReasoningStep(
            step_type="verification",
            agent="VerificationAgent",
            input={
                "answer_length": len(answer_text),
                "citation_count": len(citations),
            },
        )

        try:
            output = VerificationOutput()

            # Check 1: Citation validity
            for citation in citations:
                if not self._is_valid_citation(citation, subgraph):
                    output.citation_issues.append(
                        f"Citation {citation.source_id} not found in subgraph"
                    )

            # Check 2: Coverage (every claim should have citation)
            if len(citations) < 2 and len(answer_text) > 100:
                output.gaps.append("Answer is long but has few citations")

            # Check 3: Consistency check via LLM (if available)
            if self.llm:
                consistency_issues = self._check_logical_consistency(
                    answer_text, citations, subgraph
                )
                output.consistency_issues.extend(consistency_issues)

            # Determine validity and confidence adjustment
            output.valid = len(output.citation_issues) == 0 and len(
                output.consistency_issues
            ) == 0
            output.confidence_adjustment = -0.2 if not output.valid else 0.0

            if output.gaps:
                output.confidence_adjustment -= 0.1

            step.output = {
                "valid": output.valid,
                "citation_issues": len(output.citation_issues),
                "consistency_issues": len(output.consistency_issues),
                "gaps": len(output.gaps),
                "confidence_adjustment": output.confidence_adjustment,
            }
            step.reasoning = f"Verification: {'valid' if output.valid else 'issues found'}"

            logger.info(f"Verification complete: {step.output}")
            return output, step

        except Exception as e:
            logger.error(f"Verification agent error: {e}")
            step.output = {"error": str(e)}
            raise

    # ─────────────────────────────────────────────────────────────────────────────
    # Private methods
    # ─────────────────────────────────────────────────────────────────────────────

    def _is_valid_citation(self, citation, subgraph) -> bool:
        """Check if a citation points to a valid graph element."""
        if citation.source_type == "node":
            return citation.source_id in subgraph.entities
        elif citation.source_type == "edge":
            return citation.source_id in subgraph.relations
        elif citation.source_type == "chunk":
            return citation.source_id in subgraph.chunks
        return False

    def _check_logical_consistency(
        self,
        answer_text: str,
        citations,
        subgraph,
    ) -> list[str]:
        """
        Check logical consistency of answer vs. citations via LLM.

        Returns:
            List of identified consistency issues
        """
        if not self.llm:
            return []

        issues = []

        prompt = f"""Review this answer and its citations for logical consistency.

Answer: {answer_text[:500]}

Citations:
{self._format_citations(citations)}

Known facts from graph:
{self._format_subgraph(subgraph)}

Identify any:
1. Claims not supported by citations
2. Contradictions between claims
3. Ungrounded statements

List issues found (one per line), or "VALID" if no issues."""

        try:
            from langchain_core.messages import HumanMessage

            response = self.llm.invoke([HumanMessage(content=prompt)])
            result_text = response.content.lower()
            if "valid" not in result_text:
                # Extract issues
                for line in result_text.split("\n"):
                    if line.strip() and not line.startswith("known"):
                        issues.append(line.strip())

            return issues[:5]  # Limit to top 5 issues

        except Exception as e:
            logger.warning(f"LLM consistency check failed: {e}")
            return []

    def _format_citations(self, citations) -> str:
        """Format citations for LLM review."""
        lines = []
        for i, c in enumerate(citations[:5], 1):
            lines.append(f"{i}. [{c.source_type}] {c.source_id}: {c.claim}")
        return "\n".join(lines)

    def _format_subgraph(self, subgraph) -> str:
        """Format subgraph for LLM review."""
        lines = []
        for entity in list(subgraph.entities.values())[:3]:
            lines.append(f"- {entity.canonical_name}: {entity.description[:100]}")
        return "\n".join(lines)
