"""
Synthesis agent: ground answers with citations.

Converts retrieval results into natural language answers, grounded in graph elements.
"""

import logging
import os

from ..schemas import SynthesisOutput, Citation, ReasoningStep

logger = logging.getLogger(__name__)


class SynthesisAgent:
    """
    Synthesizes natural language answers grounded in retrieved subgraph.

    Uses an LLM to write answers that cite specific nodes, edges, and chunks.
    """

    def __init__(self, model: str = "llama-3.3-70b-versatile"):
        """
        Initialize synthesis agent.

        Args:
            model: Groq model ID for answer generation (default: llama-3.3-70b-versatile)
        """
        self.model = model
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

    def synthesize(
        self,
        query: str,
        subgraph,
    ) -> tuple[SynthesisOutput, ReasoningStep]:
        """
        Synthesize an answer from the subgraph.

        Args:
            query: User query
            subgraph: Retrieved subgraph with entities, relations, chunks

        Returns:
            Tuple of (SynthesisOutput, ReasoningStep for audit trail)
        """
        logger.info(f"SynthesisAgent synthesizing answer for: {query}")

        step = ReasoningStep(
            step_type="synthesis",
            agent="SynthesisAgent",
            input={
                "query": query,
                "subgraph_size": {
                    "entities": len(subgraph.entities),
                    "relations": len(subgraph.relations),
                    "chunks": len(subgraph.chunks),
                },
            },
        )

        try:
            if not self.client:
                # Fallback: simple answer from subgraph
                answer_text = self._fallback_answer(query, subgraph)
                citations = self._extract_citations_from_subgraph(subgraph)
            else:
                # LLM-based synthesis
                answer_text, raw_citations = self._llm_synthesize(query, subgraph)
                citations = self._ground_citations(raw_citations, subgraph)

            output = SynthesisOutput(
                answer_text=answer_text,
                citations=citations,
                cited_entities=[c.source_id for c in citations if c.source_type == "node"],
                cited_relations=[c.source_id for c in citations if c.source_type == "edge"],
                reasoning=f"Synthesized answer with {len(citations)} citations from subgraph",
            )

            step.output = {
                "answer_length": len(answer_text),
                "citation_count": len(citations),
                "cited_entities": len(output.cited_entities),
            }
            step.reasoning = output.reasoning

            logger.info(f"Synthesis complete: {step.output}")
            return output, step

        except Exception as e:
            logger.error(f"Synthesis agent error: {e}")
            step.output = {"error": str(e)}
            raise

    # ─────────────────────────────────────────────────────────────────────────────
    # Private methods
    # ─────────────────────────────────────────────────────────────────────────────

    def _llm_synthesize(self, query: str, subgraph) -> tuple[str, list[str]]:
        """
        Use LLM to synthesize answer and extract citation claims.

        Returns:
            Tuple of (answer_text, list of citation claims)
        """
        if not self.client:
            raise RuntimeError("LLM client not initialized")

        # Build subgraph summary for context
        context = self._build_context(subgraph)

        prompt = f"""You are an expert research assistant. Answer the following question
using ONLY information from the provided knowledge graph. Every claim must be traceable
to the graph.

Question: {query}

Knowledge Graph:
{context}

Instructions:
1. Write a clear, concise answer using information from the graph
2. Each sentence should make one or more specific claims
3. Reference specific entities and relationships
4. Format citations inline: [CITATION: entity_id or relation_type]

Answer:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}],
            )

            answer_text = response.choices[0].message.content

            # Extract citation markers (simplified)
            import re

            citations = re.findall(r'\[CITATION:\s*([^\]]+)\]', answer_text)

            # Remove markers from answer
            answer_text = re.sub(r'\s*\[CITATION:[^\]]*\]', '', answer_text)

            return answer_text, citations

        except Exception as e:
            logger.error(f"LLM synthesis failed: {e}")
            raise

    def _fallback_answer(self, query: str, subgraph) -> str:
        """Generate a simple answer when LLM is unavailable."""
        if not subgraph.entities:
            return f"No information found for '{query}'."

        entity_list = ", ".join(
            [e.canonical_name for e in list(subgraph.entities.values())[:5]]
        )
        return f"Based on the knowledge graph, key entities related to '{query}' are: {entity_list}."

    def _extract_citations_from_subgraph(self, subgraph) -> list[Citation]:
        """Extract basic citations from subgraph structure."""
        citations = []

        # Create citations from top entities
        for entity in list(subgraph.entities.values())[:5]:
            citations.append(
                Citation(
                    claim=f"Entity: {entity.canonical_name}",
                    source_type="node",
                    source_id=entity.id,
                    source_text=entity.description or entity.canonical_name,
                    confidence=0.8,
                )
            )

        return citations

    def _ground_citations(self, citation_claims: list[str], subgraph) -> list[Citation]:
        """
        Convert citation claims into Citation objects grounded in subgraph.

        Args:
            citation_claims: List of citation identifiers from LLM
            subgraph: Subgraph to validate against

        Returns:
            List of Citation objects
        """
        citations = []

        for claim_id in citation_claims:
            # Try to find in entities
            if claim_id in subgraph.entities:
                entity = subgraph.entities[claim_id]
                citations.append(
                    Citation(
                        claim=f"Entity information about {entity.canonical_name}",
                        source_type="node",
                        source_id=entity.id,
                        source_text=entity.description or entity.canonical_name,
                        confidence=0.9,
                    )
                )

            # Try to find in relations
            elif claim_id in subgraph.relations:
                rel = subgraph.relations[claim_id]
                citations.append(
                    Citation(
                        claim=f"Relation: {rel.relation_type}",
                        source_type="edge",
                        source_id=claim_id,
                        source_text=rel.description or rel.relation_type,
                        confidence=rel.confidence,
                    )
                )

        return citations

    def _build_context(self, subgraph) -> str:
        """Build a text summary of the subgraph for LLM context."""
        lines = []

        # Entities
        if subgraph.entities:
            lines.append("Entities:")
            for entity in list(subgraph.entities.values())[:10]:
                desc = entity.description[:100] if entity.description else "(no description)"
                lines.append(f"  - {entity.canonical_name} ({entity.entity_type}): {desc}")

        # Relations
        if subgraph.relations:
            lines.append("\nRelations:")
            for rel in list(subgraph.relations.values())[:10]:
                lines.append(f"  - {rel.source_id} --[{rel.relation_type}]--> {rel.target_id}")

        # Chunks
        if subgraph.chunks:
            lines.append("\nKey passages:")
            for chunk in list(subgraph.chunks.values())[:5]:
                text = chunk.text[:100] + "..." if len(chunk.text) > 100 else chunk.text
                lines.append(f"  - {text}")

        return "\n".join(lines)
