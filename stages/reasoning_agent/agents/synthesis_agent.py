"""
Synthesis agent: ground answers with citations.

Converts retrieval results into natural language answers, grounded in graph elements.
"""

import logging

from ...shared.llm_provider import get_llm
from ..schemas import SynthesisOutput, Citation, ReasoningStep

logger = logging.getLogger(__name__)


class SynthesisAgent:
    """
    Synthesizes natural language answers grounded in retrieved subgraph.

    Uses provider-agnostic LLM layer (Cerebras + Groq failover) to write answers
    that cite specific nodes, edges, and chunks.
    """

    def __init__(self, model: str = "gpt-oss-120b"):
        """
        Initialize synthesis agent.

        Args:
            model: Model ID for answer generation (default: gpt-oss-120b, available on Cerebras and Groq)
        """
        self.model = model
        self.llm = None
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize LLM client via provider layer (Cerebras + Groq failover)."""
        try:
            # temperature=0.0 ensures deterministic, reproducible answers for consistency
            self.llm = get_llm(model=self.model, temperature=0.0)
            logger.info(f"Initialized LLM for {self.model}")
        except Exception as e:
            logger.warning(f"Failed to initialize LLM: {e}")

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
            if not self.llm:
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

    def _llm_synthesize(self, query: str, subgraph) -> tuple[str, list[int]]:
        """
        Use LLM to synthesize answer with numbered citations at sentence ends.

        Returns:
            Tuple of (answer_text_with_references, list of citation indices used)
        """
        if not self.llm:
            raise RuntimeError("LLM client not initialized")

        # Build subgraph summary for context
        context = self._build_context(subgraph)

        prompt = f"""You are an expert research assistant providing comprehensive answers
grounded in provided knowledge.

Question: {query}

Knowledge Graph:
{context}

Instructions:
1. Answer the question as thoroughly and confidently as possible using ONLY the
   information provided in the knowledge graph above.
2. Provide a complete, well-structured answer without hedging or disclaimers.
3. Make specific factual claims supported by the graph elements provided.
4. Cite sources by placing a number in brackets at the END of sentences: [1], [2], etc.
   Place the citation after the period: "Claim here. [1]" — never wrap cited content.
5. If the graph has limited information about the question, briefly note any key
   gaps at the very end (one sentence max), but lead with what you DO know.

Answer with confidence and completeness:"""

        try:
            from langchain_core.messages import HumanMessage
            import re

            response = self.llm.invoke([HumanMessage(content=prompt)])
            answer_text = response.content

            # Log which provider served this call
            if hasattr(self.llm, 'last_provider'):
                logger.info(f"Synthesis: LLM call served by {self.llm.last_provider}")

            # Extract numbered citation indices [1], [2], etc.
            citations_used = set()
            for match in re.finditer(r'\[(\d+)\]', answer_text):
                citations_used.add(int(match.group(1)))

            # Map each used index to a source from the subgraph
            citation_map = self._build_citation_map(subgraph, max(citations_used) if citations_used else 0)

            # Build reference section
            reference_section = self._build_reference_section(citation_map, citations_used)

            # Append references to answer
            if reference_section:
                answer_text = answer_text.rstrip() + "\n\n" + reference_section

            return answer_text, sorted(list(citations_used))

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

    def _ground_citations(self, citation_indices: list[int], subgraph) -> list[Citation]:
        """
        Convert numbered citation indices into Citation objects grounded in subgraph.

        Args:
            citation_indices: List of citation numbers [1], [2], etc. from LLM
            subgraph: Subgraph to validate against

        Returns:
            List of Citation objects
        """
        citations = []
        citation_map = self._build_citation_map(subgraph, max(citation_indices) if citation_indices else 0)

        for idx in citation_indices:
            if idx in citation_map:
                source_type, source_id, source_text = citation_map[idx]
                citations.append(
                    Citation(
                        claim=f"Reference [{idx}]",
                        source_type=source_type,
                        source_id=source_id,
                        source_text=source_text,
                        confidence=0.85,
                    )
                )

        return citations

    def _build_citation_map(self, subgraph, num_citations: int) -> dict:
        """
        Build a map from citation indices to subgraph sources.

        Returns dict: {1: (source_type, source_id, source_text), ...}
        """
        citation_map = {}
        idx = 1

        # Map chunks first (highest similarity first) since they contain actual evidence
        sorted_chunks = sorted(
            subgraph.chunks.values(),
            key=lambda c: c.similarity_score if c.similarity_score else 0.0,
            reverse=True
        )
        for chunk in sorted_chunks:
            if idx > num_citations:
                break
            # Replace newlines with spaces for readable reference display
            clean_text = chunk.text.replace('\n', ' ')
            snippet = clean_text[:80] + "..." if len(clean_text) > 80 else clean_text
            citation_map[idx] = ("chunk", chunk.id, snippet)
            idx += 1

        # Then map entities
        for entity in list(subgraph.entities.values()):
            if idx > num_citations:
                break
            citation_map[idx] = ("node", entity.id, entity.canonical_name)
            idx += 1

        # Then map relations
        for rel in list(subgraph.relations.values()):
            if idx > num_citations:
                break
            citation_map[idx] = ("edge", rel.source_id, f"{rel.relation_type}")
            idx += 1

        return citation_map

    def _build_reference_section(self, citation_map: dict, citations_used: set) -> str:
        """
        Build a references section for numbered citations.

        Returns: Markdown-formatted reference list
        """
        if not citations_used or not citation_map:
            return ""

        lines = ["## References"]
        for idx in sorted(citations_used):
            if idx in citation_map:
                source_type, source_id, source_text = citation_map[idx]
                lines.append(f"[{idx}] {source_text}")

        return "\n".join(lines)

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

        # Chunks (sorted by similarity score descending)
        if subgraph.chunks:
            lines.append("\nKey passages:")
            sorted_chunks = sorted(
                subgraph.chunks.values(),
                key=lambda c: c.similarity_score if c.similarity_score else 0.0,
                reverse=True
            )

            for i, chunk in enumerate(sorted_chunks[:15], 1):
                # Increased from 100 to 350 characters to preserve full sentences and context
                # 15 chunks × ~300 chars ≈ 4500 chars, minimal overhead in LLM context
                max_chunk_len = 350
                text = chunk.text[:max_chunk_len] + "..." if len(chunk.text) > max_chunk_len else chunk.text
                # Log the actual lengths passed to LLM for verification
                actual_len = min(len(chunk.text), max_chunk_len)
                logger.debug(f"[CONTEXT-CHUNK-{i}] full_length={len(chunk.text)}, passed_to_llm={actual_len} chars")
                lines.append(f"  - {text}")

        return "\n".join(lines)
