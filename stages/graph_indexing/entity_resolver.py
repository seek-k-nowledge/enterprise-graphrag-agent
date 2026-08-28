"""
Entity resolution and deduplication for Stage 2.

Converts candidate entities from Stage 1 into canonical nodes by:
1. Collapsing within-document duplicates
2. Merging cross-document duplicates (exact match, fuzzy match, rules)
3. Returning deterministic canonical nodes and a candidate → canonical mapping

The algorithm is deterministic: same input always produces identical output,
independent of input order.
"""

import logging
from datetime import datetime
from difflib import SequenceMatcher
from typing import Optional
from collections import defaultdict

from stages.extraction.schemas import CandidateEntity, CandidateRelation
from .schemas import CanonicalNode, CanonicalRelation, EntityResolutionResult, ResolutionMetadata

logger = logging.getLogger(__name__)


class EntityResolver:
    """
    Resolves candidate entities into canonical nodes.

    Usage:
        resolver = EntityResolver(fuzzy_threshold=0.85, rules=None)
        result = resolver.resolve(candidates)
        # result.canonical_entities: dict[str, CanonicalNode]
        # result.candidate_to_canonical: dict[str, str] (mapping)
    """

    def __init__(
        self,
        fuzzy_threshold: float = 0.85,
        rules: Optional[dict] = None,
        auto_merge_fuzzy: bool = False,
    ):
        """
        Initialize the resolver.

        Args:
            fuzzy_threshold: Edit distance ratio for fuzzy matching (0-1, default 0.85)
            rules: Dict mapping (entity_type, normalized_name) → canonical_name
                   or full rule definitions. See _resolve_via_rules().
            auto_merge_fuzzy: If True, auto-merge fuzzy candidates; else flag for review
        """
        self.fuzzy_threshold = fuzzy_threshold
        self.rules = rules or {}
        self.auto_merge_fuzzy = auto_merge_fuzzy
        self.metadata = ResolutionMetadata()

    def resolve(
        self,
        candidates: list[CandidateEntity],
        document_id: str,
    ) -> EntityResolutionResult:
        """
        Resolve a list of candidate entities into canonical nodes.

        Args:
            candidates: List of CandidateEntity from Stage 1
            document_id: Document ID for audit trail

        Returns:
            EntityResolutionResult with canonical nodes and mapping
        """
        self.metadata = ResolutionMetadata(total_candidates=len(candidates))
        result = EntityResolutionResult(metadata=self.metadata)

        if not candidates:
            return result

        # Step 1: Within-document deduplication
        # Collapse candidates with identical (entity_type, canonical_name) within this document
        deduped = self._dedupe_within_document(candidates)
        self.metadata.within_doc_merges = len(candidates) - len(deduped)
        logger.info(f"Within-document deduplication: {len(candidates)} → {len(deduped)}")

        # Step 2: Cross-document resolution
        # Map each deduplicated candidate to a canonical node ID
        for candidate in deduped:
            canonical_id = self._resolve_candidate(candidate, result)
            result.candidate_to_canonical[candidate.id] = canonical_id

        # Step 3: Assemble canonical nodes from mapping
        # Collect all sources and surface forms for each canonical node
        canonical_by_id = {}
        for candidate, canonical_id in zip(deduped, [result.candidate_to_canonical[c.id] for c in deduped]):
            if canonical_id not in canonical_by_id:
                # Create new canonical node
                canonical_by_id[canonical_id] = self._create_canonical_node(
                    canonical_id=canonical_id,
                    entity_type=candidate.entity_type,
                    canonical_name=candidate.canonical_name,
                )
            # Merge candidate data into canonical node
            node = canonical_by_id[canonical_id]
            self._merge_candidate_into_canonical(node, candidate, document_id)

        result.canonical_entities = canonical_by_id
        self.metadata.total_canonical = len(canonical_by_id)
        self.metadata.total_surface_forms = sum(
            len(n.surface_forms) for n in canonical_by_id.values()
        )

        logger.info(
            f"Resolution complete: {len(deduped)} candidates → "
            f"{len(canonical_by_id)} canonical nodes"
        )
        return result

    def resolve_relations(
        self,
        relations: list[CandidateRelation],
        candidate_to_canonical: dict[str, str],
    ) -> dict[str, CanonicalRelation]:
        """
        Resolve candidate relations into canonical relations.

        Maps relation source/target from candidate IDs to canonical node IDs,
        then collapses multiple relations between the same canonical pair.

        Args:
            relations: List of CandidateRelation from Stage 1
            candidate_to_canonical: Mapping from resolve()

        Returns:
            Dict mapping (source_id:target_id:relation_type) → CanonicalRelation
        """
        canonical_relations = {}

        for rel in relations:
            # Map candidate IDs to canonical IDs
            canonical_source = candidate_to_canonical.get(rel.source_entity_id)
            canonical_target = candidate_to_canonical.get(rel.target_entity_id)

            if not canonical_source or not canonical_target:
                # Dangling reference (should not happen with valid Stage 1 output)
                logger.warning(
                    f"Dangling relation: source={rel.source_entity_id}, "
                    f"target={rel.target_entity_id}"
                )
                self.metadata.errors.append({
                    "type": "dangling_relation",
                    "source_id": rel.source_entity_id,
                    "target_id": rel.target_entity_id,
                })
                continue

            # Create canonical relation key
            rel_key = f"{canonical_source}:{canonical_target}:{rel.relation_type}"

            if rel_key not in canonical_relations:
                # New relation
                canonical_relations[rel_key] = CanonicalRelation(
                    source_id=canonical_source,
                    target_id=canonical_target,
                    relation_type=rel.relation_type,
                    description=rel.description,
                    evidence=[rel.evidence] if rel.evidence else [],
                    supporting_chunks=list(rel.chunk_ids) if rel.chunk_ids else [],
                    relation_count=1,
                    confidence=self._compute_confidence(rel),
                )
            else:
                # Merge into existing relation
                existing = canonical_relations[rel_key]
                existing.relation_count += 1
                if rel.evidence and rel.evidence not in existing.evidence:
                    existing.evidence.append(rel.evidence)
                for chunk_id in rel.chunk_ids:
                    if chunk_id not in existing.supporting_chunks:
                        existing.supporting_chunks.append(chunk_id)
                # Recalculate confidence
                existing.confidence = self._compute_confidence_from_relation(existing)
                existing.updated = datetime.utcnow()

        logger.info(f"Relations resolved: {len(relations)} → {len(canonical_relations)}")
        return canonical_relations

    # ─────────────────────────────────────────────────────────────────────────────
    # Private methods
    # ─────────────────────────────────────────────────────────────────────────────

    def _dedupe_within_document(self, candidates: list[CandidateEntity]) -> list[CandidateEntity]:
        """
        Within-document deduplication: collapse candidates with identical
        (entity_type, canonical_name).

        Args:
            candidates: List of candidates

        Returns:
            Deduplicated list (merged candidates retain all chunk IDs and surface forms)
        """
        # Group by (entity_type, canonical_name)
        groups = defaultdict(list)
        for candidate in candidates:
            key = (candidate.entity_type, candidate.canonical_name)
            groups[key].append(candidate)

        deduped = []
        for group in groups.values():
            if len(group) == 1:
                deduped.append(group[0])
            else:
                # Merge multiple candidates with same type+name
                merged = self._merge_candidates(group)
                deduped.append(merged)

        return deduped

    def _merge_candidates(self, candidates: list[CandidateEntity]) -> CandidateEntity:
        """
        Merge multiple candidates into one by combining chunk references and surface forms.

        Returns a new CandidateEntity with merged data.
        """
        first = candidates[0]
        all_chunks = []
        all_surface_forms = set()

        for candidate in candidates:
            all_chunks.extend(candidate.chunk_ids)
            all_surface_forms.add(candidate.surface_form)
            all_surface_forms.add(candidate.canonical_name)

        # Deduplicate chunks while preserving order
        deduped_chunks = []
        seen = set()
        for chunk_id in all_chunks:
            if chunk_id not in seen:
                deduped_chunks.append(chunk_id)
                seen.add(chunk_id)

        # Keep the first description; could average or concatenate if needed
        return CandidateEntity(
            id=first.id,
            entity_type=first.entity_type,
            canonical_name=first.canonical_name,
            surface_form=first.surface_form,
            description=first.description,
            chunk_ids=deduped_chunks,
        )

    def _resolve_candidate(
        self,
        candidate: CandidateEntity,
        result: EntityResolutionResult,
    ) -> str:
        """
        Resolve a single candidate into a canonical node ID.

        Tries resolution strategies in order:
        1. Rule-based: check domain rules
        2. Exact match: find existing canonical with same (type, canonical_name)
        3. Fuzzy match: find existing canonical with similar canonical_name
        4. Create new: no match found

        Returns the canonical node ID.
        """
        # Try rule-based resolution
        rule_canonical_id = self._resolve_via_rules(candidate)
        if rule_canonical_id:
            self.metadata.rule_based_merges += 1
            return rule_canonical_id

        # Try exact match in existing canonicals
        for canonical in result.canonical_entities.values():
            if (
                canonical.entity_type == candidate.entity_type
                and canonical.canonical_name == candidate.canonical_name
            ):
                self.metadata.exact_match_merges += 1
                return canonical.id

        # Try fuzzy match in existing canonicals
        fuzzy_match = self._find_fuzzy_match(candidate, result.canonical_entities.values())
        if fuzzy_match:
            if self.auto_merge_fuzzy:
                self.metadata.fuzzy_match_candidates += 1
                return fuzzy_match.id
            else:
                # Flag for review but don't merge
                self.metadata.fuzzy_match_candidates += 1
                logger.info(
                    f"Fuzzy match candidate: {candidate.canonical_name} "
                    f"≈ {fuzzy_match.canonical_name}"
                )

        # No match: create new canonical node
        canonical_id = self._generate_canonical_id(candidate)
        return canonical_id

    def _resolve_via_rules(self, candidate: CandidateEntity) -> Optional[str]:
        """
        Check if a domain rule applies to this candidate.

        Rules are typically a dict mapping (entity_type, normalized_name) → canonical_id.
        Override this method or pass rules to __init__ for custom behavior.

        Returns the canonical node ID if a rule matches, else None.
        """
        if not self.rules:
            return None

        # Normalize the candidate name for rule matching
        normalized = candidate.canonical_name.lower().strip()

        # Simple rule format: rules[(entity_type, normalized_name)] = canonical_id
        rule_key = (candidate.entity_type, normalized)
        if rule_key in self.rules:
            return self.rules[rule_key]

        return None

    def _find_fuzzy_match(
        self,
        candidate: CandidateEntity,
        existing_canonicals,
    ) -> Optional[CanonicalNode]:
        """
        Find a fuzzy match for a candidate among existing canonical nodes.

        Compares canonical_name using SequenceMatcher.ratio() against the
        fuzzy_threshold. Returns the best match (if above threshold) or None.

        Only considers nodes of the same entity_type.
        """
        best_match = None
        best_ratio = self.fuzzy_threshold

        for canonical in existing_canonicals:
            if canonical.entity_type != candidate.entity_type:
                continue

            ratio = SequenceMatcher(
                None,
                candidate.canonical_name.lower(),
                canonical.canonical_name.lower(),
            ).ratio()

            if ratio > best_ratio:
                best_ratio = ratio
                best_match = canonical

        return best_match

    def _generate_canonical_id(self, candidate: CandidateEntity) -> str:
        """
        Generate a deterministic canonical node ID from entity type and name.

        Format: entity_type:canonical_name (both lowercase, normalized)
        """
        normalized_name = candidate.canonical_name.lower().strip()
        return f"{candidate.entity_type.lower()}:{normalized_name}"

    def _create_canonical_node(
        self,
        canonical_id: str,
        entity_type: str,
        canonical_name: str,
    ) -> CanonicalNode:
        """Create a new CanonicalNode with the given properties."""
        return CanonicalNode(
            id=canonical_id,
            entity_type=entity_type,
            canonical_name=canonical_name,
            surface_forms=[canonical_name],
            sources=[],
            candidate_ids=[],
        )

    def _merge_candidate_into_canonical(
        self,
        canonical: CanonicalNode,
        candidate: CandidateEntity,
        document_id: str,
    ) -> None:
        """
        Merge a candidate's data into an existing canonical node.

        Updates surface forms, sources, candidate ID tracking, and chunk IDs.
        """
        # Add surface form if new
        if candidate.surface_form not in canonical.surface_forms:
            canonical.surface_forms.append(candidate.surface_form)

        # Add source document if new
        if document_id not in canonical.sources:
            canonical.sources.append(document_id)

        # Track candidate ID for tracing
        if candidate.id not in canonical.candidate_ids:
            canonical.candidate_ids.append(candidate.id)

        # Preserve chunk IDs from candidate (for MENTIONED_IN relationships)
        for chunk_id in candidate.chunk_ids:
            if chunk_id not in canonical.chunk_ids:
                canonical.chunk_ids.append(chunk_id)

        # Update timestamp
        canonical.updated = datetime.utcnow()

    def _compute_confidence(self, relation: CandidateRelation) -> float:
        """
        Compute initial confidence for a newly-created relation.

        Uses breadth of support: len(supporting_chunks) / entity mention count.
        For now, set to 0.5 as default; this will be recalculated during aggregation.
        """
        return 0.5

    def _compute_confidence_from_relation(self, relation: CanonicalRelation) -> float:
        """
        Recompute confidence for an aggregated relation.

        Confidence is the ratio of distinct supporting chunks to total entity mentions.
        This is a placeholder; refine based on domain knowledge.

        Current formula: (num distinct supporting chunks) / (num mentions + 1)
        Range: [0, 1]
        """
        if not relation.supporting_chunks:
            return 0.0
        # Simple heuristic: higher count = higher confidence
        # Clamp to [0, 1]
        confidence = min(len(relation.supporting_chunks) / (relation.relation_count + 1), 1.0)
        return confidence
