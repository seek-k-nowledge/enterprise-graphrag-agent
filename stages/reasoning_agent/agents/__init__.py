"""
LangGraph agent swarm for Stage 3: query routing, retrieval, synthesis, verification.

Agents:
- RouterAgent: Query classification and strategy selection (LLM-based)
- RetrievalAgent: Multi-hop Cypher authoring and traversal
- SynthesisAgent: Answer grounding and natural language synthesis
- VerificationAgent: Consistency checking and citation validation
"""

from .router_agent import RouterAgent
from .retrieval_agent import RetrievalAgent
from .synthesis_agent import SynthesisAgent
from .verification_agent import VerificationAgent

__all__ = [
    "RouterAgent",
    "RetrievalAgent",
    "SynthesisAgent",
    "VerificationAgent",
]
