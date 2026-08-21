"""Provider-agnostic LLM layer with automatic failover.

Implements intelligent provider selection:
- Primary: Cerebras API (OpenAI-compatible, gpt-oss-120b)
- Fallback: Groq (free tier, gpt-oss-120b)

Composes with existing retry/backoff and JSON fallback logic in extraction.
"""

import logging
import os
from typing import Optional

from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)


def get_llm(
    model: str = "gpt-oss-120b",
    temperature: float = 0.0,
    provider: Optional[str] = None,
) -> BaseChatModel:
    """
    Get an LLM from the primary (Cerebras) or fallback (Groq) provider.

    Args:
        model: Model ID (e.g., "gpt-oss-120b")
        temperature: Temperature for model sampling
        provider: Force a specific provider ("cerebras" or "groq"), or None for automatic

    Returns:
        Initialized BaseChatModel instance

    Raises:
        RuntimeError if all providers fail
    """
    errors = []

    # Try primary provider first (Cerebras)
    if provider is None or provider == "cerebras":
        try:
            llm = _create_cerebras_client(model, temperature)
            logger.info(f"Using Cerebras provider for {model}")
            return llm
        except Exception as e:
            error_msg = f"Cerebras initialization failed: {e}"
            logger.warning(error_msg)
            errors.append(error_msg)

    # Fall back to Groq
    if provider is None or provider == "groq":
        try:
            llm = _create_groq_client(model, temperature)
            logger.info(f"Using Groq provider (fallback) for {model}")
            return llm
        except Exception as e:
            error_msg = f"Groq initialization failed: {e}"
            logger.warning(error_msg)
            errors.append(error_msg)

    # All providers failed
    raise RuntimeError(
        f"Failed to initialize LLM. Tried providers: {provider or 'cerebras, groq'}. "
        f"Errors: {' | '.join(errors)}"
    )


def _create_cerebras_client(model: str, temperature: float) -> BaseChatModel:
    """
    Create a Cerebras LLM client (OpenAI-compatible API).

    Args:
        model: Model ID
        temperature: Temperature

    Returns:
        ChatOpenAI instance pointing to Cerebras API

    Raises:
        ImportError if langchain_openai not installed
        RuntimeError if CEREBRAS_API_KEY not set
    """
    api_key = os.getenv("CEREBRAS_API_KEY")
    if not api_key:
        raise RuntimeError("CEREBRAS_API_KEY not set in environment")

    try:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model_name=model,
            api_key=api_key,
            base_url="https://api.cerebras.ai/v1",
            temperature=temperature,
        )
    except ImportError:
        raise ImportError("langchain_openai required for Cerebras provider")


def _create_groq_client(model: str, temperature: float) -> BaseChatModel:
    """
    Create a Groq LLM client.

    Args:
        model: Model ID
        temperature: Temperature

    Returns:
        ChatGroq instance

    Raises:
        ImportError if langchain_groq not installed
        RuntimeError if GROQ_API_KEY not set
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set in environment")

    try:
        from langchain_groq import ChatGroq

        return ChatGroq(
            model_name=model,
            groq_api_key=api_key,
            temperature=temperature,
        )
    except ImportError:
        raise ImportError("langchain_groq required for Groq provider")
