"""Provider-agnostic LLM layer with automatic failover.

Implements intelligent provider selection:
- Primary: Cerebras API (OpenAI-compatible, model: gpt-oss-120b)
- Fallback: Groq (free tier, model: openai/gpt-oss-120b)

Model name translation:
- Input: "gpt-oss-120b" or similar generic model name
- Cerebras gets: "gpt-oss-120b" (no prefix)
- Groq gets: "openai/gpt-oss-120b" (with prefix)

Composes with existing retry/backoff and JSON fallback logic in extraction.
"""

import logging
import os
from typing import Optional

from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

# Model name mapping per provider
MODEL_NAMES = {
    "cerebras": {
        "gpt-oss-120b": "gpt-oss-120b",
        "openai/gpt-oss-120b": "gpt-oss-120b",  # Handle both formats
    },
    "groq": {
        "gpt-oss-120b": "openai/gpt-oss-120b",
        "openai/gpt-oss-120b": "openai/gpt-oss-120b",  # Already correct
    },
}


def _get_provider_model_name(provider: str, model: str) -> str:
    """
    Translate generic model name to provider-specific format.

    Args:
        provider: Provider name ("cerebras" or "groq")
        model: Generic model name

    Returns:
        Provider-specific model name
    """
    if provider in MODEL_NAMES and model in MODEL_NAMES[provider]:
        return MODEL_NAMES[provider][model]
    return model  # Fallback: return as-is if not mapped


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
        model: Generic model name (translated to Cerebras format)
        temperature: Temperature

    Returns:
        ChatOpenAI instance pointing to Cerebras API

    Raises:
        ImportError if langchain_openai not installed
        RuntimeError if CEREBRAS_API_KEY not set or model not found
    """
    api_key = os.getenv("CEREBRAS_API_KEY")
    if not api_key:
        raise RuntimeError("CEREBRAS_API_KEY not set in environment")

    # Translate generic model name to Cerebras format
    cerebras_model = _get_provider_model_name("cerebras", model)

    try:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model_name=cerebras_model,
            api_key=api_key,
            base_url="https://api.cerebras.ai/v1",
            temperature=temperature,
        )
        logger.info(f"Initialized Cerebras client with model {cerebras_model}")
        return llm
    except ImportError:
        raise ImportError("langchain_openai required for Cerebras provider")
    except Exception as e:
        # Re-raise to trigger fallback
        logger.error(f"Cerebras initialization error: {e}")
        raise


def _create_groq_client(model: str, temperature: float) -> BaseChatModel:
    """
    Create a Groq LLM client.

    Args:
        model: Generic model name (translated to Groq format)
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

    # Translate generic model name to Groq format
    groq_model = _get_provider_model_name("groq", model)

    try:
        from langchain_groq import ChatGroq

        llm = ChatGroq(
            model_name=groq_model,
            groq_api_key=api_key,
            temperature=temperature,
        )
        logger.info(f"Initialized Groq client with model {groq_model}")
        return llm
    except ImportError:
        raise ImportError("langchain_groq required for Groq provider")
    except Exception as e:
        logger.error(f"Groq initialization error: {e}")
        raise
