"""Provider-agnostic LLM layer with automatic failover.

Implements intelligent provider selection:
- Primary: Cerebras API (OpenAI-compatible, model: gpt-oss-120b)
- Fallback: Groq (free tier, model: openai/gpt-oss-120b)
- Optional: Anthropic (testing tier, model: claude-haiku-4-5-20251001) — explicit only

Model name translation:
- Input: "gpt-oss-120b" or similar generic model name
- Cerebras gets: "gpt-oss-120b" (no prefix)
- Groq gets: "openai/gpt-oss-120b" (with prefix)
- Anthropic gets: "claude-haiku-4-5-20251001" (native Claude model ID)

Fallover behavior:
- Initialization errors: caught immediately, tries next provider
- Call-time errors (402, 429, etc): caught during invoke/generate, retries fallback

Composes with existing retry/backoff and JSON fallback logic in extraction.
"""

import logging
import os
from typing import Optional, Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage

logger = logging.getLogger(__name__)


class FalloverLLM(BaseChatModel):
    """
    Wrapper that adds call-time fallover to a primary LLM.

    If the primary LLM fails during invoke/generate (e.g., 402, 429),
    automatically retries with a fallback LLM instead.
    """

    def __init__(
        self,
        primary: BaseChatModel,
        fallback: BaseChatModel,
        model: str,
        temperature: float,
    ):
        super().__init__()
        self.primary = primary
        self.fallback = fallback
        self.model = model
        self.temperature = temperature

    @property
    def _llm_type(self) -> str:
        return f"fallover_{self.primary._llm_type}"

    def _generate(self, messages: list[BaseMessage], **kwargs: Any) -> Any:
        """Try primary, fall back on error."""
        try:
            return self.primary._generate(messages, **kwargs)
        except Exception as e:
            logger.warning(
                f"Primary LLM failed during call: {e}. "
                f"Falling back to {self.fallback._llm_type}"
            )
            return self.fallback._generate(messages, **kwargs)

    async def _agenerate(self, messages: list[BaseMessage], **kwargs: Any) -> Any:
        """Async version: try primary, fall back on error."""
        try:
            return await self.primary._agenerate(messages, **kwargs)
        except Exception as e:
            logger.warning(
                f"Primary LLM failed during async call: {e}. "
                f"Falling back to {self.fallback._llm_type}"
            )
            return await self.fallback._agenerate(messages, **kwargs)


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
    "anthropic": {
        "claude-haiku-4-5-20251001": "claude-haiku-4-5-20251001",
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

    When provider=None, returns a FalloverLLM that wraps both providers
    and falls back at call-time if the primary fails.

    Args:
        model: Model ID (e.g., "gpt-oss-120b")
        temperature: Temperature for model sampling
        provider: Force a specific provider ("cerebras", "groq", or "anthropic"),
                  or None for automatic (Cerebras → Groq, Anthropic not in fallback chain)

    Returns:
        Initialized BaseChatModel instance (possibly wrapped with call-time fallover)

    Raises:
        RuntimeError if all providers fail
    """
    errors = []

    # Anthropic: explicit only, no fallback
    if provider == "anthropic":
        try:
            llm = _create_anthropic_client(model, temperature)
            logger.info(f"Using Anthropic provider for {model}")
            return llm
        except Exception as e:
            error_msg = f"Anthropic initialization failed: {e}"
            logger.warning(error_msg)
            raise RuntimeError(f"Failed to initialize Anthropic provider: {e}")

    # Automatic fallover (provider=None): wrap both in FalloverLLM
    if provider is None:
        primary = None
        fallback = None

        # Try primary provider first (Cerebras)
        try:
            primary = _create_cerebras_client(model, temperature)
            logger.info(f"Initialized Cerebras provider for call-time fallover")
        except Exception as e:
            error_msg = f"Cerebras initialization failed: {e}"
            logger.warning(error_msg)
            errors.append(error_msg)

        # Try fallback provider (Groq)
        try:
            fallback = _create_groq_client(model, temperature)
            logger.info(f"Initialized Groq provider for call-time fallover")
        except Exception as e:
            error_msg = f"Groq initialization failed: {e}"
            logger.warning(error_msg)
            errors.append(error_msg)

        # At least one provider must initialize
        if primary is None and fallback is None:
            raise RuntimeError(
                f"Failed to initialize any LLM provider (Cerebras, Groq). "
                f"Errors: {' | '.join(errors)}"
            )

        # If we have both, wrap in FalloverLLM
        if primary is not None and fallback is not None:
            llm = FalloverLLM(primary, fallback, model, temperature)
            logger.info("Using FalloverLLM with Cerebras→Groq chain")
            return llm

        # If only primary, use it directly (with warning)
        if primary is not None:
            logger.warning("Groq unavailable; using Cerebras only (no fallover)")
            return primary

        # If only fallback, use it directly (with warning)
        logger.warning("Cerebras unavailable; using Groq only (no fallover)")
        return fallback

    # Explicit provider selection (cerebras or groq): no fallover
    if provider == "cerebras":
        try:
            llm = _create_cerebras_client(model, temperature)
            logger.info(f"Using Cerebras provider for {model}")
            return llm
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Cerebras provider: {e}")

    if provider == "groq":
        try:
            llm = _create_groq_client(model, temperature)
            logger.info(f"Using Groq provider for {model}")
            return llm
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Groq provider: {e}")

    # Unknown provider
    raise RuntimeError(f"Unknown provider: {provider}")


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


def _create_anthropic_client(model: str, temperature: float) -> BaseChatModel:
    """
    Create an Anthropic LLM client (Claude models).

    Args:
        model: Claude model ID (e.g., "claude-haiku-4-5-20251001")
        temperature: Temperature

    Returns:
        ChatAnthropic instance

    Raises:
        ImportError if langchain_anthropic not installed
        RuntimeError if ANTHROPIC_API_KEY not set
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set in environment")

    # Anthropic model names are used as-is
    anthropic_model = _get_provider_model_name("anthropic", model)

    try:
        from langchain_anthropic import ChatAnthropic

        llm = ChatAnthropic(
            model=anthropic_model,
            api_key=api_key,
            temperature=temperature,
        )
        logger.info(f"Initialized Anthropic client with model {anthropic_model}")
        return llm
    except ImportError:
        raise ImportError("langchain_anthropic required for Anthropic provider")
    except Exception as e:
        logger.error(f"Anthropic initialization error: {e}")
        raise
