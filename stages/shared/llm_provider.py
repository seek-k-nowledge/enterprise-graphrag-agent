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

import json
import logging
import os
from typing import Optional, Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.runnables import Runnable, RunnableConfig

logger = logging.getLogger(__name__)


def _deserialize_stringified_lists(obj: Any) -> Any:
    """Recursively deserialize JSON-stringified list/dict fields.

    Some LLMs (like Claude) may return structured output with list fields
    as JSON-encoded strings instead of native lists. This helper parses them.
    """
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            if isinstance(v, str):
                try:
                    parsed = json.loads(v)
                    if isinstance(parsed, (list, dict)):
                        result[k] = _deserialize_stringified_lists(parsed)
                    else:
                        result[k] = v
                except (json.JSONDecodeError, ValueError):
                    result[k] = v
            else:
                result[k] = _deserialize_stringified_lists(v)
        return result
    elif isinstance(obj, list):
        return [_deserialize_stringified_lists(item) for item in obj]
    else:
        return obj


class StructuredOutputFallover(Runnable):
    """Wrapper Runnable for structured output with call-time fallover.

    Tries primary model's structured output first, falls back to fallback
    model's structured output on error, mirroring the same pattern as _generate.
    """

    def __init__(self, primary_runnable: Runnable, fallback_runnable: Runnable, llm_wrapper: 'FalloverLLM'):
        self.primary_runnable = primary_runnable
        self.fallback_runnable = fallback_runnable
        self.llm_wrapper = llm_wrapper

    def invoke(self, input: Any, config: Optional[RunnableConfig] = None) -> Any:
        try:
            result = self.primary_runnable.invoke(input, config)
            self.llm_wrapper.last_provider = "primary"
            logger.debug("Structured output call served by: primary")
            return result
        except Exception as e:
            if "list_type" in str(e) or "stringified" in str(e).lower():
                logger.info(f"Detected stringified field issue: {e}. Retrying with deserialization...")
                try:
                    raw_dict = self.primary_runnable.invoke(input, config)
                    if isinstance(raw_dict, dict):
                        fixed_dict = _deserialize_stringified_lists(raw_dict)
                        logger.info("Primary structured output recovered after field deserialization")
                        self.llm_wrapper.last_provider = "primary"
                        return fixed_dict
                except Exception as e2:
                    logger.warning(f"Deserialization failed: {e2}. Falling back to fallback model")
            else:
                logger.warning(
                    f"Primary structured output failed: {e}. "
                    f"Falling back to fallback model"
                )
            result = self.fallback_runnable.invoke(input, config)
            self.llm_wrapper.last_provider = "fallback"
            logger.warning("Structured output call served by: fallback (fallover)")
            return result

    async def ainvoke(self, input: Any, config: Optional[RunnableConfig] = None) -> Any:
        try:
            result = await self.primary_runnable.ainvoke(input, config)
            self.llm_wrapper.last_provider = "primary"
            logger.debug("Async structured output call served by: primary")
            return result
        except Exception as e:
            logger.warning(
                f"Primary async structured output failed: {e}. "
                f"Falling back to fallback model"
            )
            result = await self.fallback_runnable.ainvoke(input, config)
            self.llm_wrapper.last_provider = "fallback"
            logger.warning("Async structured output call served by: fallback (fallover)")
            return result


class FalloverLLM(BaseChatModel):
    """
    Wrapper that adds call-time fallover to a primary LLM.

    If the primary LLM fails during invoke/generate (e.g., 402, 429),
    automatically retries with a fallback LLM instead.
    Tracks which provider served each call for logging.
    """

    def __init__(
        self,
        primary: BaseChatModel,
        fallback: BaseChatModel,
        model: str,
        temperature: float,
    ):
        super().__init__()
        self._primary = primary
        self._fallback = fallback
        self._model = model
        self._temperature = temperature
        self._last_provider = None  # Track which provider served the last call

    @property
    def primary(self):
        return self._primary

    @property
    def fallback(self):
        return self._fallback

    @property
    def model(self):
        return self._model

    @property
    def temperature(self):
        return self._temperature

    @property
    def last_provider(self):
        return self._last_provider

    @last_provider.setter
    def last_provider(self, value):
        self._last_provider = value

    @property
    def _llm_type(self) -> str:
        return f"fallover_{self.primary._llm_type}"

    def _generate(self, messages: list[BaseMessage], **kwargs: Any) -> Any:
        """Try primary, fall back on error. Log which provider served the call."""
        try:
            result = self.primary._generate(messages, **kwargs)
            self.last_provider = "cerebras"
            logger.debug("LLM call served by: cerebras")
            return result
        except Exception as e:
            logger.warning(
                f"Primary LLM failed during call: {e}. "
                f"Falling back to {self.fallback._llm_type}"
            )
            result = self.fallback._generate(messages, **kwargs)
            self.last_provider = "groq"
            logger.warning("LLM call served by: groq (fallback)")
            return result

    async def _agenerate(self, messages: list[BaseMessage], **kwargs: Any) -> Any:
        """Async version: try primary, fall back on error. Log which provider served the call."""
        try:
            result = await self.primary._agenerate(messages, **kwargs)
            self.last_provider = "cerebras"
            logger.debug("LLM call served by: cerebras")
            return result
        except Exception as e:
            logger.warning(
                f"Primary LLM failed during async call: {e}. "
                f"Falling back to {self.fallback._llm_type}"
            )
            result = await self.fallback._agenerate(messages, **kwargs)
            self.last_provider = "groq"
            logger.warning("LLM call served by: groq (fallback)")
            return result

    def with_structured_output(self, schema: Any, **kwargs: Any) -> StructuredOutputFallover:
        """Return a Runnable that supports structured output with call-time fallover.

        When the returned Runnable is invoked, it tries the primary model's
        structured output first and falls back to the fallback model if the
        primary fails (e.g., due to rate limits, API errors).

        Args:
            schema: JSON schema or Pydantic model for structured output
            **kwargs: Additional arguments passed to underlying models

        Returns:
            StructuredOutputFallover wrapper that implements call-time fallover
        """
        primary_runnable = self.primary.with_structured_output(schema, **kwargs)
        fallback_runnable = self.fallback.with_structured_output(schema, **kwargs)
        return StructuredOutputFallover(primary_runnable, fallback_runnable, self)


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
        "gpt-oss-120b": "claude-haiku-4-5-20251001",
        "openai/gpt-oss-120b": "claude-haiku-4-5-20251001",  # Translate generic names
        "claude-haiku-4-5-20251001": "claude-haiku-4-5-20251001",  # Native Claude ID
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
    api_key_override: Optional[str] = None,
) -> BaseChatModel:
    """
    Get an LLM from the primary (Cerebras) or fallback (Groq) provider.

    When provider=None, returns a FalloverLLM that wraps both providers
    and falls back at call-time if the primary fails.

    Provider selection order:
    - If provider argument is passed: use that (Anthropic, Cerebras, or Groq)
    - Elif DEFAULT_LLM_PROVIDER env var is set: use that as override
    - Else: default order (Cerebras → Groq)

    Args:
        model: Model ID (e.g., "gpt-oss-120b")
        temperature: Temperature for model sampling
        provider: Force a specific provider ("cerebras", "groq", "anthropic", or custom),
                  or None for automatic (uses DEFAULT_LLM_PROVIDER or defaults)
        api_key_override: API key to use instead of env var (takes precedence over env)

    Returns:
        Initialized BaseChatModel instance (possibly wrapped with call-time fallover)

    Raises:
        RuntimeError if all providers fail
    """
    errors = []

    # Use explicit provider, or fall back to DEFAULT_LLM_PROVIDER env var
    if provider is None:
        provider = os.getenv("DEFAULT_LLM_PROVIDER")
        if provider:
            logger.info(f"Using DEFAULT_LLM_PROVIDER={provider}")

    # Anthropic: explicit selection, but with Groq fallback for rate limits
    if provider == "anthropic":
        try:
            primary = _create_anthropic_client(model, temperature, api_key_override)
            logger.info(f"Using Anthropic provider for {model}")

            # Add Groq fallback in case of rate limits (429) or other failures
            try:
                fallback = _create_groq_client(model, temperature, None)
                llm = FalloverLLM(primary, fallback, model, temperature)
                logger.info("Anthropic + Groq fallback enabled (for rate limit handling)")
                return llm
            except Exception as groq_error:
                logger.warning(f"Groq fallback not available: {groq_error}. Using Anthropic only.")
                return primary
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
            primary = _create_cerebras_client(model, temperature, api_key_override)
            logger.info(f"Initialized Cerebras provider for call-time fallover")
        except Exception as e:
            error_msg = f"Cerebras initialization failed: {e}"
            logger.warning(error_msg)
            errors.append(error_msg)

        # Try fallback provider (Groq)
        try:
            fallback = _create_groq_client(model, temperature, api_key_override)
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
            llm = _create_cerebras_client(model, temperature, api_key_override)
            logger.info(f"Using Cerebras provider for {model}")
            return llm
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Cerebras provider: {e}")

    if provider == "groq":
        try:
            llm = _create_groq_client(model, temperature, api_key_override)
            logger.info(f"Using Groq provider for {model}")
            return llm
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Groq provider: {e}")

    # Unknown provider
    raise RuntimeError(f"Unknown provider: {provider}")


def _create_cerebras_client(model: str, temperature: float, api_key_override: Optional[str] = None) -> BaseChatModel:
    """
    Create a Cerebras LLM client (OpenAI-compatible API).

    Args:
        model: Generic model name (translated to Cerebras format)
        temperature: Temperature
        api_key_override: API key to use instead of env var (takes precedence)

    Returns:
        ChatOpenAI instance pointing to Cerebras API

    Raises:
        ImportError if langchain_openai not installed
        RuntimeError if CEREBRAS_API_KEY not set or model not found
    """
    api_key = api_key_override or os.getenv("CEREBRAS_API_KEY")
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


def _create_groq_client(model: str, temperature: float, api_key_override: Optional[str] = None) -> BaseChatModel:
    """
    Create a Groq LLM client.

    Args:
        model: Generic model name (translated to Groq format)
        temperature: Temperature
        api_key_override: API key to use instead of env var (takes precedence)

    Returns:
        ChatGroq instance

    Raises:
        ImportError if langchain_groq not installed
        RuntimeError if GROQ_API_KEY not set
    """
    api_key = api_key_override or os.getenv("GROQ_API_KEY")
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


def _create_anthropic_client(model: str, temperature: float, api_key_override: Optional[str] = None) -> BaseChatModel:
    """
    Create an Anthropic LLM client (Claude models).

    Args:
        model: Claude model ID (e.g., "claude-haiku-4-5-20251001")
        temperature: Temperature
        api_key_override: API key to use instead of env var (takes precedence)

    Returns:
        ChatAnthropic instance

    Raises:
        ImportError if langchain_anthropic not installed
        RuntimeError if ANTHROPIC_API_KEY not set
    """
    api_key = api_key_override or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set in environment")

    # Translate generic model name to Anthropic format
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
