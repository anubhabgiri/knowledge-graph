"""Centralized LLM manager for the knowledge-graph pipeline.

Supports three providers out of the box:

* ``gemini``  — Google Generative AI via ``langchain_google_genai``
* ``openai``  — OpenAI via ``langchain_openai``
* ``ollama``  — Local Ollama server via ``langchain_ollama``

Usage
-----
    from llm_manager import LLMManager, LLMProvider

    # Factory-style (recommended)
    llm = LLMManager.build(provider="gemini", model="gemini-2.5-flash")

    # Enum-style
    llm = LLMManager.build(provider=LLMProvider.OPENAI, model="gpt-4o-mini")

The returned object is a ``langchain_core.language_models.BaseChatModel`` so
it drops in anywhere a LangChain chat model is expected (``with_structured_output``,
chains, etc.).
"""
from __future__ import annotations

import logging
from enum import Enum
from typing import Union

from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    """Supported LLM provider identifiers."""

    GEMINI = "gemini"
    OPENAI = "openai"
    OLLAMA = "ollama"


# Default model names per provider
_DEFAULT_MODELS: dict[LLMProvider, str] = {
    LLMProvider.GEMINI: "gemini-2.5-flash",
    LLMProvider.OPENAI: "gpt-4o-mini",
    LLMProvider.OLLAMA: "llama3.2",
}


class LLMManager:
    """Factory for building LangChain chat models from a provider/model pair.

    All provider-specific imports are deferred so that uninstalled packages
    do not cause import errors at module load time — only when the relevant
    provider is actually requested.
    """

    @staticmethod
    def build(
        provider: Union[str, LLMProvider] = LLMProvider.GEMINI,
        model: str | None = None,
        max_retries: int = 2,
        **kwargs,
    ) -> BaseChatModel:
        """Build and return a chat model for the given *provider*.

        Parameters
        ----------
        provider:
            One of ``"gemini"``, ``"openai"``, or ``"ollama"`` (case-insensitive),
            or the corresponding :class:`LLMProvider` enum value.
        model:
            Model identifier string. When omitted the provider's default is used:

            * Gemini  → ``gemini-2.5-flash``
            * OpenAI  → ``gpt-4o-mini``
            * Ollama  → ``llama3.2``
        max_retries:
            Number of automatic retries on transient API errors.
            Not applicable for Ollama (ignored).
        **kwargs:
            Extra keyword arguments forwarded verbatim to the underlying
            LangChain chat-model constructor (e.g. ``temperature``,
            ``base_url`` for Ollama, ``api_key`` overrides, etc.).

        Returns
        -------
        BaseChatModel
            A ready-to-use LangChain chat model instance.

        Raises
        ------
        ValueError
            If *provider* is not one of the supported values.
        ImportError
            If the required ``langchain_*`` package for the chosen provider is
            not installed.
        """
        if isinstance(provider, str):
            try:
                provider = LLMProvider(provider.lower())
            except ValueError:
                supported = ", ".join(f'"{p.value}"' for p in LLMProvider)
                raise ValueError(
                    f"Unknown LLM provider '{provider}'. Supported: {supported}."
                ) from None

        resolved_model = model or _DEFAULT_MODELS[provider]
        logger.info("Building LLM | provider=%s  model=%s", provider.value, resolved_model)

        if provider is LLMProvider.GEMINI:
            return LLMManager._build_gemini(resolved_model, max_retries, **kwargs)
        elif provider is LLMProvider.OPENAI:
            return LLMManager._build_openai(resolved_model, max_retries, **kwargs)
        elif provider is LLMProvider.OLLAMA:
            return LLMManager._build_ollama(resolved_model, **kwargs)

        # Unreachable, but keeps type checkers happy
        raise ValueError(f"Unhandled provider: {provider}")  # pragma: no cover

    # ------------------------------------------------------------------
    # Provider-specific builders
    # ------------------------------------------------------------------

    @staticmethod
    def _build_gemini(model: str, max_retries: int, **kwargs) -> BaseChatModel:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:
            raise ImportError(
                "langchain_google_genai is not installed. "
                "Run: pip install langchain-google-genai"
            ) from exc
        return ChatGoogleGenerativeAI(model=model, max_retries=max_retries, **kwargs)

    @staticmethod
    def _build_openai(model: str, max_retries: int, **kwargs) -> BaseChatModel:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise ImportError(
                "langchain_openai is not installed. "
                "Run: pip install langchain-openai"
            ) from exc
        return ChatOpenAI(model=model, max_retries=max_retries, **kwargs)

    @staticmethod
    def _build_ollama(model: str, **kwargs) -> BaseChatModel:
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:
            raise ImportError(
                "langchain_ollama is not installed. "
                "Run: pip install langchain-ollama"
            ) from exc
        return ChatOllama(model=model, **kwargs)

