"""Builds the configured LLM provider. Adding a provider = one file + one entry here."""

from collections.abc import Callable
from functools import lru_cache

from app.core.config import Settings, get_settings
from app.llm.base import LLMError, LLMProvider
from app.llm.fake import FakeProvider
from app.llm.openai_provider import OpenAIProvider
from app.llm.resilience import ResilientProvider

ProviderFactory = Callable[[Settings], LLMProvider]

PROVIDERS: dict[str, ProviderFactory] = {
    "openai": lambda s: OpenAIProvider(
        api_key=s.llm_api_key,
        model=s.llm_chat_model,
        base_url=s.llm_base_url,
        timeout_s=s.llm_timeout_s,
        reasoning_field=s.llm_reasoning_field,
        max_retries=s.llm_max_retries,
    ),
    "fake": lambda _s: FakeProvider(),
}


def build_provider(settings: Settings) -> LLMProvider:
    """Instantiate the provider named by `settings.llm_provider`, with its safety net.

    The primary (and the optional fallback model) are wrapped in a circuit
    breaker unless LLM_BREAKER_FAILURE_THRESHOLD=0 and no fallback is set.
    """
    try:
        factory = PROVIDERS[settings.llm_provider]
    except KeyError:
        known = ", ".join(sorted(PROVIDERS))
        raise LLMError(f"Unknown LLM_PROVIDER {settings.llm_provider!r} (known: {known})") from None
    primary = factory(settings)
    fallback = _build_fallback(settings)
    if settings.llm_breaker_failure_threshold == 0 and fallback is None:
        return primary
    return ResilientProvider(
        primary,
        fallback,
        # With the breaker off, a huge threshold keeps it closed but still fails over.
        failure_threshold=settings.llm_breaker_failure_threshold or 2**31,
        reset_timeout_s=settings.llm_breaker_reset_s,
    )


def _build_fallback(settings: Settings) -> LLMProvider | None:
    if not settings.llm_fallback_model:
        return None
    return OpenAIProvider(
        api_key=settings.llm_fallback_api_key or settings.llm_api_key,
        model=settings.llm_fallback_model,
        base_url=settings.llm_fallback_base_url or settings.llm_base_url,
        timeout_s=settings.llm_timeout_s,
        reasoning_field=settings.llm_fallback_reasoning_field,
        max_retries=settings.llm_max_retries,
    )


@lru_cache
def get_llm_provider() -> LLMProvider:
    """FastAPI dependency: one provider (and HTTP connection pool) per process."""
    return build_provider(get_settings())
