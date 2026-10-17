"""Builds the configured LLM provider. Adding a provider = one file + one entry here."""

from collections.abc import Callable
from functools import lru_cache

from app.core.config import Settings, get_settings
from app.llm.base import LLMError, LLMProvider
from app.llm.fake import FakeProvider
from app.llm.openai_provider import OpenAIProvider

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
    """Instantiate the provider named by `settings.llm_provider`."""
    try:
        factory = PROVIDERS[settings.llm_provider]
    except KeyError:
        known = ", ".join(sorted(PROVIDERS))
        raise LLMError(f"Unknown LLM_PROVIDER {settings.llm_provider!r} (known: {known})") from None
    return factory(settings)


@lru_cache
def get_llm_provider() -> LLMProvider:
    """FastAPI dependency: one provider (and HTTP connection pool) per process."""
    return build_provider(get_settings())
