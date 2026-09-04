"""groundkit — бесплатные LLM + веб-поиск с проверяемыми цитатами."""

from .answer import Answer, Answerer, verify_citations
from .images import (
    CloudflareImages,
    ImageResult,
    Pollinations,
    generate_image,
)
from .llm import (
    ClaudeCLI,
    GigaChat,
    LiteLLM,
    LLMResponse,
    OpenAICompat,
    complete_with_fallback,
    default_chain,
    list_models,
)
from .search import SearchResult, SearchRun, run_search, search_with_fallback

__version__ = "0.1.1"
__all__ = [
    "Answerer",
    "Answer",
    "verify_citations",
    "SearchResult",
    "SearchRun",
    "run_search",
    "search_with_fallback",
    "LiteLLM",
    "OpenAICompat",
    "GigaChat",
    "ClaudeCLI",
    "LLMResponse",
    "complete_with_fallback",
    "default_chain",
    "list_models",
    "generate_image",
    "ImageResult",
    "Pollinations",
    "CloudflareImages",
]
