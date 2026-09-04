"""groundkit — бесплатные LLM + веб-поиск с проверяемыми цитатами."""

from .answer import Answer, Answerer, verify_citations
from .llm import ClaudeCLI, LiteLLM, LLMResponse, complete_with_fallback, default_chain, list_models
from .search import SearchResult, SearchRun, run_search, search_with_fallback

__version__ = "0.1.0"
__all__ = [
    "Answerer",
    "Answer",
    "verify_citations",
    "SearchResult",
    "SearchRun",
    "run_search",
    "search_with_fallback",
    "LiteLLM",
    "ClaudeCLI",
    "LLMResponse",
    "complete_with_fallback",
    "default_chain",
    "list_models",
]
