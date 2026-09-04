import os

import pytest

from groundkit.search import SearchResult


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    """Тесты не должны зависеть от ключей на машине разработчика и не должны писать в ~/.groundkit."""
    from groundkit import usage

    monkeypatch.setenv("GROUNDKIT_USAGE_FILE", str(tmp_path / "usage.json"))
    usage._ledger = None
    for key in ["GEMINI_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY", "MISTRAL_API_KEY", "CEREBRAS_API_KEY",
                "ANTHROPIC_API_KEY", "EXA_API_KEY", "BRAVE_API_KEY", "JINA_API_KEY", "GROUNDKIT_CLAUDE_CLI",
                "CLOUDFLARE_API_KEY", "CLOUDFLARE_ACCOUNT_ID", "DASHSCOPE_API_KEY",
                "GROUNDKIT_ACCESS_TOKEN"]:
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def results():
    return [
        SearchResult(title="ГК РФ ст. 651", url="https://www.consultant.ru/document/cons_doc_LAW_9027/a1/",
                     snippet="Договор аренды здания заключается в письменной форме", published="2026-01-10", index=1),
        SearchResult(title="Публикация", url="http://publication.pravo.gov.ru/doc/0001?utm_source=x",
                     snippet="Официальное опубликование", index=2),
        SearchResult(title="Гарант", url="https://base.garant.ru/10164072/", snippet="Комментарий", index=3),
    ]


def live_enabled() -> bool:
    return os.getenv("GROUNDKIT_LIVE") == "1"
