import os

import pytest

from groundkit.search import SearchResult


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path, request):
    """Тесты не должны зависеть от ключей на машине разработчика и не должны писать в ~/.groundkit.

    LiteLLM при импорте сам подтягивает ``.env`` из рабочей папки, поэтому импортируем
    его до очистки, а чистим по шаблону — чтобы новый провайдер не пришлось дописывать руками.
    Живым тестам (``-m live``) ключи, наоборот, нужны — их окружение не трогаем.
    """
    import litellm  # noqa: F401 — импорт с побочным эффектом: загружает .env

    from groundkit import usage

    if not request.node.get_closest_marker("live"):
        for key in list(os.environ):
            if key.endswith(("_API_KEY", "_AUTH_KEY", "_ACCOUNT_ID", "_CA_BUNDLE")) or key.startswith(
                ("GROUNDKIT_", "GIGACHAT_", "SEARXNG_")
            ):
                monkeypatch.delenv(key, raising=False)
    # Журнал лимитов — во временный файл, иначе тесты пишут в настоящий ~/.groundkit/usage.json.
    monkeypatch.setenv("GROUNDKIT_USAGE_FILE", str(tmp_path / "usage.json"))
    usage._ledger = None


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
