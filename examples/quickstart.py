"""Минимальный пример: вопрос по правовым источникам РФ."""

from groundkit import Answerer

qa = Answerer(
    allowed_domains=[
        "pravo.gov.ru",
        "publication.pravo.gov.ru",
        "sozd.duma.gov.ru",
        "regulation.gov.ru",
    ],
    search=["searxng", "exa", "brave"],   # по очереди, пока не сработает
    model=["gemini/gemini-flash-latest", "groq/llama-3.3-70b-versatile"],
)

result = qa.ask("Какие требования предъявляются к форме договора аренды недвижимости?")

print(result.answer)
print("\n--- Источники ---")
for s in result.sources:
    print(f"[{s.index}] {s.title}\n    {s.url}")

print(f"\nПоиск: {result.search_provider} | Модель: {result.model}")
if result.dropped_citations:
    print("Отброшено выдуманных ссылок:", result.dropped_citations)
if result.injection_flagged:
    print("⚠️ В источниках найден текст, похожий на инструкции модели")
