"""Командная строка: ``groundkit ask | search | providers | serve``."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _load_dotenv() -> None:
    """Подхватывает .env из текущей папки без внешних зависимостей."""
    path = Path(".env")
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.split(" #", 1)[0].strip().strip("'\"")
        if key.strip() and value and key.strip() not in os.environ:
            os.environ[key.strip()] = value


def _csv(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [v.strip() for v in value.split(",") if v.strip()]


def cmd_ask(args: argparse.Namespace) -> int:
    from .answer import Answerer

    qa = Answerer(
        allowed_domains=_csv(args.domains),
        search=_csv(args.search) or ["searxng", "ddg"],
        model=_csv(args.model),
        limit=args.limit,
        fetch_pages=not args.no_fetch,
        rewrite_query=args.rewrite,
    )
    result = qa.ask(args.question)
    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return 0 if result.ok else 1

    print(result.answer)
    if result.search_query != args.question:
        print(f"\n(запрос для поиска: {result.search_query})")
    print("\n--- Источники ---")
    for s in result.sources:
        mark = "✓" if s.index in result.cited else " "
        date = f" · {s.published}" if s.published else ""
        print(f"{mark} [{s.index}] {s.title}{date}\n      {s.url}")
    print(f"\nПоиск: {result.search_provider or '—'} | Модель: {result.model or '—'} | "
          f"{result.timing.get('total_s', 0)} с")
    if result.llm and result.llm.attempts and len(result.llm.attempts) > 1:
        print("Переключения моделей:", "; ".join(
            f"{a['model']} — {'ок' if a['ok'] else a['error'][:80]}" for a in result.llm.attempts))
    if result.dropped_citations:
        print("Отброшено выдуманных ссылок:", result.dropped_citations)
    if result.dropped_indices:
        print("Отброшено несуществующих номеров источников:", result.dropped_indices)
    if result.injection_flagged:
        print("⚠️ В источниках найден текст, похожий на инструкции модели (вырезан)")
    if result.search_errors:
        print("Ошибки поиска:", result.search_errors)
    return 0 if result.ok else 1


def cmd_search(args: argparse.Namespace) -> int:
    from .search import run_search

    run = run_search(args.query, _csv(args.search) or ["searxng", "ddg"], _csv(args.domains), args.limit)
    if args.json:
        print(json.dumps({"provider": run.provider, "errors": run.errors,
                          "results": [r.to_dict() for r in run.results]}, ensure_ascii=False, indent=2))
        return 0 if run.results else 1
    for r in run.results:
        date = f" · {r.published}" if r.published else ""
        print(f"[{r.index}] {r.title}{date}\n    {r.url}\n    {r.snippet[:200]}")
    print(f"\nПровайдер: {run.provider or '—'}", f"| Ошибки: {run.errors}" if run.errors else "")
    return 0 if run.results else 1


def cmd_providers(_: argparse.Namespace) -> int:
    from .llm import CLAUDE_CLI_ENV, list_models
    from .search import PROVIDER_INFO, provider_configured

    print("Поиск:")
    for name, info in PROVIDER_INFO.items():
        state = "✓ готов" if provider_configured(name) else f"✗ нужен {info['env']}"
        print(f"  {name:8} {info['label']:26} {info['free']:28} {state}")
    print("\nМодели (порядок = порядок fallback):")
    for m in list_models():
        how = f"{m['env']}" if m["env"] else f"{CLAUDE_CLI_ENV}=1 и `claude` в PATH"
        state = "✓ готова" if m["configured"] else f"✗ нужен {how}"
        print(f"  {m['model']:52} {m['free']:26} {state}")
    return 0


def cmd_image(args: argparse.Namespace) -> int:
    from .images import ImageError, generate_image

    try:
        width, height = (int(v) for v in args.size.lower().split("x", 1))
    except ValueError:
        print(f"Размер должен быть вида 1024x1024, а не {args.size!r}", file=sys.stderr)
        return 2
    try:
        image = generate_image(args.prompt, _csv(args.provider), size=(width, height), seed=args.seed)
    except ImageError as exc:
        print(f"Не получилось: {exc}", file=sys.stderr)
        return 1
    path = image.save(args.out)
    if args.json:
        print(json.dumps({**image.to_dict(), "path": str(path)}, ensure_ascii=False, indent=2))
    else:
        print(f"{path} — {image.width}×{image.height}, {len(image.data) // 1024} КБ, "
              f"{image.provider}/{image.model}, {image.latency_s:.1f} с")
    return 0


def cmd_usage(args: argparse.Namespace) -> int:
    from .llm import list_models
    from .usage import get_ledger

    ledger = get_ledger()
    rows = ledger.summary(list_models())
    if args.json:
        print(json.dumps({"models": rows, "days": ledger.days()}, ensure_ascii=False, indent=2))
        return 0
    print(f"Журнал: {ledger.path}\n")
    print(f"{'модель':52} {'сегодня':>8} {'осталось':>9}  {'сброс (UTC)':19}  статус")
    for r in rows:
        if not r.get("configured") and not r["used_today"]:
            continue
        star = "*" if r["remaining_source"] == "estimate" else ""
        remaining = "—" if r["remaining"] is None else f"{r['remaining']}{star}"
        reset = (r["resets_at"] or "")[11:16]
        if r["blocked_until"]:
            status = f"лимит до {r['blocked_until'][11:16]}"
        elif r.get("last_error") and not r["ok_today"]:
            status = "ошибка: " + r["last_error"][:40]
        else:
            status = "ok"
        print(f"{r['model']:52} {r['used_today']:>8} {remaining:>9}  {reset:19}  {status}")
    print("\n* — оценка по известному дневному лимиту, провайдер точных данных не прислал")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print("Установите веб-зависимости: pip install 'groundkit[web]'", file=sys.stderr)
        return 2
    uvicorn.run("groundkit.web.app:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    parser = argparse.ArgumentParser(prog="groundkit", description="Бесплатные LLM + веб-поиск с проверяемыми цитатами")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ask", help="задать вопрос с опорой на веб-источники")
    p.add_argument("question")
    p.add_argument("--domains", "-d", help="белый список доменов через запятую")
    p.add_argument("--search", "-s", help="провайдеры поиска через запятую (searxng,ddg,jina,exa,brave)")
    p.add_argument("--model", "-m", help="модели через запятую (gemini/…, groq/…, claude-cli)")
    p.add_argument("--limit", type=int, default=8)
    p.add_argument("--no-fetch", action="store_true", help="не скачивать страницы, только сниппеты")
    p.add_argument("--rewrite", "-r", action="store_true", help="переформулировать вопрос в поисковый запрос моделью")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("search", help="только поиск, без модели")
    p.add_argument("query")
    p.add_argument("--domains", "-d")
    p.add_argument("--search", "-s")
    p.add_argument("--limit", type=int, default=8)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("providers", help="что настроено, а что нет")
    p.set_defaults(func=cmd_providers)

    p = sub.add_parser("image", help="сгенерировать картинку бесплатным провайдером")
    p.add_argument("prompt")
    p.add_argument("--out", "-o", default="image", help="куда сохранить; расширение подставится само")
    p.add_argument("--provider", "-p", help="pollinations, cloudflare (через запятую — по очереди)")
    p.add_argument("--size", default="1024x1024")
    p.add_argument("--seed", type=int, help="одинаковый seed даёт одинаковую картинку")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_image)

    p = sub.add_parser("usage", help="лимиты: сколько использовано сегодня, сколько осталось, когда сброс")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_usage)

    p = sub.add_parser("serve", help="запустить веб-демо")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true")
    p.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
