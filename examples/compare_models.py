"""Прогон эталонного набора вопросов через несколько моделей.

Это тот самый скрипт сравнения, с которого стоит начинать проект:
сначала узнаём, какая модель работает на вашей предметной области,
потом строим вокруг неё код.
"""

import csv
import time
from pathlib import Path

from groundkit import Answerer

MODELS = [
    "gemini/gemini-flash-latest",
    "groq/llama-3.3-70b-versatile",
]

QUESTIONS = Path("questions.txt")   # по одному вопросу на строку
OUT = Path("results/comparison.csv")


def main() -> None:
    OUT.parent.mkdir(exist_ok=True)
    questions = [q.strip() for q in QUESTIONS.read_text(encoding="utf-8").splitlines() if q.strip()]

    with OUT.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["question", "model", "seconds", "dropped_links", "answer"])

        for question in questions:
            for model in MODELS:
                qa = Answerer(search="searxng", model=model)
                started = time.monotonic()
                try:
                    result = qa.ask(question)
                    elapsed = round(time.monotonic() - started, 2)
                    writer.writerow(
                        [question, model, elapsed, len(result.dropped_citations), result.answer]
                    )
                except Exception as exc:  # noqa: BLE001
                    writer.writerow([question, model, "", "", f"ОШИБКА: {exc}"])

    print(f"Готово: {OUT}")


if __name__ == "__main__":
    main()
