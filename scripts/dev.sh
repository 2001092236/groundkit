#!/usr/bin/env bash
# Локальный запуск демо с Claude CLI в роли модели (только для собственных экспериментов).
cd "$(dirname "$0")/.." || exit 1
export GROUNDKIT_CLAUDE_CLI=${GROUNDKIT_CLAUDE_CLI:-1}
export GROUNDKIT_SEARCH=${GROUNDKIT_SEARCH:-ddg,jina}
exec .venv/bin/python -m uvicorn groundkit.web.app:app --host 127.0.0.1 --port "${PORT:-8765}"
