FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
COPY pyproject.toml README.md ./
COPY groundkit ./groundkit
RUN pip install --no-cache-dir ".[web,ddg,spaces]"
# Сертификат Минцифры: без него не проверить TLS у GigaChat. Сборка не падает, если он недоступен.
ADD --chmod=444 https://gu-st.ru/content/Other/doc/russian_trusted_root_ca.cer /etc/ssl/certs/russian_trusted_root_ca.cer
ENV GIGACHAT_CA_BUNDLE=/etc/ssl/certs/russian_trusted_root_ca.cer
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import httpx; httpx.get('http://127.0.0.1:8000/api/health').raise_for_status()"
CMD ["uvicorn", "groundkit.web.app:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
