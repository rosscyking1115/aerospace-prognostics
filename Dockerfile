FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src ./src

RUN pip install --no-cache-dir uv \
    && uv sync --frozen --no-dev

RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && mkdir -p /models \
    && chown -R appuser:appuser /app /models

ENV PATH="/app/.venv/bin:${PATH}"
ENV AEROSPACE_PROGNOSTICS_HEALTHCHECK_URL="http://127.0.0.1:8000/health"
ENV AEROSPACE_PROGNOSTICS_HEALTHCHECK_TIMEOUT_SECONDS=2

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -m aerospace_prognostics.serving.healthcheck

CMD ["uvicorn", "aerospace_prognostics.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
