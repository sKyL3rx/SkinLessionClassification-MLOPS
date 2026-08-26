FROM ghcr.io/astral-sh/uv:0.11.29 AS uv_binary


FROM nvidia/cuda:13.0.3-cudnn-runtime-ubuntu24.04

COPY --from=uv_binary /uv /uvx /bin/

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive \
    UV_PYTHON=3.11 \
    UV_MANAGED_PYTHON=1 \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:${PATH}" \
    MODEL_BUNDLE_DIR="/app/model" \
    ONNX_PROVIDER="cuda" \
    MAX_UPLOAD_MB="10"

RUN apt-get update \
    && apt-get install -y \
        --no-install-recommends \
        ca-certificates \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd \
        --create-home \
        --uid 10001 \
        appuser

COPY pyproject.toml uv.lock README.md ./
COPY src ./src

RUN --mount=type=cache,target=/root/.cache/uv \
    uv python install 3.11

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync \
        --locked \
        --no-dev \
        --extra api \
        --no-editable

COPY artifacts/deployment/model /app/model

RUN chown -R appuser:appuser \
    /app \
    /opt/uv/python

USER appuser

EXPOSE 8000

HEALTHCHECK \
    --interval=30s \
    --timeout=5s \
    --start-period=45s \
    --retries=3 \
    CMD python -c \
        "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD [ \
    "uvicorn", \
    "lesion_ml.api.main:app", \
    "--host", \
    "0.0.0.0", \
    "--port", \
    "8000", \
    "--workers", \
    "1" \
]