# syntax=docker/dockerfile:1.7
# Multi-stage build: deps + model weights are baked in so cold start never hits the network.

FROM python:3.11-slim AS builder
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps .

# Pre-download ONNX model weights into the image layer.
ENV FASTEMBED_CACHE_PATH=/opt/models
RUN mkdir -p /opt/models && python - <<'PY'
from fastembed import TextEmbedding, SparseTextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
TextEmbedding("BAAI/bge-base-en-v1.5", cache_dir="/opt/models")
SparseTextEmbedding("Qdrant/bm25", cache_dir="/opt/models")
TextCrossEncoder("Xenova/ms-marco-MiniLM-L-6-v2", cache_dir="/opt/models")
print("models cached")
PY

FROM python:3.11-slim AS runtime
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    FASTEMBED_CACHE_PATH=/app/models \
    HF_HOME=/app/.cache/hf \
    PORT=7860
RUN apt-get update && apt-get install -y --no-install-recommends curl libgomp1 && \
    rm -rf /var/lib/apt/lists/* && \
    useradd -m -u 1000 emrag
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/models /app/models
WORKDIR /app
COPY --chown=emrag:emrag configs ./configs
COPY --chown=emrag:emrag scripts ./scripts
RUN mkdir -p /app/var /app/.cache/hf && chown -R emrag:emrag /app
USER emrag
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${PORT}/healthz" || exit 1
CMD ["sh", "-c", "uvicorn emrag.api.main:app --host 0.0.0.0 --port ${PORT} --workers 1 --timeout-keep-alive 75"]
