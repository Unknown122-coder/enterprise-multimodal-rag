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

# Pre-download every model weight and vocabulary into the image layer, so a cold
# start never touches the network. The tiktoken vocabulary matters as much as the
# ONNX models: fetching it at runtime costs about a minute on a free-tier host,
# and failing to fetch it silently changes chunk boundaries and therefore chunk
# ids. See the module docstring in emrag/core/tokens.py.
#
# This runs before the source is copied on purpose: the weights depend only on
# requirements.txt, so a code change must not re-download 500 MB of models
# (and fail the build when the hub hiccups, as it did on the second deploy).
ENV FASTEMBED_CACHE_PATH=/opt/models \
    TIKTOKEN_CACHE_DIR=/opt/tiktoken
RUN mkdir -p /opt/models /opt/tiktoken && python - <<'PY'
import tiktoken
from fastembed import SparseTextEmbedding, TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder

TextEmbedding("BAAI/bge-base-en-v1.5", cache_dir="/opt/models")
SparseTextEmbedding("Qdrant/bm25", cache_dir="/opt/models")
TextCrossEncoder("Xenova/ms-marco-MiniLM-L-6-v2", cache_dir="/opt/models")
assert tiktoken.get_encoding("cl100k_base").encode("warm up"), "tokeniser not cached"
print("models and tokeniser cached")
PY
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps .

FROM python:3.11-slim AS runtime
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    FASTEMBED_CACHE_PATH=/app/models \
    TIKTOKEN_CACHE_DIR=/app/.cache/tiktoken \
    HF_HOME=/app/.cache/hf \
    PORT=7860
RUN apt-get update && apt-get install -y --no-install-recommends curl libgomp1 && \
    rm -rf /var/lib/apt/lists/* && \
    useradd -m -u 1000 emrag
COPY --from=builder /opt/venv /opt/venv
# Ownership is set during the COPY. A later `chown -R /app` would rewrite every
# file's metadata and so duplicate the whole 300 MB model directory into a new
# layer, which cost more than the models themselves.
COPY --from=builder --chown=emrag:emrag /opt/models /app/models
COPY --from=builder --chown=emrag:emrag /opt/tiktoken /app/.cache/tiktoken
WORKDIR /app
COPY --chown=emrag:emrag configs ./configs
COPY --chown=emrag:emrag scripts ./scripts
# Only these two are created here, and the chown is not recursive.
RUN mkdir -p /app/var /app/.cache/hf && chown emrag:emrag /app /app/var /app/.cache/hf
USER emrag
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${PORT}/healthz" || exit 1
CMD ["sh", "-c", "uvicorn emrag.api.main:app --host 0.0.0.0 --port ${PORT} --workers 1 --timeout-keep-alive 75"]
