FROM python:3.12-slim

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Non-root user ─────────────────────────────────────────────────────────────
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

# Create runtime directories and set ownership before switching user
RUN mkdir -p ./data ./logs ./uploads && chown -R user:user /home/user/app

# ── Python dependencies ───────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Switch to non-root ────────────────────────────────────────────────────────
USER user

# ── Pre-cache HuggingFace embedding model ─────────────────────────────────────
RUN python -c "from langchain_huggingface import HuggingFaceEmbeddings; HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')"

# ── Copy application code + pre-built vector store ────────────────────────────
COPY --chown=user app/ ./app/
COPY --chown=user vectorstore/ ./vectorstore/

# ── Port & Start ──────────────────────────────────────────────────────────────
EXPOSE 7860
ENV PORT=7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')" || exit 1

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
