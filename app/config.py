"""
Configuration module — loads settings from .env using Pydantic BaseSettings.
All new enterprise settings have safe defaults so existing deployments keep working.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings — loaded from environment variables / .env file."""

    # ── Core LLM / Embeddings ──────────────────────────────────────────────
    GOOGLE_API_KEY: str = ""
    MODEL_NAME: str = "gemini-2.0-flash-lite"

    # ── Vector Store ───────────────────────────────────────────────────────
    VECTORSTORE_PATH: str = "./vectorstore"
    COLLECTION_NAME: str = "python_qa"
    TOP_K: int = 6
    FETCH_K: int = 15
    CHUNK_SIZE: int = 1500
    CHUNK_OVERLAP: int = 200

    # ── BM25 (Hybrid Retrieval) ────────────────────────────────────────────
    BM25_INDEX_PATH: str = "./vectorstore/bm25_index.pkl"
    BM25_TOP_K: int = 6

    # ── Retrieval Quality ──────────────────────────────────────────────────
    RETRIEVAL_THRESHOLD: float = 0.4   # below this → trigger Tavily fallback

    # ── External Tools ─────────────────────────────────────────────────────
    TAVILY_API_KEY: str = ""           # empty = Tavily disabled

    # ── Authentication ─────────────────────────────────────────────────────
    API_KEY: str = ""                  # empty = auth disabled (dev mode)

    # ── Memory / SQLite ────────────────────────────────────────────────────
    DB_PATH: str = "./data/enterprise_rag.db"
    MAX_HISTORY_TURNS: int = 10        # conversation turns injected into prompts

    # ── Turso / Cloud SQLite ───────────────────────────────────────────────
    TURSO_DATABASE_URL: str = ""
    TURSO_AUTH_TOKEN: str = ""

    # ── Generation Latency Controls ───────────────────────────────────────
    GENERATION_MAX_OUTPUT_TOKENS: int = 700
    GENERATION_HISTORY_TURNS: int = 4
    GENERATION_CONTEXT_TOKEN_BUDGET: int = 2200

    # ── File Upload ────────────────────────────────────────────────────────
    UPLOAD_DIR: str = "./uploads"
    MAX_UPLOAD_SIZE_MB: int = 20
    USER_DOCS_COLLECTION_PREFIX: str = "user_docs_"

    # ── Observability ──────────────────────────────────────────────────────
    LOG_PATH: str = "./logs/pipeline.jsonl"

    # ── Rate Limiting ──────────────────────────────────────────────────────
    RATE_LIMIT: str = "30/minute"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Cached singleton — loaded once at first call, reused everywhere."""
    return Settings()
