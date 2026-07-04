"""
Shared embeddings module — single source of truth for HuggingFaceEmbeddings instance.
Eliminates duplicate model loading in rag_pipeline.py, graph.py, namespace.py, ingest.py.
"""

from functools import lru_cache
from langchain_huggingface import HuggingFaceEmbeddings


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Cached singleton — loads all-MiniLM-L6-v2 once per process."""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")