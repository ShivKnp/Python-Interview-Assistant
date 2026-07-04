"""
Tavily web search tool — wraps TavilySearchResults as a LangChain Document
producer for use in the Hybrid Retrieval node when KB confidence is low.
"""

from __future__ import annotations

from typing import Optional

from langchain_core.documents import Document

from app.config import get_settings
from app.observability.logger import logger


def _make_tavily_client():
    """Lazy-init Tavily; returns None if API key is missing."""
    settings = get_settings()
    if not settings.TAVILY_API_KEY:
        logger.info("Tavily API key not configured — web search fallback disabled")
        return None
    try:
        import os
        os.environ["TAVILY_API_KEY"] = settings.TAVILY_API_KEY
        from langchain_community.tools.tavily_search import TavilySearchResults
        return TavilySearchResults(
            max_results=5,
            tavily_api_key=settings.TAVILY_API_KEY,
            search_depth="advanced",
            include_answer=True,
        )
    except ImportError:
        logger.warning("langchain-community not installed — Tavily unavailable")
        return None


class WebSearchTool:
    """Thin wrapper that returns results as LangChain Documents."""

    def __init__(self):
        self._client = _make_tavily_client()
        self.enabled = self._client is not None

    def search(self, query: str) -> list[Document]:
        """
        Execute a Tavily search and return results as Documents.
        Returns empty list if disabled or on error.
        """
        if not self.enabled:
            return []
        try:
            raw_results = self._client.invoke(query)
            docs = []
            for r in raw_results:
                content = r.get("content", "") or r.get("answer", "")
                if not content:
                    continue
                docs.append(
                    Document(
                        page_content=content,
                        metadata={
                            "title": r.get("title", "Web Result"),
                            "url": r.get("url", ""),
                            "source": "web",
                            "question_id": 0,
                            "q_score": 0,
                        },
                    )
                )
            logger.info_data("Tavily search completed", query=query, results=len(docs))
            return docs
        except Exception as exc:
            logger.error(f"Tavily search failed: {exc}")
            return []
