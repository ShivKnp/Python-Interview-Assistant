"""
User Document Namespace — manages per-user ChromaDB collections.

Each user gets their own isolated ChromaDB collection named `user_docs_{user_id}`.
Documents can be added, searched, or deleted independently per user.
"""

from __future__ import annotations

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.config import get_settings
from app.core.embeddings import get_embeddings
from app.observability.logger import logger


class UserDocNamespace:
    """
    Manages per-user ChromaDB collections for uploaded documents.
    Collections are created on demand and cached in memory.
    """

    def __init__(self):
        self.embeddings = get_embeddings()
        self.settings = get_settings()
        self._cache: dict[str, Chroma] = {}  # user_id → Chroma instance

    def _collection_name(self, user_id: str) -> str:
        safe_id = user_id.replace("-", "_")[:32]
        return f"{self.settings.USER_DOCS_COLLECTION_PREFIX}{safe_id}"

    def _get_or_create(self, user_id: str) -> Chroma:
        """Return cached Chroma instance for user, creating if needed."""
        if user_id not in self._cache:
            coll_name = self._collection_name(user_id)
            self._cache[user_id] = Chroma(
                persist_directory=self.settings.VECTORSTORE_PATH,
                embedding_function=self.embeddings,
                collection_name=coll_name,
            )
        return self._cache[user_id]

    def add_documents(self, user_id: str, documents: list[Document]) -> int:
        """
        Index documents into the user's namespace.
        Returns number of chunks added.
        """
        if not documents:
            return 0
        vs = self._get_or_create(user_id)
        vs.add_documents(documents)
        logger.info_data(
            "User docs indexed",
            user_id=user_id,
            chunks=len(documents),
            collection=self._collection_name(user_id),
        )
        return len(documents)

    def search(self, user_id: str, query: str, k: int = 4) -> list[Document]:
        """Search the user's document namespace. Returns empty list if no docs."""
        try:
            vs = self._get_or_create(user_id)
            # Check if collection has any documents
            count = vs._collection.count()
            if count == 0:
                return []
            retriever = vs.as_retriever(
                search_type="mmr",
                search_kwargs={"k": k, "fetch_k": k * 2},
            )
            return retriever.invoke(query)
        except Exception as exc:
            logger.warning(f"User doc search failed for {user_id}: {exc}")
            return []

    def delete_document(self, user_id: str, doc_id: str) -> int:
        """
        Remove all chunks belonging to a specific doc_id from the user's namespace.
        Returns number of chunks deleted.
        """
        try:
            vs = self._get_or_create(user_id)
            collection = vs._collection
            # Find all chunk IDs with this doc_id in metadata
            results = collection.get(where={"doc_id": doc_id})
            if not results["ids"]:
                return 0
            collection.delete(ids=results["ids"])
            logger.info_data("User doc deleted", user_id=user_id, doc_id=doc_id,
                             chunks_removed=len(results["ids"]))
            return len(results["ids"])
        except Exception as exc:
            logger.warning(f"User doc deletion failed for {user_id}/{doc_id}: {exc}")
            return 0

    def get_collection_name(self, user_id: str) -> str:
        return self._collection_name(user_id)

    def invalidate_cache(self, user_id: str) -> None:
        """Force re-load of user's Chroma instance."""
        self._cache.pop(user_id, None)
