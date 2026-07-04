"""
EnterpriseRAGPipeline — the central orchestrator.

Initializes all resources (LLM, ChromaDB, BM25 index, Tavily, user namespace
manager) and assembles the 13-node LangGraph state machine with conditional
routing for auth failures, validation errors, and safety blocks.
"""

from __future__ import annotations

import functools
import pickle
import time
import uuid
from pathlib import Path
from typing import AsyncIterator, Optional

from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

from app.config import get_settings
from app.memory.database import init_db
from app.observability.logger import logger
from app.observability.events import create_bus, destroy_bus, read_events
from app.pipeline.state import PipelineState
from app.tools.web_search import WebSearchTool
from app.upload.namespace import UserDocNamespace
from app.core.embeddings import get_embeddings

# Node imports
from app.pipeline.nodes.auth import auth_node
from app.pipeline.nodes.validator import validator_node
from app.pipeline.nodes.classifier import classifier_node
from app.pipeline.nodes.rewriter import rewriter_node
from app.pipeline.nodes.retriever import retriever_node
from app.pipeline.nodes.reranker import reranker_node
from app.pipeline.nodes.compressor import compressor_node
from app.pipeline.nodes.hallucination import hallucination_node
from app.pipeline.nodes.generator import generator_node
from app.pipeline.nodes.citation import citation_node
from app.pipeline.nodes.guardrails import guardrails_node
from app.pipeline.nodes.observer import observer_node


class EnterpriseRAGPipeline:
    """
    13-stage LangGraph pipeline for enterprise Python Q&A.

    Stages:
      ① auth → ② validator → ③ classifier → ④ rewriter →
      ⑤ retriever → ⑥ reranker → ⑦ compressor → ⑧ hallucination →
      ⑨ generator → ⑩ citation → ⑪ guardrails → ⑫ observer
    """

    def __init__(self):
        self.settings = get_settings()
        self.graph = None

        logger.info("🚀 Initializing Enterprise RAG Pipeline...")

        # LLM
        self.llm = ChatGoogleGenerativeAI(
            model=self.settings.MODEL_NAME,
            google_api_key=self.settings.GOOGLE_API_KEY,
            temperature=0.3,
            convert_system_message_to_human=False,
        )

        # Embeddings + Vector store
        embeddings = get_embeddings()
        self.vectorstore = Chroma(
            persist_directory=self.settings.VECTORSTORE_PATH,
            embedding_function=embeddings,
            collection_name=self.settings.COLLECTION_NAME,
        )
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": self.settings.TOP_K, "fetch_k": self.settings.FETCH_K},
        )
        logger.info(f"✅ ChromaDB loaded — {self._doc_count():,} chunks")

        # BM25 index
        self.bm25_index = None
        self.bm25_corpus: list[dict] = []
        self._init_bm25()

        # External tools
        self.web_search = WebSearchTool()

        # User document namespace manager
        self.user_ns = UserDocNamespace()

    async def initialize(self) -> None:
        """Async post-init: create DB tables, compile graph."""
        await init_db()
        logger.info("✅ SQLite tables ready")
        self._build_graph()
        logger.info("✅ LangGraph pipeline compiled")

    # ── BM25 ─────────────────────────────────────────────────────────────

    def _init_bm25(self) -> None:
        bm25_path = Path(self.settings.BM25_INDEX_PATH)
        if bm25_path.exists():
            try:
                logger.info("Loading BM25 index from cache...")
                with open(bm25_path, "rb") as f:
                    data = pickle.load(f)
                self.bm25_index = data["index"]
                self.bm25_corpus = data["corpus"]
                logger.info(f"✅ BM25 loaded — {len(self.bm25_corpus):,} docs")
                return
            except Exception as exc:
                logger.warning(f"BM25 cache load failed ({exc}), rebuilding...")

        self._build_bm25()

    def _build_bm25(self) -> None:
        """Build BM25 index from ChromaDB corpus and persist to disk."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.warning("rank-bm25 not installed — BM25 retrieval disabled")
            return

        logger.info("Building BM25 index from ChromaDB (this may take ~30s for 50K docs)...")
        t0 = time.time()

        collection = self.vectorstore._collection
        results = collection.get(include=["documents", "metadatas"])

        self.bm25_corpus = [
            {"content": doc, "metadata": meta}
            for doc, meta in zip(results["documents"], results["metadatas"])
        ]

        tokenized = [doc["content"].lower().split() for doc in self.bm25_corpus]
        self.bm25_index = BM25Okapi(tokenized)

        # Persist
        bm25_path = Path(self.settings.BM25_INDEX_PATH)
        bm25_path.parent.mkdir(parents=True, exist_ok=True)
        with open(bm25_path, "wb") as f:
            pickle.dump({"index": self.bm25_index, "corpus": self.bm25_corpus}, f)

        elapsed = time.time() - t0
        logger.info(f"✅ BM25 index built — {len(self.bm25_corpus):,} docs in {elapsed:.1f}s")

    # ── Graph Assembly ────────────────────────────────────────────────────

    def _build_graph(self) -> None:
        """Assemble and compile the 13-node LangGraph state machine."""
        workflow = StateGraph(PipelineState)

        # Bind dependencies to nodes via functools.partial
        _llm = self.llm
        _retriever = self.retriever
        _bm25_idx = self.bm25_index
        _bm25_corpus = self.bm25_corpus
        _web = self.web_search
        _user_ns = self.user_ns

        # Register nodes
        workflow.add_node("auth", auth_node)
        workflow.add_node("validator", validator_node)
        workflow.add_node("classifier",
                          functools.partial(classifier_node, llm=_llm))
        workflow.add_node("rewriter",
                          functools.partial(rewriter_node, llm=_llm))
        workflow.add_node("retriever",
                          functools.partial(retriever_node,
                                           retriever=_retriever,
                                           bm25_index=_bm25_idx,
                                           bm25_corpus=_bm25_corpus,
                                           web_search_tool=_web,
                                           user_ns_manager=_user_ns))
        workflow.add_node("reranker", reranker_node)
        workflow.add_node("compressor", compressor_node)
        workflow.add_node("hallucination",
                          functools.partial(hallucination_node, llm=_llm))
        workflow.add_node("generator",
                          functools.partial(generator_node, llm=_llm, web_search_tool=_web))
        workflow.add_node("citation", citation_node)
        workflow.add_node("guardrails",
                          functools.partial(guardrails_node, llm=_llm))
        workflow.add_node("observer", observer_node)

        # Entry point
        workflow.set_entry_point("auth")

        # Conditional edges for early-exit on auth/validation failure
        workflow.add_conditional_edges("auth",
            lambda s: "validator" if s.get("is_authenticated") else "observer",
            {"validator": "validator", "observer": "observer"})

        workflow.add_conditional_edges("validator",
            lambda s: "classifier" if s.get("is_valid_query") else "observer",
            {"classifier": "classifier", "observer": "observer"})

        # Sequential main pipeline
        for src, dst in [
            ("classifier", "rewriter"),
            ("rewriter", "retriever"),
            ("retriever", "reranker"),
            ("reranker", "compressor"),
            ("compressor", "hallucination"),
            ("hallucination", "generator"),
            ("generator", "citation"),
            ("citation", "guardrails"),
            ("guardrails", "observer"),
            ("observer", END),
        ]:
            workflow.add_edge(src, dst)

        self.graph = workflow.compile()

    # ── Public API ────────────────────────────────────────────────────────

    async def ask(
        self,
        question: str,
        session_id: str,
        user_id: str = "anonymous",
        api_key: str = "",
        include_user_docs: bool = True,
        conversation_history: Optional[list[dict]] = None,
    ) -> dict:
        """Run the full pipeline and return the final state dict."""
        response_id = str(uuid.uuid4())

        # Create SSE bus (even for non-streaming calls, events go in queue but are drained)
        create_bus(response_id)

        initial_state: PipelineState = {
            "raw_query": question,
            "session_id": session_id,
            "user_id": user_id,
            "api_key": api_key,
            "include_user_docs": include_user_docs,
            "conversation_history": conversation_history or [],
            "pipeline_trace": [],
            "response_id": response_id,
            "pipeline_start_time": time.perf_counter(),
            # Defaults
            "is_authenticated": False,
            "is_valid_query": False,
            "cleaned_query": "",
            "intent": "rag",
            "rewritten_queries": [],
            "vector_docs": [],
            "bm25_docs": [],
            "web_docs": [],
            "user_doc_results": [],
            "retrieval_score": 0.0,
            "reranked_docs": [],
            "compressed_context": "",
            "can_answer": False,
            "generation": "",
            "citations": [],
            "confidence": "low",
            "is_safe": True,
            "safety_message": None,
            "error_type": None,
            "error_message": None,
        }

        final_state = await self.graph.ainvoke(initial_state)

        # Drain SSE bus (observer already closed it)
        destroy_bus(response_id)

        return final_state

    async def astream(
        self,
        question: str,
        session_id: str,
        user_id: str = "anonymous",
        api_key: str = "",
        include_user_docs: bool = True,
        conversation_history: Optional[list[dict]] = None,
    ) -> AsyncIterator[dict]:
        """
        Run pipeline in background; yield SSE events as they arrive.
        Yields dicts: {type, node, label, status, latency_ms, detail, ...}
        """
        import asyncio

        response_id = str(uuid.uuid4())
        q = create_bus(response_id)

        initial_state: PipelineState = {
            "raw_query": question,
            "session_id": session_id,
            "user_id": user_id,
            "api_key": api_key,
            "include_user_docs": include_user_docs,
            "conversation_history": conversation_history or [],
            "pipeline_trace": [],
            "response_id": response_id,
            "pipeline_start_time": time.perf_counter(),
            "is_authenticated": False,
            "is_valid_query": False,
            "cleaned_query": "",
            "intent": "rag",
            "rewritten_queries": [],
            "vector_docs": [],
            "bm25_docs": [],
            "web_docs": [],
            "user_doc_results": [],
            "retrieval_score": 0.0,
            "reranked_docs": [],
            "compressed_context": "",
            "can_answer": False,
            "generation": "",
            "citations": [],
            "confidence": "low",
            "is_safe": True,
            "safety_message": None,
            "error_type": None,
            "error_message": None,
        }

        # Run pipeline as background task
        task = asyncio.create_task(self.graph.ainvoke(initial_state))

        # Yield events as they arrive
        async for event in read_events(response_id):
            yield event

        # Ensure pipeline task completes
        try:
            await task
        except Exception as exc:
            logger.error(f"Pipeline task failed: {exc}")

        destroy_bus(response_id)

    def get_doc_count(self) -> int:
        return self._doc_count()

    def _doc_count(self) -> int:
        try:
            return self.vectorstore._collection.count()
        except Exception:
            return -1
