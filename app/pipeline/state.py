"""
PipelineState — the single TypedDict that flows through all 13 LangGraph nodes.

Every field is Optional or has a default so partial pipeline runs (on auth/
validation failure) never KeyError downstream nodes.
"""

from __future__ import annotations

import time
from typing import Any, Optional, TypedDict

from langchain_core.documents import Document


class PipelineState(TypedDict, total=False):
    """Shared state flowing through the 13-stage enterprise pipeline."""

    # ── Input ────────────────────────────────────────────────────────────
    raw_query: str
    session_id: str
    user_id: str
    api_key: str
    include_user_docs: bool          # Search uploaded docs too?

    # ── Node ①: Auth ─────────────────────────────────────────────────────
    is_authenticated: bool

    # ── Node ②: Validation ───────────────────────────────────────────────
    is_valid_query: bool
    cleaned_query: str               # PII-scrubbed, trimmed version
    validation_error: Optional[str]

    # ── Node ③: Intent Classification ────────────────────────────────────
    intent: str                      # "rag" | "debug" | "codegen" | "concept"

    # ── Node ④: Query Rewriting ───────────────────────────────────────────
    rewritten_queries: list[str]     # 2-3 expanded query variants

    # ── Node ⑤: Hybrid Retrieval ──────────────────────────────────────────
    vector_docs: list[Document]
    bm25_docs: list[Document]
    web_docs: list[Document]         # Tavily results (web fallback)
    user_doc_results: list[Document] # From user-uploaded namespace
    retrieval_score: float           # Max semantic similarity score (0-1)

    # ── Node ⑥: Re-ranking ────────────────────────────────────────────────
    reranked_docs: list[Document]

    # ── Node ⑦: Context Compression ──────────────────────────────────────
    compressed_context: str          # Token-budget–trimmed context string

    # ── Node ⑧: Hallucination Pre-check ──────────────────────────────────
    can_answer: bool                 # Can the context ground an answer?

    # ── Node ⑨: Answer Generation ────────────────────────────────────────
    generation: str

    # ── Node ⑩: Citation Verification ────────────────────────────────────
    citations: list[dict]            # [{question_id, title, score, source}]

    # ── Node ⑪: Safety / Guardrails ──────────────────────────────────────
    is_safe: bool
    safety_message: Optional[str]    # Replacement message if unsafe

    # ── Pipeline Metadata (used by observer & UI trace panel) ─────────────
    pipeline_trace: list[dict]       # [{node, label, status, latency_ms, detail}]
    response_id: str                 # UUID — links response to feedback rows
    confidence: str                  # "high" | "medium" | "low"
    conversation_history: list[dict] # [{role, content}] from SQLite

    # ── Error Handling ────────────────────────────────────────────────────
    error_type: Optional[str]        # None | "auth_error" | "validation_error" | "safety_block" | "pipeline_error"
    error_message: Optional[str]

    # ── Timing ────────────────────────────────────────────────────────────
    pipeline_start_time: float       # time.time() at entry point
