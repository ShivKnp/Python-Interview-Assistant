"""
Pydantic v2 schemas — request/response models for all API endpoints.
"""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


# ─── Chat ────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """Body for POST /chat and POST /stream."""
    question: str = Field(..., min_length=3, max_length=2000)
    session_id: Optional[str] = Field(default=None, description="Omit to create a new session")
    user_id: str = Field(default="anonymous", description="User identifier for doc namespacing")
    include_user_docs: bool = Field(default=True, description="Also search uploaded documents")


class Citation(BaseModel):
    question_id: int
    title: str
    score: int
    source: str = Field(description="'knowledge_base' | 'user_doc' | 'web'")
    url: Optional[str] = None


class PipelineTraceEntry(BaseModel):
    node: str
    label: str
    status: str   # "success" | "warning" | "error" | "skipped"
    latency_ms: int
    detail: Optional[str] = None


class ChatResponse(BaseModel):
    """Full response returned by POST /chat."""
    question: str
    answer: str
    intent: str
    confidence: str
    citations: list[Citation]
    response_id: str
    session_id: str
    pipeline_trace: list[PipelineTraceEntry]
    response_time_ms: int


# ─── Sessions ────────────────────────────────────────────────────────────────

class SessionInfo(BaseModel):
    session_id: str
    user_id: str
    title: Optional[str]
    created_at: str
    last_active_at: str
    message_count: int


class MessageRecord(BaseModel):
    id: str
    session_id: str
    role: str
    content: str
    intent: Optional[str] = None
    confidence: Optional[str] = None
    citations: Optional[list[Citation]] = None
    pipeline_trace: Optional[list[PipelineTraceEntry]] = None
    created_at: str


class HistoryResponse(BaseModel):
    session_id: str
    messages: list[MessageRecord]


# ─── Feedback ────────────────────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    response_id: str
    session_id: str
    rating: int = Field(..., ge=-1, le=1, description="1 = helpful, -1 = not helpful")
    comment: Optional[str] = Field(default=None, max_length=500)


class FeedbackResponse(BaseModel):
    status: str
    feedback_id: str


# ─── Document Upload ─────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    file_type: str
    chunk_count: int
    status: str


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    file_type: str
    chunk_count: int
    file_size_bytes: int
    uploaded_at: str


class DocumentListResponse(BaseModel):
    user_id: str
    documents: list[DocumentInfo]


# ─── Health & Metrics ────────────────────────────────────────────────────────

class ComponentStatus(BaseModel):
    name: str
    status: str
    detail: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    components: list[ComponentStatus]
    uptime_seconds: float
    model: str
    vectorstore_documents: int
    bm25_indexed: bool


class MetricsResponse(BaseModel):
    uptime_seconds: float
    total_queries: int
    successful_queries: int
    failed_queries: int
    auth_failures: int
    validation_failures: int = 0
    safety_blocks: int
    hallucination_checks_failed: int
    tavily_fallbacks: int
    user_doc_hits: int
    feedback_count: int
    intent_distribution: dict[str, int]
    avg_pipeline_latency_ms: float
    node_avg_latency_ms: dict[str, float]
