"""
Enterprise Multi-Agent AI Assistant — FastAPI Application with User Authentication.

Endpoints:
  POST  /signup           — Register a new user
  POST  /login            — Login, create session token, set cookie
  POST  /logout           — Logout, invalidate session token, clear cookie
  POST  /chat             — Multi-turn chat (full pipeline, returns JSON)
  POST  /stream           — Multi-turn chat with SSE streaming pipeline events
  POST  /sessions         — Create a new session
  GET   /sessions         — List sessions for a user
  GET   /sessions/{id}/history — Full conversation history
  DELETE /sessions/{id}   — Delete a session
  POST  /feedback         — Submit thumbs up/down feedback
  POST  /upload           — Upload a document (Enterprise)
  GET   /documents        — List uploaded documents
  DELETE /documents/{id}  — Delete a document
  GET   /health           — Enhanced health check
  GET   /metrics          — Pipeline metrics
"""

import sys
import time
import uuid
import json
from contextlib import asynccontextmanager
from pathlib import Path
from datetime import datetime, timezone, timedelta
from pydantic import BaseModel

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from fastapi import FastAPI, HTTPException, UploadFile, File, Header, Form, Request, Cookie, Depends, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.config import get_settings
from app.schemas import (
    ChatRequest, ChatResponse, Citation, PipelineTraceEntry,
    SessionInfo, MessageRecord, HistoryResponse,
    FeedbackRequest, FeedbackResponse,
    UploadResponse, DocumentInfo, DocumentListResponse,
    HealthResponse, ComponentStatus, MetricsResponse,
)
from app.memory import session_store
from app.observability.logger import logger
from app.observability.metrics import metrics
from app.pipeline.graph import EnterpriseRAGPipeline
from app.utils.auth_utils import hash_password, verify_password
from app.memory.database import connect_db, close_db
import aiosqlite

# ─── Global state ─────────────────────────────────────────────────────────────
pipeline: EnterpriseRAGPipeline | None = None
start_time: float = 0.0


# ─── App Lifecycle ────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline, start_time
    start_time = time.time()
    logger.info("🚀 Starting Enterprise Multi-Agent AI Assistant...")
    pipeline = EnterpriseRAGPipeline()
    await pipeline.initialize()
    logger.info(f"✅ Ready — {pipeline.get_doc_count():,} chunks indexed")
    yield
    logger.info("👋 Shutting down gracefully...")
    await close_db()


# ─── Rate Limiter ─────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Enterprise Multi-Agent AI Assistant",
    description="13-stage enterprise RAG pipeline with hybrid retrieval, memory, and document upload.",
    version="2.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Helper: extract API key from header ──────────────────────────────────────
def _get_api_key(x_api_key: str | None) -> str:
    return (x_api_key or "").strip()


# ─── Helper: ensure pipeline ready ───────────────────────────────────────────
def _require_pipeline() -> EnterpriseRAGPipeline:
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    return pipeline


# ─── AUTHENTICATION DEPENDENCY & SCHEMAS ──────────────────────────────────────

class UserAuthSchema(BaseModel):
    username: str
    password: str


async def get_current_user(
    x_guest_user: str | None = Header(default=None),
    x_session_token: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
    session_token: str | None = Cookie(default=None)
) -> str:
    # 1. Guest Mode Bypass
    if x_guest_user and x_guest_user.startswith("guest-"):
        return x_guest_user

    # 2. Extract token from cookie, header, or Bearer auth
    token = session_token
    if not token and x_session_token:
        token = x_session_token
    if not token and authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ")[1]

    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    async with connect_db() as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT username, expires_at FROM user_sessions WHERE token=?",
            (token,)
        ) as cur:
            row = await cur.fetchone()

    if not row:
        raise HTTPException(status_code=401, detail="Invalid session token")

    # Expiry verification
    expires_at = datetime.fromisoformat(row["expires_at"])
    if expires_at < datetime.now(timezone.utc):
        raise HTTPException(status_code=401, detail="Session expired")

    return row["username"]


# ─── AUTH ENDPOINTS ───────────────────────────────────────────────────────────

@app.post("/signup", tags=["Auth"])
async def signup(body: UserAuthSchema, response: Response):
    username = body.username.strip()
    password = body.password

    # Basic validations
    if len(username) < 3 or len(username) > 20 or not username.isalnum():
        raise HTTPException(status_code=400, detail="Username must be alphanumeric and 3-20 characters long")
    if len(password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters long")

    async with connect_db() as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT 1 FROM users WHERE username = ?", (username,)) as cur:
            if await cur.fetchone():
                raise HTTPException(status_code=400, detail="Username already taken")

        pwd_hash = hash_password(password)
        await db.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (username, pwd_hash))

        # Create session
        token = str(uuid.uuid4())
        expires = datetime.now(timezone.utc) + timedelta(days=7)
        await db.execute(
            "INSERT INTO user_sessions (token, username, expires_at) VALUES (?, ?, ?)",
            (token, username, expires.isoformat())
        )
        await db.commit()

    response.set_cookie(key="session_token", value=token, httponly=True, max_age=3600*24*7, samesite="none", secure=True)
    return {"status": "registered", "username": username, "session_token": token}


@app.post("/login", tags=["Auth"])
async def login(body: UserAuthSchema, response: Response):
    username = body.username.strip()
    password = body.password

    async with connect_db() as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT password_hash FROM users WHERE username = ?", (username,)) as cur:
            row = await cur.fetchone()

    if not row or not verify_password(password, row["password_hash"]):
        raise HTTPException(status_code=400, detail="Invalid username or password")

    # Create session
    token = str(uuid.uuid4())
    expires = datetime.now(timezone.utc) + timedelta(days=7)
    
    async with connect_db() as db:
        await db.execute(
            "INSERT INTO user_sessions (token, username, expires_at) VALUES (?, ?, ?)",
            (token, username, expires.isoformat())
        )
        await db.commit()

    response.set_cookie(key="session_token", value=token, httponly=True, max_age=3600*24*7, samesite="none", secure=True)
    return {"status": "logged_in", "username": username, "session_token": token}


@app.post("/logout", tags=["Auth"])
async def logout(response: Response, session_token: str | None = Cookie(default=None)):
    if session_token:
        async with connect_db() as db:
            await db.execute("DELETE FROM user_sessions WHERE token = ?", (session_token,))
            await db.commit()
    response.delete_cookie("session_token", samesite="none", secure=True)
    return {"status": "logged_out"}


# ─── CHAT ENDPOINTS ───────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
@limiter.limit("30/minute")
async def chat(
    request: Request,
    body: ChatRequest,
    username: str = Depends(get_current_user),
    x_api_key: str | None = Header(default=None),
):
    """Multi-turn chat — runs the full 13-stage pipeline synchronously."""
    p = _require_pipeline()
    settings = get_settings()

    # Bind user_id to authenticated username
    body.user_id = username

    # Ensure session exists
    session_id = body.session_id
    if not session_id:
        session_id = await session_store.create_session(body.user_id)

    # Load conversation history
    history = await session_store.get_conversation_pairs(
        session_id, last_n_turns=settings.MAX_HISTORY_TURNS
    )

    t_start = time.time()
    final_state = await p.ask(
        question=body.question,
        session_id=session_id,
        user_id=body.user_id,
        api_key=_get_api_key(x_api_key),
        include_user_docs=body.include_user_docs,
        conversation_history=history,
    )
    elapsed_ms = int((time.time() - t_start) * 1000)

    error_type = final_state.get("error_type")
    if error_type == "auth_error":
        raise HTTPException(status_code=401, detail=final_state.get("error_message"))
    if error_type == "validation_error":
        raise HTTPException(status_code=422, detail=final_state.get("error_message"))

    citations = [
        Citation(
            question_id=c.get("question_id", 0),
            title=c.get("title", ""),
            score=c.get("score", 0),
            source=c.get("source", "knowledge_base"),
        )
        for c in final_state.get("citations", [])
    ]
    trace = [
        PipelineTraceEntry(
            node=t["node"],
            label=t["label"],
            status=t["status"],
            latency_ms=t["latency_ms"],
            detail=t.get("detail"),
        )
        for t in final_state.get("pipeline_trace", [])
    ]

    return ChatResponse(
        question=body.question,
        answer=final_state.get("generation", ""),
        intent=final_state.get("intent", "rag"),
        confidence=final_state.get("confidence", "low"),
        citations=citations,
        response_id=final_state.get("response_id", ""),
        session_id=session_id,
        pipeline_trace=trace,
        response_time_ms=elapsed_ms,
    )


@app.post("/stream", tags=["Chat"])
@limiter.limit("30/minute")
async def stream_chat(
    request: Request,
    body: ChatRequest,
    username: str = Depends(get_current_user),
    x_api_key: str | None = Header(default=None),
):
    """
    Multi-turn chat with SSE streaming — yields pipeline events in real time.
    Each event is a JSON object with type: node_start | node_done | complete | error.
    """
    p = _require_pipeline()
    settings = get_settings()

    # Bind user_id to authenticated username
    body.user_id = username

    session_id = body.session_id
    if not session_id:
        session_id = await session_store.create_session(body.user_id)

    history = await session_store.get_conversation_pairs(
        session_id, last_n_turns=settings.MAX_HISTORY_TURNS
    )

    async def event_generator():
        # First yield the session_id so client can store it
        yield {"data": json.dumps({"type": "session", "session_id": session_id})}

        async for event in p.astream(
            question=body.question,
            session_id=session_id,
            user_id=body.user_id,
            api_key=_get_api_key(x_api_key),
            include_user_docs=body.include_user_docs,
            conversation_history=history,
        ):
            yield {"data": json.dumps(event, default=str)}

    return EventSourceResponse(event_generator())


# ─── SESSION ENDPOINTS ────────────────────────────────────────────────────────

@app.post("/sessions", tags=["Sessions"])
async def create_session(username: str = Depends(get_current_user)):
    """Create a new conversation session."""
    session_id = await session_store.create_session(username)
    return {"session_id": session_id, "user_id": username}


@app.get("/sessions", tags=["Sessions"])
async def list_sessions(username: str = Depends(get_current_user)):
    """List all sessions for a user."""
    rows = await session_store.list_sessions(username)
    return {
        "user_id": username,
        "sessions": [
            SessionInfo(
                session_id=r["id"],
                user_id=r["user_id"],
                title=r.get("title"),
                created_at=r["created_at"],
                last_active_at=r["last_active_at"],
                message_count=r.get("message_count", 0),
            )
            for r in rows
        ],
    }


@app.get("/sessions/{session_id}/history", response_model=HistoryResponse, tags=["Sessions"])
async def get_history(session_id: str, username: str = Depends(get_current_user)):
    """Get full conversation history for a session."""
    sess = await session_store.get_session(session_id)
    if not sess or sess["user_id"] != username:
        raise HTTPException(status_code=404, detail="Session not found")

    rows = await session_store.get_history(session_id)
    
    messages = []
    for r in rows:
        citations = None
        if r.get("citations"):
            try:
                c_list = json.loads(r["citations"])
                citations = [Citation(**c) for c in c_list]
            except Exception:
                citations = []

        trace = None
        if r.get("pipeline_trace"):
            try:
                t_list = json.loads(r["pipeline_trace"])
                trace = [PipelineTraceEntry(**t) for t in t_list]
            except Exception:
                trace = []

        messages.append(
            MessageRecord(
                id=r["id"],
                session_id=r["session_id"],
                role=r["role"],
                content=r["content"],
                intent=r.get("intent"),
                confidence=r.get("confidence"),
                citations=citations,
                pipeline_trace=trace,
                created_at=r["created_at"],
            )
        )

    return HistoryResponse(
        session_id=session_id,
        messages=messages,
    )


@app.delete("/sessions/{session_id}", tags=["Sessions"])
async def delete_session(session_id: str, username: str = Depends(get_current_user)):
    """Delete a session and all its messages."""
    sess = await session_store.get_session(session_id)
    if not sess or sess["user_id"] != username:
        raise HTTPException(status_code=404, detail="Session not found")

    deleted = await session_store.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "deleted", "session_id": session_id}


# ─── FEEDBACK ENDPOINT ────────────────────────────────────────────────────────

@app.post("/feedback", response_model=FeedbackResponse, tags=["Feedback"])
async def submit_feedback(body: FeedbackRequest, username: str = Depends(get_current_user)):
    """Submit thumbs up/down feedback on a response."""
    sess = await session_store.get_session(body.session_id)
    if not sess or sess["user_id"] != username:
        raise HTTPException(status_code=404, detail="Session not found")

    fb_id = await session_store.save_feedback(
        response_id=body.response_id,
        session_id=body.session_id,
        rating=body.rating,
        comment=body.comment,
    )
    metrics.record_feedback()
    return FeedbackResponse(status="recorded", feedback_id=fb_id)


# ─── DOCUMENT UPLOAD ENDPOINTS ────────────────────────────────────────────────

@app.post("/upload", response_model=UploadResponse, tags=["Documents"])
async def upload_document(
    file: UploadFile = File(...),
    username: str = Depends(get_current_user),
    x_api_key: str | None = Header(default=None),
):
    """Upload a document (PDF, DOCX, TXT, MD, PY) for personal Q&A."""
    if username.startswith("guest-"):
        raise HTTPException(status_code=403, detail="Guest users cannot upload documents")

    from app.upload.processor import process_file, SUPPORTED_EXTENSIONS

    p = _require_pipeline()
    settings = get_settings()

    # Size check
    content = await file.read()
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max {settings.MAX_UPLOAD_SIZE_MB}MB.",
        )

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported type. Allowed: {', '.join(SUPPORTED_EXTENSIONS)}",
        )

    doc_id = str(uuid.uuid4())

    try:
        chunks = process_file(content, file.filename, username, doc_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # Index into user namespace
    chunk_count = p.user_ns.add_documents(username, chunks)

    # Persist metadata to SQLite
    coll_name = p.user_ns.get_collection_name(username)
    await session_store.save_user_document(
        user_id=username,
        filename=file.filename,
        file_type=suffix.lstrip("."),
        collection_name=coll_name,
        chunk_count=chunk_count,
        file_size_bytes=len(content),
    )

    return UploadResponse(
        doc_id=doc_id,
        filename=file.filename,
        file_type=suffix.lstrip("."),
        chunk_count=chunk_count,
        status="indexed",
    )


@app.get("/documents", response_model=DocumentListResponse, tags=["Documents"])
async def list_documents(username: str = Depends(get_current_user)):
    """List uploaded documents for a user."""
    rows = await session_store.list_user_documents(username)
    return DocumentListResponse(
        user_id=username,
        documents=[
            DocumentInfo(
                doc_id=r["id"],
                filename=r["filename"],
                file_type=r["file_type"],
                chunk_count=r["chunk_count"],
                file_size_bytes=r["file_size_bytes"],
                uploaded_at=r["uploaded_at"],
            )
            for r in rows
        ],
    )


@app.delete("/documents/{doc_id}", tags=["Documents"])
async def delete_document(doc_id: str, username: str = Depends(get_current_user)):
    """Delete an uploaded document from index and metadata store."""
    if username.startswith("guest-"):
        raise HTTPException(status_code=403, detail="Guest users cannot delete documents")

    p = _require_pipeline()
    coll_name = await session_store.delete_user_document(doc_id, username)
    if coll_name is None:
        raise HTTPException(status_code=404, detail="Document not found")
    p.user_ns.delete_document(username, doc_id)
    return {"status": "deleted", "doc_id": doc_id}


# ─── SYSTEM ENDPOINTS ─────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Enhanced health check with per-component status."""
    settings = get_settings()
    components = []

    # ChromaDB
    try:
        doc_count = pipeline.get_doc_count() if pipeline else -1
        components.append(ComponentStatus(
            name="ChromaDB",
            status="healthy" if doc_count >= 0 else "unhealthy",
            detail=f"{doc_count:,} chunks",
        ))
    except Exception as e:
        components.append(ComponentStatus(name="ChromaDB", status="unhealthy", detail=str(e)))

    # BM25
    bm25_indexed = bool(pipeline and pipeline.bm25_index)
    components.append(ComponentStatus(
        name="BM25 Index",
        status="healthy" if bm25_indexed else "degraded",
        detail=f"{len(pipeline.bm25_corpus):,} docs" if bm25_indexed else "Not built",
    ))

    # SQLite
    try:
        from app.memory.database import get_db_path
        db_exists = Path(await get_db_path()).exists()
        components.append(ComponentStatus(
            name="SQLite", status="healthy" if db_exists else "degraded"
        ))
    except Exception:
        components.append(ComponentStatus(name="SQLite", status="unhealthy"))

    # Tavily
    if pipeline:
        components.append(ComponentStatus(
            name="Tavily",
            status="healthy" if pipeline.web_search.enabled else "disabled",
            detail="Web search " + ("enabled" if pipeline.web_search.enabled else "disabled"),
        ))

    overall = "healthy" if all(c.status == "healthy" for c in components) else "degraded"
    return HealthResponse(
        status=overall,
        components=components,
        uptime_seconds=round(time.time() - start_time, 1),
        model=settings.MODEL_NAME,
        vectorstore_documents=doc_count if pipeline else -1,
        bm25_indexed=bm25_indexed,
    )


@app.get("/metrics", response_model=MetricsResponse, tags=["System"])
async def get_metrics():
    """Pipeline performance metrics."""
    snap = metrics.snapshot()
    return MetricsResponse(**snap)


# ─── UI ROOT ──────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, tags=["UI"])
async def root():
    """Serve the premium 3-panel Enterprise Chat UI."""
    ui_file = Path(__file__).parent / "static" / "index.html"
    return FileResponse(ui_file, media_type="text/html")
