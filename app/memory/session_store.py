"""
Session store — async CRUD helpers for sessions, messages, feedback,
and user document metadata (backed by connect_db or in-memory guest store).
"""

import uuid
from datetime import datetime, timezone
from typing import Optional

import aiosqlite

from app.config import get_settings
from app.memory.database import connect_db


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# In-memory guest session store: session_id -> session_details
GUEST_SESSIONS = {}


# ─── Sessions ────────────────────────────────────────────────────────────────

async def create_session(user_id: str, title: Optional[str] = None) -> str:
    session_id = f"guest-{uuid.uuid4()}" if user_id.startswith("guest") else str(uuid.uuid4())
    now = _now()
    if title is None:
        title = "New Session"

    if user_id.startswith("guest") or session_id.startswith("guest"):
        GUEST_SESSIONS[session_id] = {
            "id": session_id,
            "user_id": user_id,
            "title": title,
            "created_at": now,
            "last_active_at": now,
            "messages": []
        }
        return session_id

    async with connect_db() as db:
        await db.execute(
            "INSERT INTO sessions (id, user_id, title, created_at, last_active_at) VALUES (?,?,?,?,?)",
            (session_id, user_id, title, now, now),
        )
        await db.commit()
    return session_id


async def get_session(session_id: str) -> Optional[dict]:
    if session_id.startswith("guest"):
        s = GUEST_SESSIONS.get(session_id)
        if s:
            return {
                "id": s["id"],
                "user_id": s["user_id"],
                "title": s["title"],
                "created_at": s["created_at"],
                "last_active_at": s["last_active_at"]
            }
        return None

    async with connect_db() as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM sessions WHERE id=?", (session_id,)) as cur:
            row = await cur.fetchone()
    return dict(row) if row else None


async def list_sessions(user_id: str) -> list[dict]:
    if user_id.startswith("guest"):
        res = []
        for s in GUEST_SESSIONS.values():
            if s["user_id"] == user_id:
                res.append({
                    "id": s["id"],
                    "user_id": s["user_id"],
                    "title": s["title"],
                    "created_at": s["created_at"],
                    "last_active_at": s["last_active_at"],
                    "message_count": len(s["messages"])
                })
        return sorted(res, key=lambda x: x["last_active_at"], reverse=True)

    async with connect_db() as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """SELECT s.*, COUNT(m.id) AS message_count
               FROM sessions s
               LEFT JOIN messages m ON m.session_id = s.id
               WHERE s.user_id=?
               GROUP BY s.id
               ORDER BY s.last_active_at DESC""",
            (user_id,),
        ) as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def touch_session(session_id: str) -> None:
    """Update last_active_at timestamp."""
    now = _now()
    if session_id.startswith("guest"):
        if session_id in GUEST_SESSIONS:
            GUEST_SESSIONS[session_id]["last_active_at"] = now
        return

    async with connect_db() as db:
        await db.execute(
            "UPDATE sessions SET last_active_at=? WHERE id=?", (now, session_id)
        )
        await db.commit()


async def delete_session(session_id: str) -> bool:
    if session_id.startswith("guest"):
        if session_id in GUEST_SESSIONS:
            GUEST_SESSIONS.pop(session_id)
            return True
        return False

    async with connect_db() as db:
        await db.execute("DELETE FROM feedback WHERE session_id=?", (session_id,))
        await db.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
        cur = await db.execute("DELETE FROM sessions WHERE id=?", (session_id,))
        await db.commit()
    return (cur.rowcount or 0) > 0


# ─── Messages ────────────────────────────────────────────────────────────────

async def save_message(
    session_id: str,
    role: str,
    content: str,
    intent: Optional[str] = None,
    confidence: Optional[str] = None,
    response_id: Optional[str] = None,
    citations: Optional[list] = None,
    pipeline_trace: Optional[list] = None,
) -> str:
    msg_id = f"guest-msg-{uuid.uuid4()}" if session_id.startswith("guest") else str(uuid.uuid4())
    import json
    citations_json = json.dumps(citations) if citations else None
    pipeline_trace_json = json.dumps(pipeline_trace) if pipeline_trace else None

    if session_id.startswith("guest"):
        if session_id in GUEST_SESSIONS:
            msg = {
                "id": msg_id,
                "session_id": session_id,
                "role": role,
                "content": content,
                "intent": intent,
                "confidence": confidence,
                "response_id": response_id,
                "citations": citations_json,
                "pipeline_trace": pipeline_trace_json,
                "created_at": _now()
            }
            GUEST_SESSIONS[session_id]["messages"].append(msg)
            if role == "user":
                title = GUEST_SESSIONS[session_id]["title"]
                if title in ("", "Session", "New Session"):
                    new_title = content.strip().replace("\n", " ")
                    if len(new_title) > 40:
                        new_title = new_title[:37] + "..."
                    GUEST_SESSIONS[session_id]["title"] = new_title
            GUEST_SESSIONS[session_id]["last_active_at"] = _now()
        return msg_id

    async with connect_db() as db:
        await db.execute(
            """INSERT INTO messages
               (id, session_id, role, content, intent, confidence, response_id, citations, pipeline_trace, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (msg_id, session_id, role, content, intent, confidence, response_id, citations_json, pipeline_trace_json, _now()),
        )
        if role == "user":
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT title FROM sessions WHERE id=?", (session_id,)) as cur:
                row = await cur.fetchone()
                if row and (row["title"] is None or row["title"] in ("", "Session", "New Session")):
                    title = content.strip().replace("\n", " ")
                    if len(title) > 40:
                        title = title[:37] + "..."
                    await db.execute("UPDATE sessions SET title=? WHERE id=?", (title, session_id))
        await db.commit()
    await touch_session(session_id)
    return msg_id


async def get_history(session_id: str, last_n: Optional[int] = None) -> list[dict]:
    """Return messages ordered chronologically (oldest first)."""
    if session_id.startswith("guest"):
        if session_id not in GUEST_SESSIONS:
            return []
        msgs = GUEST_SESSIONS[session_id]["messages"]
        if last_n:
            return msgs[-last_n:]
        return msgs

    limit_clause = f"LIMIT {last_n}" if last_n else ""
    query = f"""
        SELECT * FROM (
            SELECT * FROM messages WHERE session_id=? ORDER BY created_at DESC {limit_clause}
        ) ORDER BY created_at ASC
    """
    async with connect_db() as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(query, (session_id,)) as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def get_conversation_pairs(session_id: str, last_n_turns: int = 10) -> list[dict]:
    """Return last N turn pairs as [{role, content}] for LLM context injection."""
    rows = await get_history(session_id, last_n=last_n_turns * 2)
    return [{"role": r["role"], "content": r["content"]} for r in rows]


# ─── Feedback ────────────────────────────────────────────────────────────────

async def save_feedback(
    response_id: str,
    session_id: str,
    rating: int,
    comment: Optional[str] = None,
) -> str:
    if session_id.startswith("guest"):
        return f"guest-fb-{uuid.uuid4()}"

    fb_id = str(uuid.uuid4())
    async with connect_db() as db:
        await db.execute(
            "INSERT INTO feedback (id, response_id, session_id, rating, comment, created_at) VALUES (?,?,?,?,?,?)",
            (fb_id, response_id, session_id, rating, comment, _now()),
        )
        await db.commit()
    return fb_id


# ─── User Documents ───────────────────────────────────────────────────────────

async def save_user_document(
    user_id: str,
    filename: str,
    file_type: str,
    collection_name: str,
    chunk_count: int,
    file_size_bytes: int,
) -> str:
    if user_id.startswith("guest"):
        return f"guest-doc-{uuid.uuid4()}"

    doc_id = str(uuid.uuid4())
    async with connect_db() as db:
        await db.execute(
            """INSERT INTO user_documents
               (id, user_id, filename, file_type, collection_name, chunk_count, file_size_bytes, uploaded_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (doc_id, user_id, filename, file_type, collection_name, chunk_count, file_size_bytes, _now()),
        )
        await db.commit()
    return doc_id


async def list_user_documents(user_id: str) -> list[dict]:
    if user_id.startswith("guest"):
        return []

    async with connect_db() as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM user_documents WHERE user_id=? ORDER BY uploaded_at DESC", (user_id,)
        ) as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def delete_user_document(doc_id: str, user_id: str) -> Optional[str]:
    """Return collection_name so caller can also purge ChromaDB. None if not found."""
    if user_id.startswith("guest"):
        return None

    async with connect_db() as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT collection_name FROM user_documents WHERE id=? AND user_id=?", (doc_id, user_id)
        ) as cur:
            row = await cur.fetchone()
        if not row:
            return None
        coll_name = row["collection_name"]
        await db.execute("DELETE FROM user_documents WHERE id=?", (doc_id,))
        await db.commit()
    return coll_name
