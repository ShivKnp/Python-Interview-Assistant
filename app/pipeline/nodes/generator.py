"""
Node ⑨: Answer Generation
Uses intent-specific system prompts and injects conversation history for
multi-turn coherence. Falls back to general knowledge when context is insufficient.
"""

import asyncio
import time
from contextlib import suppress

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.config import get_settings
from app.pipeline.state import PipelineState
from app.observability.events import publish

NODE_LABEL = "Answer Generation"
NODE_NAME = "generator"

# ── Intent-specific system prompts ────────────────────────────────────────────

_SYSTEM_RAG = """You are an expert Python programming assistant grounded in real Stack Overflow content.

Rules:
1. Answer using ONLY the provided context documents. Do not fabricate information.
2. If context is insufficient, say so clearly.
3. Format code in Python code blocks with triple backticks.
4. Be concise but thorough. Explain clearly for learners.
5. Reference sources naturally ("According to a highly-voted Stack Overflow answer...").
6. Present multiple approaches when available, with trade-off explanations.
7. Always include practical code examples when relevant."""

_SYSTEM_DEBUG = """You are an expert Python debugger. Diagnose errors methodically and precisely.

Rules:
1. Identify the ROOT CAUSE of the error, not just symptoms.
2. Explain WHY the error occurs (the underlying mechanism).
3. Provide a clear FIX with code examples.
4. List common related pitfalls to avoid.
5. Reference context documents where available for grounding.
6. If multiple causes are possible, address each."""

_SYSTEM_CODEGEN = """You are an expert Python developer generating clean, production-quality code.

Rules:
1. Write complete, runnable Python code with no placeholders.
2. Follow PEP 8 style guidelines.
3. Add docstrings and inline comments for clarity.
4. Include usage examples with expected output.
5. Mention dependencies (imports) and explain non-obvious design decisions.
6. Prefer standard library solutions unless context shows a library is better."""

_SYSTEM_CONCEPT = """You are a Python educator explaining concepts clearly and deeply.

Rules:
1. Start with a concise one-sentence definition.
2. Use analogies and real-world comparisons.
3. Show concrete code examples that illustrate the concept.
4. When comparing things, use a clear comparison table or bullet list.
5. End with "When to use X vs Y" guidance where applicable.
6. Calibrate depth for intermediate developers."""

_SYSTEM_USER_DOCS = """You are an expert Python programming and career assistant.
You have access to the user's uploaded documents (such as resumes, design docs, or codebase files) as well as Stack Overflow reference content.

Rules:
1. Ground your answers in the provided context documents (e.g. the resume, design docs, code).
2. Synthesize, analyze, or brainstorm based on the documents when requested (e.g., generating interview questions based on the candidate's experience, or summarizing design specs).
3. If the context is completely unrelated or insufficient to answer the question, state so clearly.
4. Format code in Python code blocks with triple backticks.
5. Reference sources naturally (e.g. "According to your uploaded resume...").
6. Answer queries thoroughly, keeping the tone helpful and professional."""

INTENT_SYSTEMS = {
    "rag": _SYSTEM_RAG,
    "debug": _SYSTEM_DEBUG,
    "codegen": _SYSTEM_CODEGEN,
    "concept": _SYSTEM_CONCEPT,
}

_GROUNDED_HUMAN = """Context documents:
---
{context}
---

Conversation history:
{history}

Question: {question}

Provide a clear, well-structured answer grounded in the context above.
Default to a concise response unless the user explicitly asks for deep detail."""

_FALLBACK_HUMAN = """Conversation history:
{history}

Question: {question}

Note: Limited relevant context was found. Provide your best answer based on Python expertise.
Be honest that this answer is from general knowledge, not grounded sources.
Default to a concise response unless the user explicitly asks for deep detail."""


def _format_history(history: list[dict], max_turns: int) -> str:
    recent = history[-max_turns:] if max_turns > 0 else []
    return "\n".join(
        f"{h['role'].capitalize()}: {h['content']}" for h in recent
    ) or "None"


async def generator_node(state: PipelineState, llm, web_search_tool=None) -> dict:
    t0 = time.perf_counter()
    response_id = state.get("response_id", "")
    settings = get_settings()

    await publish(response_id, {"type": "node_start", "node": NODE_NAME, "label": NODE_LABEL})

    intent = state.get("intent", "rag")
    query = state.get("cleaned_query") or state.get("raw_query", "")
    context = state.get("compressed_context", "")
    can_answer = state.get("can_answer", True)
    history = state.get("conversation_history", [])

    # ── Out of Topic Handling ───────────────────────────────────────────────
    if intent == "out_of_topic":
        generation = "I am a Python assistant. I can only answer questions related to Python programming, libraries, syntax, or debugging."
        await publish(response_id, {"type": "token", "token": generation})
        latency = int((time.perf_counter() - t0) * 1000)
        await publish(response_id, {"type": "node_done", "node": NODE_NAME, "label": NODE_LABEL,
                                    "status": "success", "latency_ms": latency, "detail": "Skipped (out of topic)"})
        return {
            "generation": generation,
            "pipeline_trace": state.get("pipeline_trace", []) + [{
                "node": NODE_NAME, "label": NODE_LABEL,
                "status": "success", "latency_ms": latency, "detail": "Out of Topic Response"
            }]
        }

    # ── On-the-fly Web Fallback ─────────────────────────────────────────────
    reranked_docs = state.get("reranked_docs", [])
    web_docs = state.get("web_docs", [])

    if (not context or not can_answer) and web_search_tool and web_search_tool.enabled:
        await publish(response_id, {"type": "thinking", "message": "Query not found on Stack Overflow. Triggering web search..."})
        web_docs = web_search_tool.search(query)
        if web_docs:
            context = "\n\n".join(
                f"[Source: Web] {doc.page_content}"
                for doc in web_docs
            )
            can_answer = True
            reranked_docs = list(reranked_docs) + web_docs
            from app.observability.metrics import metrics
            metrics.record_tavily_fallback()

    # Keep prompt compact to reduce generation latency.
    history_turns = max(1, settings.GENERATION_HISTORY_TURNS)
    history_str = _format_history(history, history_turns)

    has_user_docs = bool(state.get("user_doc_results"))
    if has_user_docs and intent == "rag":
        system_prompt = _SYSTEM_USER_DOCS
    else:
        system_prompt = INTENT_SYSTEMS.get(intent, _SYSTEM_RAG)

    if can_answer and context:
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", _GROUNDED_HUMAN),
        ])
    else:
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", _FALLBACK_HUMAN),
        ])

    llm_for_generation = llm.bind(
        max_output_tokens=max(128, settings.GENERATION_MAX_OUTPUT_TOKENS),
        temperature=0.2,
    )
    chain = prompt | llm_for_generation | StrOutputParser()

    await publish(response_id, {"type": "thinking", "message": "Reading retrieved context..."})

    first_token_at: float | None = None

    async def _emit_waiting_updates() -> None:
        milestones = [
            (2.0, "Planning answer structure..."),
            (5.0, "Synthesizing the most relevant points..."),
            (9.0, "Generating final wording..."),
        ]
        emitted = 0
        while first_token_at is None and emitted < len(milestones):
            await asyncio.sleep(0.25)
            elapsed = time.perf_counter() - t0
            mark, message = milestones[emitted]
            if elapsed >= mark:
                await publish(response_id, {"type": "thinking", "message": message})
                emitted += 1

    waiting_task = asyncio.create_task(_emit_waiting_updates())

    generation = ""
    try:
        async for chunk in chain.astream({
            "context": context if (can_answer and context) else "",
            "history": history_str,
            "question": query,
        }):
            if first_token_at is None:
                first_token_at = time.perf_counter()
            generation += chunk
            await publish(response_id, {"type": "token", "token": chunk})
    finally:
        waiting_task.cancel()
        with suppress(asyncio.CancelledError):
            await waiting_task

    latency = int((time.perf_counter() - t0) * 1000)
    ttft_ms = int(((first_token_at or time.perf_counter()) - t0) * 1000)
    token_est = len(generation) // 4
    if web_docs:
        detail = f"Generated (Switched to Tavily Web Search fallback, TTFT {ttft_ms}ms, ~{token_est} tokens, {latency}ms)"
    else:
        mode = "grounded" if (can_answer and context) else "fallback"
        detail = f"Generated ({mode} mode, TTFT {ttft_ms}ms, ~{token_est} tokens, {latency}ms)"

    await publish(response_id, {"type": "node_done", "node": NODE_NAME, "label": NODE_LABEL,
                                "status": "success", "latency_ms": latency, "detail": detail})
    return {
        "generation": generation,
        "reranked_docs": reranked_docs,
        "web_docs": web_docs,
        "can_answer": can_answer,
        "compressed_context": context,
        "pipeline_trace": state.get("pipeline_trace", []) + [{
            "node": NODE_NAME, "label": NODE_LABEL,
            "status": "success", "latency_ms": latency, "detail": detail
        }]
    }
