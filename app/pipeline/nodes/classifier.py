"""
Node ③: Intent Classification
Uses Gemini to classify the query into one of four intents, which determines
prompt selection, guardrail strictness, and response style for all downstream nodes.
"""

import time

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.pipeline.state import PipelineState
from app.observability.events import publish

NODE_LABEL = "Intent Classification"
NODE_NAME = "classifier"

VALID_INTENTS = {"rag", "debug", "codegen", "concept", "out_of_topic"}
DEFAULT_INTENT = "rag"

CLASSIFICATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an intent classifier for a Python programming assistant.

Your first job is to determine if the query is related to Python programming, Python libraries, Python syntax, software engineering using Python, standard programming concepts compared to Python, OR {user_docs_instruction}.

Note: You are provided with the recent conversation history to help you classify short or context-dependent queries. If the query refers to concepts or tasks mentioned in the history, classify the query in that context.

If the query (even when interpreted with the context of the history) is NOT related to Python programming and does not reference {user_docs_ref} (e.g. questions about general news, sports, other programming languages like Java/C++ without any reference to Python, movies, recipes, history, general career advice, trivia, user's personal identity/name/chat history facts, etc.), classify it as:
- out_of_topic

If the query IS related to Python programming OR references {user_docs_ref}, classify it into exactly ONE of these intents:
- rag      : factual questions ({rag_instruction}, libraries, APIs, syntax)
- debug    : diagnosing errors, stack traces, exceptions, "why doesn't this work"
- codegen  : request to write/generate/create Python code, functions, classes, scripts
- concept  : explain a concept, theory, comparison ("what is", "difference between", "how does X work")

Rules:
- Respond with ONLY the single intent word (rag, debug, codegen, concept, or out_of_topic).
- No punctuation, no explanation, no markdown."""),
    ("human", """Conversation History:
{history}

Query: {query}"""),
])


def _format_history(history: list[dict], max_turns: int) -> str:
    recent = history[-max_turns:] if max_turns > 0 else []
    return "\n".join(
        f"{h['role'].capitalize()}: {h['content']}" for h in recent
    ) or "None"


async def classifier_node(state: PipelineState, llm) -> dict:
    t0 = time.perf_counter()
    response_id = state.get("response_id", "")
    query = state.get("cleaned_query") or state.get("raw_query", "")

    await publish(response_id, {"type": "node_start", "node": NODE_NAME, "label": NODE_LABEL})

    try:
        import asyncio
        from app.config import get_settings
        settings = get_settings()
        history = state.get("conversation_history", [])
        history_turns = max(1, settings.GENERATION_HISTORY_TURNS)
        history_str = _format_history(history, history_turns)

        include_user_docs = state.get("include_user_docs", False)
        if include_user_docs:
            user_docs_instruction = "references the user's uploaded documents, resume, or files"
            user_docs_ref = "the user's uploaded documents, resume, or files"
            rag_instruction = "including questions about the user's uploaded resume/documents"
        else:
            user_docs_instruction = "references files (if any are uploaded, which currently is disabled)"
            user_docs_ref = "none (document search is disabled)"
            rag_instruction = "factual questions about Python"

        chain = CLASSIFICATION_PROMPT | llm | StrOutputParser()
        raw = await asyncio.wait_for(
            chain.ainvoke({
                "query": query,
                "history": history_str,
                "user_docs_instruction": user_docs_instruction,
                "user_docs_ref": user_docs_ref,
                "rag_instruction": rag_instruction
            }),
            timeout=5.0
        )
        intent = raw.strip().lower().split()[0] if raw.strip() else DEFAULT_INTENT
        if intent not in VALID_INTENTS:
            intent = DEFAULT_INTENT
    except Exception as exc:
        intent = DEFAULT_INTENT

    latency = int((time.perf_counter() - t0) * 1000)
    intent_labels = {
        "rag": "📚 RAG Query",
        "debug": "🐛 Debug Query",
        "codegen": "⚙️ Code Generation",
        "concept": "💡 Concept Explanation",
        "out_of_topic": "❌ Out of Topic",
    }
    detail = intent_labels.get(intent, intent)

    await publish(response_id, {"type": "node_done", "node": NODE_NAME, "label": NODE_LABEL,
                                "status": "success", "latency_ms": latency, "detail": detail})
    return {
        "intent": intent,
        "pipeline_trace": state.get("pipeline_trace", []) + [{
            "node": NODE_NAME, "label": NODE_LABEL,
            "status": "success", "latency_ms": latency, "detail": detail
        }]
    }

