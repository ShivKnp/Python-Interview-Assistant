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

Your first job is to determine if the query is related to Python programming, Python libraries, Python syntax, software engineering using Python, or standard programming concepts compared to Python.

If the query is NOT related to Python programming (e.g. questions about general news, sports, other programming languages like Java/C++ without any reference to Python, movies, recipes, history, general career advice, trivia, etc.), classify it as:
- out_of_topic

If the query IS related to Python programming, classify it into exactly ONE of these intents:
- rag      : factual question about Python (library usage, APIs, syntax, best practices)
- debug    : diagnosing errors, stack traces, exceptions, "why doesn't this work"
- codegen  : request to write/generate/create Python code, functions, classes, scripts
- concept  : explain a concept, theory, comparison ("what is", "difference between", "how does X work")

Rules:
- Respond with ONLY the single intent word (rag, debug, codegen, concept, or out_of_topic).
- No punctuation, no explanation, no markdown."""),
    ("human", "Query: {query}"),
])


async def classifier_node(state: PipelineState, llm) -> dict:
    t0 = time.perf_counter()
    response_id = state.get("response_id", "")
    query = state.get("cleaned_query") or state.get("raw_query", "")

    await publish(response_id, {"type": "node_start", "node": NODE_NAME, "label": NODE_LABEL})

    try:
        import asyncio
        chain = CLASSIFICATION_PROMPT | llm | StrOutputParser()
        raw = await asyncio.wait_for(chain.ainvoke({"query": query}), timeout=5.0)
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

