# 🧠 PyMind — System Workflow & Architecture Guide

This document provides a comprehensive guide to the workflow, architecture, and features of the **PyMind Enterprise Python AI Assistant**. It traces every module, data flow, and specifically details the **12-stage LangGraph pipeline**.

---

## 📋 Table of Contents
1. [Core Features & Architecture Overview](#1-core-features--architecture-overview)
2. [12-Stage LangGraph Pipeline Details](#2-12-stage-langgraph-pipeline-details)
3. [Out-of-Topic Routing & Web Fallback Flows](#3-out-of-topic-routing--web-fallback-flows)
4. [Authentication & Session Management](#4-authentication--session-management)
5. [Database Architecture & Connection Cache](#5-database-architecture--connection-cache)
6. [Frontend Rendering & UI Systems](#6-frontend-rendering--ui-systems)

---

## 1. Core Features & Architecture Overview

PyMind is built on a modular client-server architecture designed for high throughput, low latency, and premium security boundaries.

```
                      [ Client Browser (HTML5/JS/CSS3) ]
                                  │       ▲
          SSE Streams (Real-Time) │       │ REST APIs (JSON)
                                  ▼       │
                       [ FastAPI Backend Gateway ]
                                  │
      ┌───────────────────────────┼───────────────────────────┐
      ▼                           ▼                           ▼
[ LangGraph Pipeline ]    [ Database Layer ]      [ Vector Database ]
  - 12-Node RAG Graph       - Turso Cloud Sync      - Local ChromaDB
  - Gemini LLM Core         - Volatile Guest Store  - Embeddings Cache
```

### Key Capabilities:
* **Hybrid Retrieval (Vector + Lexical)**: Queries ChromaDB (using semantic embeddings from the local `all-MiniLM-L6-v2` transformer) and executes BM25 keyword matching across 600K+ Stack Overflow index chunks.
* **On-the-Fly Web Search Fallback**: If the local index doesn't contain the answer to a Python query, the pipeline triggers a Tavily search fallback mid-stream, fetching current web sources and generating a grounded response.
* **Personalized RAG (User Uploads)**: Allows users to upload code scripts, PDFs, text, or markdown files. These files are chunked, embedded locally, and searched alongside Stack Overflow.
* **Persistent Chat Logs**: Connects to Turso cloud database for permanent accounts, sessions, and histories.
* **Visual Pipeline Tracer**: Streams status events to the UI so users can view the latency, badge status, and detail trace of every stage.

---

## 2. 12-Stage LangGraph Pipeline Details

The core request processor is an asynchronous **LangGraph** state machine. Each request executes a sequence of 12 distinct nodes, passing an instance of `PipelineState` between them.

```
┌──────┐    ┌───────────┐    ┌────────────┐    ┌──────────┐    ┌───────────┐    ┌──────────┐
│ Auth ├───⟶│ Validator ├───⟶│ Classifier ├───⟶│ Rewriter ├───⟶│ Retriever ├───⟶│ Reranker │
└──────┘    └───────────┘    └──────┬─────┘    └──────────┘    └───────────┘    └────┬─────┘
                                    │ (Out of Topic)                                 │
                                    ▼                                                ▼
┌──────┐    ┌───────────┐    ┌──────┴─────┐    ┌──────────┐    ┌───────────┐    ┌────┴─────┐
│ Log  │⟵───┤  Safety   │⟵───┤    Cite    │⟵───┤ Generator│⟵───┤  Ground   │⟵───┤ Compress │
└──────┘    └───────────┘    └────────────┘    └──────────┘    └───────────┘    └──────────┘
```

### Node ①: Authentication (`auth`)
* **Purpose**: Validates the requester's identity.
* **Input**: Session tokens from cookies or request headers (`X-Session-Token`, `Authorization: Bearer`).
* **Processing**: Looks up token against active database sessions. Bypasses for guest users.
* **Output**: Sets authenticated `user_id` or raises `401 Unauthorized`.

### Node ②: Query Validation (`validator`)
* **Purpose**: Rejects empty, dangerous, or malicious queries before they hit database indices.
* **Input**: User's raw text query.
* **Processing**: Performs input length constraints, strips PII (Personal Identifiable Information) like email patterns, and flags gibberish inputs.
* **Output**: Sets validation status; if invalid, sets `error_type` to halt execution.

### Node ③: Intent Classification (`classifier`)
* **Purpose**: Determines the intent of the query to optimize retrieval strategy.
* **Input**: Cleaned user query.
* **Processing**: Leverages Gemini to classify query into: `rag` (factual), `debug` (error code), `codegen` (writing scripts), `concept` (theoretical explanations), or `out_of_topic`.
* **Output**: Sets `intent` in state.

### Node ④: Query Rewriting (`rewriter`)
* **Purpose**: Expands keyword matching keywords to resolve slang and generate ideal retrieval variants.
* **Input**: Cleaned query.
* **Processing**: Uses Gemini in a HyDE (Hypothetical Document Embedding) prompt to generate two query variants plus a hypothetical ideal answer snippet.
* **Output**: Returns a list of 3-4 expanded search variants.

### Node ⑤: Hybrid Retrieval (`retriever`)
* **Purpose**: Gathers source context.
* **Input**: Expanded queries.
* **Processing**: Performs semantic search on ChromaDB, keyword search via BM25, and queries uploaded user documents if enabled.
* **Output**: Generates parallel lists of candidate documents.

### Node ⑥: Re-ranking (`reranker`)
* **Purpose**: Selects the absolute highest quality passages from multiple index types.
* **Input**: Vector, BM25, and User Doc list candidates.
* **Processing**: Merges lists using **Reciprocal Rank Fusion (RRF)**. Applies diversity filtering to select top 5 documents.
* **Output**: Sets final list of `reranked_docs`.

### Node ⑦: Context Compression (`compressor`)
* **Purpose**: Compresses context to save LLM tokens and maximize generation speed.
* **Input**: Reranked documents list.
* **Processing**: Formats headers (`[Stack Overflow Q#... | Score: ...]`), trims content, and merges passages up to a character budget (approx. 2400 characters).
* **Output**: Sets `compressed_context` string.

### Node ⑧: Hallucination Check (`hallucination`)
* **Purpose**: Checks if retrieved context is actually relevant to the question to prevent hallucinated answers.
* **Input**: Compressed context and query.
* **Processing**: Prompts Gemini to review the context and state if it contains the answer (`can_answer: True/False`).
* **Output**: Sets `can_answer` boolean.

### Node ⑨: Answer Generation (`generator`)
* **Purpose**: Generates the final user-facing response.
* **Input**: Compressed context, intent, history, and `can_answer` state.
* **Processing**: 
  * If `intent == "out_of_topic"`, immediately outputs out-of-topic canned rejection.
  * If `can_answer` is `False` (Stack Overflow database missing info), triggers **on-the-fly Tavily search**, reconstructs context, and generates grounded answer.
  * Otherwise, generates a grounded answer using Stack Overflow docs.
* **Output**: Streams generated answer tokens to SSE, returns final `generation` text.

### Node ⑩: Citation Verification (`citation`)
* **Purpose**: Extracts and maps sources back to clickable links.
* **Input**: Reranked documents (including any on-the-fly web fallback docs).
* **Processing**: Isolates URLs and Stack Overflow question IDs, deduplicates references, and maps them to clean metadata.
* **Output**: Returns structured list of `citations`.

### Node ⑪: Safety Guardrails (`guardrails`)
* **Purpose**: Ensures assistant's outputs are safe and free from toxic content.
* **Input**: Final generated response.
* **Processing**: Reviews generated text against content policy.
* **Output**: Sets `is_safe` boolean. Replaces generation with safe fallback if flagged.

### Node ⑫: Logging & Observability (`observer`)
* **Purpose**: Logs pipeline metrics and saves conversation logs.
* **Input**: Complete accumulated pipeline state.
* **Processing**: Logs structured execution metadata, records latencies on global metrics counters, and persists chat history to SQLite/Turso.
* **Output**: Emits the final `complete` SSE event and closes the stream.

---

## 3. Out-of-Topic Routing & Web Fallback Flows

PyMind implements specialized branching paths inside the pipeline to save latency and ensure accuracy:

### A. Out-of-Topic Early Exit (Latency: ~0ms after Classification)
When a user asks a question unrelated to Python:
1. **Node ③ (Intent)** tags the query as `out_of_topic`.
2. Downstream nodes (**④ Rewriter, ⑤ Retriever, ⑥ Reranker, ⑦ Compressor, ⑧ Hallucination**) detect `intent == "out_of_topic"` at start, log a skipped status, and immediately exit in 0ms.
3. **Node ⑨ (Generator)** skips LLM synthesis and instantly outputs the pre-set canned response.
4. **Result**: Fast response without querying database or wasting tokens.

### B. On-the-Fly Web Search Fallback (Triggered in Generation Node)
If a Python query is asked, but the local Stack Overflow index doesn't have the answer:
1. **Node ⑧ (Hallucination)** evaluates the Stack Overflow context and sets `can_answer = False`.
2. **Node ⑨ (Generator)** intercepts `can_answer == False`, halts local generation, and triggers Tavily Web Search.
3. The retrieved web passages are combined into a fresh context string, `can_answer` is overridden to `True`, and the web docs are appended to `reranked_docs`.
4. The generator executes the standard Gemini grounded generator using the web context.
5. **Node ⑩ (Citations)** parses the updated `reranked_docs` and extracts the web URLs to render active external link pills.

---

## 4. Authentication & Session Management

To support environments like **Hugging Face Spaces** (which runs the app inside a third-party `<iframe>` where browsers block standard cookies), PyMind implements a **hybrid dual-auth channel**:

```
                       [ Login / Registration ]
                                  │
                      ┌───────────┴───────────┐
                      ▼                       ▼
           [ Cookie Auth Channel ]   [ Header Auth Channel ]
             SameSite=None; Secure     X-Session-Token Header
             (Direct tab / API)        (Iframe fallback)
```

### 1. Account Creation & Password Protection
* **Registration (`/signup`)**: Hashes the user's password using the **PBKDF2** algorithm with SHA-256 and 100,000 hashing iterations. Generates a random session token.
* **Authentication (`/login`)**: Verifies password hashes and generates a secure UUID4 session token.

### 2. Dual Authorization Verification
In `get_current_user`, the backend searches for session tokens in this order:
1. **HTTP Only Session Cookie** (`session_token`)
2. **Custom HTTP Header** (`X-Session-Token`)
3. **Bearer Authorization Header** (`Authorization: Bearer <token>`)

This ensures that even if a browser blocks the cookie inside a Hugging Face Space iframe, the frontend will fall back to reading the token from `localStorage` and passing it in headers, making registration and login work seamlessly.

### 3. Guest Mode
* If the user bypasses auth by selecting **Guest Mode**, the frontend generates a temporary username (`guest-xxxxxxxx`).
* The backend intercepts the `X-Guest-User` header. All guest chats and sessions are routed to the **volatile in-memory `GUEST_SESSIONS` cache**.
* No guest data is ever written to the persistent SQLite/Turso database. When the guest logs out or exits, the dictionary entry is deleted, preserving privacy.

---

## 5. Database Architecture & Connection Cache

### 1. Database Schema
PyMind utilizes 6 relational tables:
* `users`: Stores unique usernames and PBKDF2 password hashes.
* `user_sessions`: Map of active UUID session tokens, user associations, and expirations.
* `sessions`: Conversation session identifiers, titles, and creation timestamps.
* `messages`: Message payloads associated with a session (records role, content, intent, confidence, response ID, citations list, and pipeline trace).
* `feedback`: User feedback evaluations (thumbs up/down) mapped to specific messages.
* `user_documents`: Document upload metadata (filename, types, chunk count, upload time).

### 2. Singleton Connection Caching
To eliminate connection establishment latencies (DNS, TLS handshake) on database queries to the cloud:
* `connect_db` is implemented as an async context manager that maintains a single **cached global database client instance** (`_global_client`).
* Queries reuse the existing active client session. This optimizes database transaction time from **5-6 seconds down to under 50ms**.
* The global client is gracefully closed in the FastAPI lifecycle shutdown phase.

---

## 6. Frontend Rendering & UI Systems

The frontend is a single-page app utilizing custom CSS and JavaScript designed for visual excellence.

* **SSE Streaming Parsing**: As tokens are received from the `/stream` endpoint, they are rendered in real time. Latency values and status badges are updated inside the Tracer panel as `node_start` and `node_done` events are parsed.
* **Interactive Citations Toggle**: If a response yields more than 5 citation pills, the frontend displays only the first 5 alongside a dashed `+ Show [x] more` button. Clicking the button uses a `display: contents` CSS override, enabling the rest of the pills to wrap seamlessly inside the message box layout.
* **Theme System**: Persists the user's chosen mode (light/dark) in `localStorage`. Defaults to Light Mode on first visit, transitioning to Midnight Blue Dark Mode seamlessly with CSS variables.
