# 🐍 PyMind — Enterprise Python AI Assistant

> A premium, production-grade 12-stage RAG-powered Python assistant — grounded in 607K+ Stack Overflow posts, custom user document uploads, and dynamic Tavily Web Search fallback.
> 
> **Active Deployment**: [Hugging Face Space](https://huggingface.co/spaces/fearman99/python-interview-assistant)

---

## 🌟 Key Features

* **12-Stage Enterprise Pipeline**: Built with **LangGraph** to execute a production-grade 12-stage pipeline:
  `Authentication → Query Validation → Intent Classification → Query Rewriting → Hybrid Retrieval (Vector + BM25) → Re-ranking (RRF) → Context Compression → Hallucination Check → Answer Generation → Citation Verification → Safety Guardrails → Observability Logging`.
* **Dynamic Web Fallback (Tavily)**: If a Python-related query is not found in the local Stack Overflow database, the assistant automatically triggers a web search fallback via Tavily, citing live external sources.
* **Topic Routing**: Filters out-of-topic queries instantly during intent classification, bypassing database lookups to save latency.
* **Persistent Cloud Sync (Turso)**: Fully integrated with Turso SQLite in the cloud to store user accounts, sessions, and chat history.
* **User Authentication & Guest Mode**:
  * **Authenticated Mode**: Secure pbkdf2 password hashing, persistent session tracking.
  * **Guest Mode**: Isolated, volatile in-memory storage (`GUEST_SESSIONS`) that cleans up completely upon logout/exit.
* **Dynamic Document Uploads**: Upload `.pdf`, `.docx`, `.txt`, `.md`, and `.py` files. Chunks are embedded and queried dynamically alongside Stack Overflow.
* **Premium UI/UX**: Cohesive light and midnight-blue dark themes, live interactive pipeline tracer, and expandable citation pill layouts.

---

## 🗺️ Pipeline Architecture

```
                                  POST /stream {"question": "..."}
                                                │
                                                ▼
                                      ┌──────────────────┐
                                      │   FastAPI Server │
                                      └────────┬─────────┘
                                               │
               ┌───────────────────────────────▼───────────────────────────────┐
               │                     12-Stage LangGraph Pipeline               │
               │                                                               │
               │  ① Authentication      ⟶ Validates API keys & cookies          │
               │  ② Query Validation    ⟶ Disallows empty/gibberish queries     │
               │  ③ Intent Classify     ⟶ Detects intent & topic boundaries     │
               │  ④ Query Rewriting     ⟶ HyDE query expansion                  │
               │  ⑤ Hybrid Retrieval    ⟶ ChromaDB (Vector) + BM25 index        │
               │  ⑥ Re-ranking          ⟶ Reciprocal Rank Fusion (RRF)          │
               │  ⑦ Context Compression ⟶ Fits context inside token budget      │
               │  ⑧ Hallucination Check ⟶ Grounds context (Fallback trigger)    │
               │  ⑨ Answer Generation   ⟶ Streams grounded answer (or Tavily)   │
               │  ⑩ Citation Verify     ⟶ Links sources (SO & Web)              │
               │  ⑪ Safety Guardrails   ⟶ LLM Content Moderation                │
               │  ⑫ Observability Log   ⟶ Saves to Turso/SQLite & computes logs │
               └───────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Component |
| :--- | :--- |
| **Backend & API** | FastAPI, SSE (Server-Sent Events) |
| **Database & History**| Turso (Cloud libSQL) / SQLite |
| **Vector Index** | ChromaDB (Local Embeddings: `all-MiniLM-L6-v2`) |
| **Search Fallback** | Tavily Web Search API |
| **Orchestration** | LangGraph & LangChain |
| **LLM Core** | Gemini 1.5 Flash (via Google Generative AI) |
| **Frontend** | Vanilla HTML5, JS (ES6), CSS3 custom design system |

---

## 🚀 Setup & Installation

### 1. Prerequisites
* Python 3.10+
* [Google Gemini API Key](https://aistudio.google.com/apikey)
* [Tavily API Key](https://tavily.com)
* [Turso Cloud DB Account & Credentials](https://turso.tech)

### 2. Local Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd dev

# Set up virtual environment
python -m venv venv
venv\Scripts\activate      # On Windows
# source venv/bin/activate # On macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
copy .env.example .env     # On Windows
# cp .env.example .env     # On macOS/Linux
```

Open `.env` and fill in your keys:
```env
GOOGLE_API_KEY=AIzaSy...
TAVILY_API_KEY=tvly-...
TURSO_DATABASE_URL=libsql://...
TURSO_AUTH_TOKEN=ey...
```

### 3. Startup

```bash
# Ingest Stack Overflow dataset (one-time setup, ingests ~28k chunks)
python -m app.ingest --sample-size 15000

# Start local server
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Access the interactive web UI at [http://127.0.0.1:8000](http://127.0.0.1:8000).

---

## 🐳 Docker & Cloud Deployment

### Local Docker Build & Run
```bash
docker build -t python-qa .
docker run -p 7860:7860 --env-file .env python-qa
```

### Deploying to Hugging Face Spaces (Free hosting, 24/7 uptime)
1. Create a new **Docker** Space on Hugging Face using the **Blank** template.
2. Under space **Settings**, add the environment secrets: `GOOGLE_API_KEY`, `TAVILY_API_KEY`, `TURSO_DATABASE_URL`, and `TURSO_AUTH_TOKEN`.
3. Add Hugging Face remote and push your code:
   ```bash
   git remote add hf https://huggingface.co/spaces/YOUR_HF_USERNAME/YOUR_SPACE_NAME
   git push -f hf main
   ```

---

## 📂 Project Structure

```
dev/
├── app/
│   ├── core/             # Embeddings initializer
│   ├── memory/           # Turso/SQLite Database & Session Stores
│   ├── observability/    # Logging, Event emitter & Pipeline metrics
│   ├── pipeline/         # LangGraph stages & Nodes
│   ├── static/           # UI Frontend (index.html with design system)
│   ├── tools/            # Web Search (Tavily) Wrapper
│   ├── upload/           # Document parser (PDF, DOCX, TXT, PY, MD)
│   ├── utils/            # Authentication crypt utilities & PII scanners
│   ├── main.py           # FastAPI server configuration & routing
│   └── config.py         # Application settings
├── dataset/              # Source datasets
├── vectorstore/          # Persistent ChromaDB collection
├── test_api.py           # Evaluation report builder
├── Dockerfile            # Container configs
├── render.yaml           # Render deployment template
└── requirements.txt      # Required packages
```

---

## 📄 License
This project is open-source. Stack Overflow dataset source is retrieved from the [Kaggle Stack Overflow Python Questions Dataset](https://www.kaggle.com/datasets/stackoverflow/pythonquestions) and licensed under [CC-BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/).
