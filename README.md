# 🐍 Python Programming Q&A Assistant

> RAG-powered Q&A system for Python learners — grounded in 607K+ Stack Overflow posts.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-0.3+-1C3C3C?logo=langchain&logoColor=white)](https://langchain.com)
[![Gemini](https://img.shields.io/badge/Gemini_3.1_Flash_Lite-Free_Tier-4285F4?logo=google&logoColor=white)](https://ai.google.dev)

---

## Architecture

```
              POST /ask {"question": "..."}
                        │
                        ▼
              ┌──────────────────┐
              │   FastAPI Server │
              └────────┬─────────┘
                       │
          ┌────────────▼────────────┐
          │   LangGraph RAG Pipeline │
          │                          │
          │  ┌──────────┐            │
          │  │ RETRIEVE  │ ChromaDB   │
          │  │ (MMR, k=6)│ 30K+ chunks│
          │  └─────┬─────┘            │
          │        ▼                  │
          │  ┌──────────┐            │
          │  │  GRADE    │ Gemini LLM │
          │  │ (filter   │ relevance  │
          │  │  noise)   │ check      │
          │  └─────┬─────┘            │
          │        ▼                  │
          │  ┌──────────┐            │
          │  │ GENERATE  │ Gemini LLM │
          │  │ (grounded │ + citations│
          │  │  answer)  │            │
          │  └──────────┘            │
          └──────────────────────────┘
```

**Pipeline**: `Retrieve → Grade → Generate`

| Component      | Technology                    |
|----------------|-------------------------------|
| Backend        | FastAPI                       |
| LLM            | Gemini 3.1 Flash Lite (Free)  |
| Embeddings     | all-MiniLM-L6-v2 (Local, HF) |
| Orchestration  | LangGraph                     |
| Vector Store   | ChromaDB (persistent)         |
| Data Processing| Pandas + BeautifulSoup        |

---

## Setup

### Prerequisites

- Python 3.10+
- [Gemini API Key](https://aistudio.google.com/apikey) (free tier)

### Install & Run

```bash
# Clone & enter project
cd dev

# Virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Configure API key
copy .env.example .env
# Edit .env → add your GOOGLE_API_KEY

# Ingest data (one-time, ~5-10 min)
python -m app.ingest --sample-size 15000

# Start API
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Endpoints

| Method | Path      | Description                    |
|--------|-----------|--------------------------------|
| GET    | `/`       | Interactive web UI             |
| GET    | `/health` | Service health + vectorstore stats |
| POST   | `/ask`    | Submit a question, get answer  |
| GET    | `/docs`   | Swagger UI                     |

### `/ask` Request / Response

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I reverse a list in Python?"}'
```

```json
{
  "question": "How do I reverse a list in Python?",
  "answer": "...",
  "sources": [
    {"question_id": 613183, "title": "...", "score": 87}
  ],
  "confidence": "high",
  "response_time_ms": 2340
}
```

---

## Project Structure

```
dev/
├── app/
│   ├── main.py           # FastAPI endpoints + landing page
│   ├── rag_pipeline.py   # LangGraph RAG (retrieve→grade→generate)
│   ├── ingest.py         # Data profiling + vectorstore builder
│   └── config.py         # Pydantic settings from .env
├── dataset/              # Stack Overflow CSVs (Questions, Answers, Tags)
├── vectorstore/          # ChromaDB persistent storage
├── test_api.py           # API test suite (10 queries -> PDF report)
├── Dockerfile            # Container build
├── requirements.txt      # Python dependencies
├── render.yaml           # Render.com deploy config
└── .env                  # API keys (not committed)
```

---

## Deployment

**Docker** (Hugging Face Spaces / Render / any container host):

```bash
docker build -t python-qa .
docker run -p 7860:7860 -e GOOGLE_API_KEY=your_key python-qa
```

Set `GOOGLE_API_KEY` as a secret/env variable on your platform.

---

## License

Dataset: [CC-BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) (Stack Overflow)
