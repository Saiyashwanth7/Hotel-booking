# Hotel Booking Analytics and QA (RAG)

FastAPI service that does two things on the hotel bookings dataset (119,390 rows):

1. `/analytics`: charts and stats (revenue trends, cancellation rate, top countries, lead time, guest composition)
2. `/ask`: natural language questions answered with a hybrid approach

## How /ask works

1. The question is matched against ~350 summary documents (monthly, yearly, per country, per segment) using TF-IDF retrieval. For better semantic search you can switch to sentence-transformers + FAISS (see below).
2. Common metrics (revenue for a month, cancellation rate, average lead time, average price, top cancellation countries, best revenue month) are computed exactly with pandas. The response says `"source": "computed"`.
3. Everything else goes to an LLM (any OpenAI-compatible API, default OpenRouter free models) with the retrieved documents as context. The response says `"source": "rag_llm"`.
4. If the LLM is unavailable, the retrieved documents are returned with `"source": "retrieval_only"`.

Revenue is ADR x total nights and includes canceled bookings.

## Run locally

```
python -m venv venv
venv\Scripts\activate          (Windows)   |   source venv/bin/activate   (Mac/Linux)
pip install -r requirements.txt
copy .env.example .env         (then put your API key in .env)
python main.py
```

Open http://localhost:8000/docs

Optional transformer retrieval: `pip install -r requirements-transformer.txt` and set `EMBEDDING_BACKEND=transformer` (needs about 1 GB RAM).

## Tests

```
pytest -q                                   # offline, uses a fake LLM
python tests/evaluate_performance.py        # against a running server
```

## Deploy on Render (free)

1. Push this project to a GitHub repo (`.env` is git-ignored, never commit it).
2. Render dashboard > New > Web Service > connect the repo.
3. Settings:
   - Runtime: Python
   - Build command: `pip install -r requirements.txt`
   - Start command: `uvicorn src.api:app --host 0.0.0.0 --port $PORT`
   - Instance type: Free
4. Environment variables:
   - `PYTHON_VERSION` = `3.11.9`
   - `LLM_API_KEY` = your OpenRouter key
   - `LLM_BASE_URL` = `https://openrouter.ai/api/v1`
   - `LLM_MODEL` = `openrouter/free`
5. Deploy, then open `https://<service-name>.onrender.com/docs`.

The free instance sleeps after about 15 minutes without traffic, so the first request after a break takes around a minute.
A `Dockerfile` and `render.yaml` are included if you prefer those routes.

## Sample requests

```
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d "{\"question\": \"Show me total revenue for July 2017.\"}"
curl -X POST http://localhost:8000/analytics -H "Content-Type: application/json" -d "{\"filters\": {\"hotel\": \"Resort Hotel\"}}"
```
