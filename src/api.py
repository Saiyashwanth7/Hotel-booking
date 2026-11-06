# src/api.py
import json
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

load_dotenv()

from .analytics import BookingAnalytics  # noqa: E402
from .data_processing import DataProcessor  # noqa: E402
from .rag_system import RAGSystem  # noqa: E402

DATA_PATH = os.environ.get(
    "DATA_PATH",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "hotel_bookings.csv"),
)

state: Dict[str, Any] = {"data": None, "rag": None, "analytics_cache": {}}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading data...")
    state["data"] = DataProcessor(DATA_PATH).get_processed_data()
    print(f"Loaded {len(state['data'])} records. Building vector index...")
    rag = RAGSystem(state["data"])
    n_docs = rag.build_index()
    state["rag"] = rag
    print(f"Index ready with {n_docs} documents.")
    yield


app = FastAPI(
    title="Hotel Booking Analytics & QA API",
    description="Analytics and RAG-based question answering over hotel booking data",
    lifespan=lifespan,
)


class AnalyticsRequest(BaseModel):
    filters: Optional[Dict[str, Any]] = None


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)

@app.head("/")
@app.get("/")
def root():
    return {"message": "Hotel Booking Analytics & QA API. See /docs for the interactive API docs."}


@app.post("/analytics")
def get_analytics(request: AnalyticsRequest):
    data = state["data"]
    filters = request.filters or {}
    applied, ignored = {}, []
    for col, value in filters.items():
        if col in data.columns:
            data = data[data[col] == value]
            applied[col] = value
        else:
            ignored.append(col)

    if data.empty:
        raise HTTPException(status_code=404, detail="No bookings match the given filters")

    key = json.dumps(applied, sort_keys=True, default=str)
    cache = state["analytics_cache"]
    if key not in cache:
        cache[key] = BookingAnalytics(data).generate_all_analytics()
    return {
        "analytics": cache[key],
        "metadata": {"total_records": len(data), "filters_applied": applied, "filters_ignored": ignored},
    }


@app.post("/ask")
def ask_question(request: QuestionRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question must not be empty")
    try:
        return state["rag"].answer_question(question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {e}")


@app.get("/query-history")
def get_query_history():
    return {"history": state["rag"].get_query_history()}


@app.get("/health")
def health_check():
    return {
        "status": "healthy" if state["rag"] and state["rag"].index is not None else "initializing",
        "components": {
            "data_processor": {"status": "healthy", "records": len(state["data"])},
            "rag_system": {
                "status": "healthy" if state["rag"].index is not None else "initializing",
                "documents": len(state["rag"].documents),
                "llm_configured": bool(state["rag"].api_key),
                "llm_model": state["rag"].llm_model,
                "llm_base_url": state["rag"].base_url,
            },
        },
    }
