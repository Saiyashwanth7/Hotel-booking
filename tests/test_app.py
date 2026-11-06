# tests/test_app.py  (run from project root: pytest -q)
# Uses the real TF-IDF retriever and a fake LLM, so no API key is needed.
import pytest
from fastapi.testclient import TestClient

import src.rag_system as rag_module
from src.api import app


@pytest.fixture(scope="module")
def client():
    rag_module.RAGSystem._call_llm = lambda self, prompt: "LLM_ANSWER"
    with TestClient(app) as c:
        # pretend a token exists so the LLM path is used
        from src.api import state

        state["rag"].api_key = "fake"
        yield c


def ask(client, q):
    r = client.post("/ask", json={"question": q})
    assert r.status_code == 200, r.text
    return r.json()


def test_basic_endpoints(client):
    assert client.get("/").status_code == 200
    h = client.get("/health").json()
    assert h["status"] == "healthy"
    assert h["components"]["data_processor"]["records"] == 119390


def test_analytics_filters(client):
    r = client.post("/analytics", json={}).json()
    assert r["metadata"]["total_records"] == 119390
    r = client.post("/analytics", json={"filters": {"hotel": "Resort Hotel"}}).json()
    assert r["metadata"]["total_records"] == 40060
    r = client.post("/analytics", json={"filters": {"abc": 1}}).json()
    assert r["metadata"]["filters_ignored"] == ["abc"]
    assert (
        client.post("/analytics", json={"filters": {"hotel": "Taj Hotel"}}).status_code
        == 404
    )


@pytest.mark.parametrize(
    "question,expected",
    [
        ("Show me total revenue for July 2017.", "3,132,959.07"),
        ("What is the average price of a hotel booking?", "101.83"),
        ("Which locations had the highest booking cancellations?", "Portugal"),
        ("What is the cancellation rate for resort hotels?", "27.76%"),
        ("What is the average lead time for Portugal?", "115.57"),
        ("Which month had the highest revenue in 2016?", "August 2016"),
        ("What is the overall cancellation rate?", "37.04%"),
        ("What is the cancellation rate for city hotel?", "41.73%"),
    ],
)
def test_computed_answers(client, question, expected):
    out = ask(client, question)
    assert expected in out["answer"]
    assert out["metadata"]["source"] == "computed"


def test_hotel_comparison(client):
    out = ask(
        client, "Which hotel type has a higher cancellation rate, resort or city?"
    )
    assert "27.76%" in out["answer"] and "41.73%" in out["answer"]
    assert out["metadata"]["source"] == "computed"


def test_llm_path_and_history(client):
    out = ask(client, "Who is the manager of the hotel?")
    assert out["answer"] == "LLM_ANSWER"
    assert out["metadata"]["source"] == "rag_llm"
    assert len(out["retrieved_documents"]) > 0
    hist = client.get("/query-history").json()["history"]
    assert hist[-1]["question"] == "Who is the manager of the hotel?"


def test_llm_failure_falls_back(client, monkeypatch):
    def boom(self, prompt):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(rag_module.RAGSystem, "_call_llm", boom)
    out = ask(client, "Tell me something interesting about weekends")
    assert out["metadata"]["source"] == "retrieval_only"


def test_validation(client):
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/ask", json={"question": "   "}).status_code == 422
    assert client.post("/ask", json={"q": "hi"}).status_code == 422
