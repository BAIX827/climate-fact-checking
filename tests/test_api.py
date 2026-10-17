"""API tests with a fake pipeline (no models or corpus are loaded)."""

import pytest
from app.main import app
from fastapi.testclient import TestClient


class FakePipeline:
    class retriever:
        text = {"demo-01": "Sea level is rising."}

    def verify(self, claim, top_k):
        return {
            "claim": claim,
            "label": "SUPPORTS",
            "label_scores": {"SUPPORTS": 0.9},
            "evidence": [{"id": "demo-01", "text": "Sea level is rising.", "score": 5.0}][:top_k],
        }


@pytest.fixture
def client():
    app.state.pipeline = FakePipeline()
    with TestClient(app) as c:
        yield c
    app.state.pipeline = None


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "passages": 1}


def test_verify(client):
    r = client.post("/verify", json={"claim": "Sea levels are rising", "top_k": 1})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] == "SUPPORTS" and body["evidence"][0]["id"] == "demo-01"


@pytest.mark.parametrize("payload", [{}, {"claim": "x"}, {"claim": "valid claim", "top_k": 0}])
def test_verify_rejects_bad_input(client, payload):
    assert client.post("/verify", json=payload).status_code == 422


def test_index_page(client):
    assert "Climate Fact-Checking" in client.get("/").text
