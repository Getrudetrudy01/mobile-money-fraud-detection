import pytest
from fastapi.testclient import TestClient

from src.config import PROJECT_ROOT, load_config
from src.serving.app import app

MODEL_PATH = PROJECT_ROOT / load_config()["training"]["model_path"]
pytestmark = pytest.mark.skipif(not MODEL_PATH.exists(), reason="Run `make train` first")

DRAIN = {"step": 3, "type": "TRANSFER", "amount": 500000,
         "oldbalanceOrg": 500000, "oldbalanceDest": 0}
NORMAL = {"step": 14, "type": "CASH_OUT", "amount": 1200,
          "oldbalanceOrg": 85000, "oldbalanceDest": 40000}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:   # `with` triggers lifespan, so the model loads
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_score_returns_valid_probability(client):
    r = client.post("/score", json=NORMAL)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert len(body["top_factors"]) == 3


def test_account_drain_scores_higher_than_normal(client):
    drain = client.post("/score", json=DRAIN).json()["fraud_probability"]
    normal = client.post("/score", json=NORMAL).json()["fraud_probability"]
    assert drain > normal


@pytest.mark.parametrize("bad_field", [
    {"amount": -5},
    {"type": "PAYMENT"},
    {"oldbalanceOrg": -1},
])
def test_invalid_input_is_rejected(client, bad_field):
    r = client.post("/score", json={**NORMAL, **bad_field})
    assert r.status_code == 422