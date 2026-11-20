import json
from contextlib import asynccontextmanager

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI

from src.config import PROJECT_ROOT, load_config
from src.features.build_features import (
    REALISTIC_FEATURES,
    add_balance_features,
    add_time_features,
    add_transaction_features,
)
from src.serving.schemas import Factor, Prediction, Transaction

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model ONCE at startup, not on every request."""
    cfg = load_config()
    model_path = PROJECT_ROOT / cfg["training"]["model_path"]
    meta = json.loads((model_path.parent / "model_meta.json").read_text())

    state["model"] = joblib.load(model_path)
    state["threshold"] = meta["threshold"]
    state["version"] = meta["version"]
    state["night_hours"] = cfg["features"]["night_hours"]
    yield
    state.clear()


app = FastAPI(title="Mobile Money Fraud Scoring API", version="1.0.0", lifespan=lifespan)


def to_features(txn: Transaction) -> pd.DataFrame:
    """Turn one transaction into model features using the SAME code as training."""
    df = pd.DataFrame([txn.model_dump()])
    df = add_transaction_features(df)
    df = add_balance_features(df)
    df = add_time_features(df, state["night_hours"])
    return df[REALISTIC_FEATURES].astype("float64")  # None -> NaN for missing history


@app.get("/health")
def health():
    return {"status": "ok", "model_version": state.get("version")}


@app.post("/score", response_model=Prediction)
def score(txn: Transaction) -> Prediction:
    X = to_features(txn)
    model = state["model"]

    probability = float(model.predict_proba(X)[0, 1])

    # Per-feature contributions (SHAP values computed by LightGBM); last value is the baseline
    contributions = model.predict(X, pred_contrib=True)[0][:-1]
    top = np.argsort(-np.abs(contributions))[:3]
    factors = [Factor(feature=REALISTIC_FEATURES[i], contribution=round(float(contributions[i]), 4))
               for i in top]

    return Prediction(
        fraud_probability=round(probability, 6),
        is_fraud=probability >= state["threshold"],
        threshold=state["threshold"],
        model_version=state["version"],
        top_factors=factors,
    )