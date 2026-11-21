import json
import logging

import joblib
import lightgbm as lgb
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import mlflow.lightgbm
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import PROJECT_ROOT, load_config
from src.features.build_features import LEAKY_FEATURES, REALISTIC_FEATURES, TARGET
from src.models.evaluate import evaluate
from src.models.split import time_split
from src.models.threshold import best_threshold, cost_curve

logger = logging.getLogger(__name__)

NO_DRAIN_FEATURES = [c for c in REALISTIC_FEATURES
                     if c not in ("takes_full_balance", "amount_to_balance")]

EXPERIMENTS = [
    # (run name,             model type, feature list)
    ("rule_full_balance",   "rule",     ["takes_full_balance"]),
    ("logistic_regression", "logreg",   REALISTIC_FEATURES),
    ("lightgbm_realistic",  "lightgbm", REALISTIC_FEATURES),
    ("lightgbm_leaky",      "lightgbm", REALISTIC_FEATURES + LEAKY_FEATURES),
    ("lightgbm_no_drain",   "lightgbm", NO_DRAIN_FEATURES),
]


def train_logistic(X_train, y_train):
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1000),
    )
    return model.fit(X_train, y_train)


def train_lightgbm(X_train, y_train, X_val, y_val, params: dict):
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    return model


def run_experiment(name, model_type, features, train, val, test, cfg):
    """Train one model, evaluate it, choose its cost-based threshold, log to MLflow."""
    with mlflow.start_run(run_name=name):
        mlflow.log_params({"model_type": model_type, "n_features": len(features)})
        mlflow.log_dict({"features": features}, "features.json")

        # --- train ---
        model = None
        if model_type == "rule":
            def score(d):
                return d["takes_full_balance"].to_numpy().astype(float)
        elif model_type == "logreg":
            model = train_logistic(train[features], train[TARGET])
        else:
            mlflow.log_params(cfg["lightgbm"])
            model = train_lightgbm(train[features], train[TARGET],
                                   val[features], val[TARGET], cfg["lightgbm"])
            mlflow.log_metric("best_iteration", model.best_iteration_)
            importance = pd.Series(model.booster_.feature_importance("gain"), index=features)
            importance = (importance / importance.sum()).sort_values(ascending=False)
            mlflow.log_dict(importance.round(4).to_dict(), "feature_importance.json")

        if model is not None:
            def score(d):
                return model.predict_proba(d[features])[:, 1]

        val_scores, test_scores = score(val), score(test)

        # --- ranking metrics ---
        metrics = {"val": evaluate(val[TARGET], val_scores),
                   "test": evaluate(test[TARGET], test_scores)}
        for split, m in metrics.items():
            mlflow.log_metrics({f"{split}_{k}": v for k, v in m.items()})

        # --- cost-based threshold: CHOSEN on validation, REPORTED on test ---
        cost = cfg["false_alarm_cost"]
        chosen = best_threshold(cost_curve(val[TARGET], val_scores, val["amount"], cost))
        at_test = cost_curve(test[TARGET], test_scores, test["amount"], cost,
                             thresholds=[chosen["threshold"]]).iloc[0]
        decision = {
            "threshold": float(chosen["threshold"]),
            "test_recall": float(at_test["recall"]),
            "test_precision": float(at_test["precision"]),
            "test_false_alarms": int(at_test["false_alarms"]),
            "test_missed_fraud_amount": float(at_test["missed_fraud_amount"]),
            "test_total_cost": float(at_test["cost"]),
        }
        mlflow.log_metrics(decision)
        metrics["decision"] = decision

        example = val[features].head(5).astype("float64")
        if model_type == "logreg":
            mlflow.sklearn.log_model(model, name="model", input_example=example,
                                     skops_trusted_types=["numpy.dtype"])
        elif model_type == "lightgbm":
            mlflow.lightgbm.log_model(model, name="model", input_example=example)

        logger.info("%-20s | test PR-AUC %.4f | threshold %.2f | recall %.3f | false alarms %d",
                    name, metrics["test"]["pr_auc"], decision["threshold"],
                    decision["test_recall"], decision["test_false_alarms"])
        return model, metrics


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    cfg = load_config()["training"]

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment"])

    df = pd.read_parquet(PROJECT_ROOT / cfg["features_path"])
    train, val, test = time_split(df, cfg["train_frac"], cfg["val_frac"])
    for name, part in [("train", train), ("val", val), ("test", test)]:
        logger.info("%-5s | steps %3d-%3d | %9d rows | fraud rate %.3f%%",
                    name, part["step"].min(), part["step"].max(), len(part),
                    100 * part[TARGET].mean())

    results, models = {}, {}
    for name, model_type, features in EXPERIMENTS:
        models[name], results[name] = run_experiment(
            name, model_type, features, train, val, test, cfg)

    summary = pd.DataFrame({
        name: {"pr_auc": r["test"]["pr_auc"],
               "recall@0.1%fpr": r["test"]["recall_at_0.1pct_fpr"],
               **{k: v for k, v in r["decision"].items()
                  if k in ("threshold", "test_recall", "test_precision", "test_false_alarms")}}
        for name, r in results.items()
    }).T
    print("\nTEST SET RESULTS\n", summary.round(4).to_string())

    metrics_path = PROJECT_ROOT / cfg["metrics_path"]
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(results, indent=2))

    model_path = PROJECT_ROOT / cfg["model_path"]
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(models["lightgbm_realistic"], model_path)
    meta = {
        "version": pd.Timestamp.now(tz="UTC").strftime("%Y%m%d-%H%M%S"),
        "model": "lightgbm_realistic",
        "features": REALISTIC_FEATURES,
        "threshold": results["lightgbm_realistic"]["decision"]["threshold"],
        "test_pr_auc": results["lightgbm_realistic"]["test"]["pr_auc"],
    }
    (model_path.parent / "model_meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("Saved metrics to %s and model to %s", metrics_path, model_path)


if __name__ == "__main__":
    main()