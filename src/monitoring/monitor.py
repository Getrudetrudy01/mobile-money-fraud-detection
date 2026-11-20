import json
import logging

import joblib
import matplotlib
import pandas as pd

matplotlib.use("Agg")  # draw charts to files, no screen needed
import matplotlib.pyplot as plt

from src.config import PROJECT_ROOT, load_config
from src.features.build_features import REALISTIC_FEATURES, TARGET
from src.models.split import time_split
from src.monitoring.drift import psi

logger = logging.getLogger(__name__)


def daily_drift(reference: pd.DataFrame, current: pd.DataFrame, features: list, bins: int) -> pd.DataFrame:
    """PSI of every feature, for every day of the current period."""
    rows = []
    for day, chunk in current.groupby("day"):
        row = {"day": int(day), "n_txns": len(chunk)}
        for f in features:
            row[f] = psi(reference[f], chunk[f], bins)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_heatmap(drift: pd.DataFrame, features: list, path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(drift[features].T, aspect="auto", cmap="Reds", vmin=0, vmax=0.5)
    ax.set_yticks(range(len(features)), labels=features)
    ax.set_xticks(range(len(drift)), labels=drift["day"])
    ax.set_xlabel("Day of simulation")
    ax.set_title("Feature drift vs. training data (PSI; > 0.25 = significant)")
    fig.colorbar(im, label="PSI")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    cfg = load_config()
    tcfg, mcfg = cfg["training"], cfg["monitoring"]
    report_dir = PROJECT_ROOT / mcfg["report_dir"]
    report_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(PROJECT_ROOT / tcfg["features_path"])
    train, _, production = time_split(df, tcfg["train_frac"], tcfg["val_frac"])
    production = production.assign(day=production["step"] // 24)

    # A random sample of training data is enough to describe "normal" and is much faster
    reference = train.sample(n=min(len(train), mcfg["reference_sample"]), random_state=42)

    # 1. Data drift
    drift = daily_drift(reference, production, REALISTIC_FEATURES, mcfg["psi_bins"])

    # 2. Prediction drift: how often does the model raise alerts each day?
    model_path = PROJECT_ROOT / tcfg["model_path"]
    model = joblib.load(model_path)
    meta = json.loads((model_path.parent / "model_meta.json").read_text())
    production = production.assign(
        alert=model.predict_proba(production[REALISTIC_FEATURES])[:, 1] >= meta["threshold"])
    daily = (production.groupby("day")
             .agg(alert_rate=("alert", "mean"), fraud_rate=(TARGET, "mean"))
             .reset_index())
    report = drift.merge(daily, on="day")

    # Which features drifted each day?
    threshold = mcfg["psi_threshold"]
    report["drifted"] = report[REALISTIC_FEATURES].apply(
        lambda row: [f for f in REALISTIC_FEATURES if row[f] > threshold], axis=1)
    report["n_drifted"] = report["drifted"].str.len()

    for _, r in report.iterrows():
        logger.info("day %2d | %7d txns | alert rate %6.3f%% | drifted: %s",
                    r["day"], r["n_txns"], 100 * r["alert_rate"], r["drifted"] or "-")

    # 3. Retraining decision
    recent = report["n_drifted"].tail(mcfg["consecutive_days"])
    retrain = bool((recent > 0).all())
    reference_alert_rate = float(
        (model.predict_proba(reference[REALISTIC_FEATURES])[:, 1] >= meta["threshold"]).mean())

    summary = {
        "model_version": meta["version"],
        "days_monitored": len(report),
        "days_with_drift": int((report["n_drifted"] > 0).sum()),
        "most_drifted_features": report[REALISTIC_FEATURES].max().sort_values(ascending=False)
                                       .head(5).round(3).to_dict(),
        "reference_alert_rate": reference_alert_rate,
        "production_alert_rate": float(production["alert"].mean()),
        "retrain_recommended": retrain,
    }

    report.to_csv(report_dir / "drift_report.csv", index=False)
    (report_dir / "monitoring_summary.json").write_text(json.dumps(summary, indent=2))
    plot_heatmap(report, REALISTIC_FEATURES, report_dir / "drift_heatmap.png")

    logger.info("Alert rate: training %.3f%% -> production %.3f%%",
                100 * summary["reference_alert_rate"], 100 * summary["production_alert_rate"])
    logger.info("RETRAIN RECOMMENDED: %s", retrain)


if __name__ == "__main__":
    main()