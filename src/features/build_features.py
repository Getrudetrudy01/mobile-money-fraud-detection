import logging

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, load_config

logger = logging.getLogger(__name__)

# Features available at decision time (safe for a real-time model)
REALISTIC_FEATURES = [
    "amount", "log_amount", "is_transfer",
    "oldbalanceOrg", "oldbalanceDest",
    "amount_to_balance", "takes_full_balance", "orig_zero_balance", "dest_zero_balance",
    "hour", "is_night", "day_of_week",
    "dest_prior_txn_count", "dest_prior_amount_sum", "dest_hours_since_last",
]
# Post-transaction information: ONLY for the leakage comparison experiment
LEAKY_FEATURES = ["newbalanceOrig", "newbalanceDest", "errorOrig", "errorDest"]
TARGET = "isFraud"


def add_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    df["log_amount"] = np.log1p(df["amount"]).astype("float32")
    df["is_transfer"] = (df["type"] == "TRANSFER").astype("int8")
    return df


def add_balance_features(df: pd.DataFrame) -> pd.DataFrame:
    old = df["oldbalanceOrg"]
    # Ratio is undefined when the balance is 0, so leave it as NaN (missing)
    df["amount_to_balance"] = (df["amount"] / old.where(old > 0)).astype("float32")
    df["takes_full_balance"] = ((old > 0) & (df["amount"] == old)).astype("int8")
    df["orig_zero_balance"] = (old == 0).astype("int8")
    df["dest_zero_balance"] = (df["oldbalanceDest"] == 0).astype("int8")
    return df


def add_time_features(df: pd.DataFrame, night_hours: list) -> pd.DataFrame:
    df["hour"] = (df["step"] % 24).astype("int8")
    df["is_night"] = df["hour"].isin(night_hours).astype("int8")
    df["day_of_week"] = ((df["step"] // 24) % 7).astype("int8")
    return df


def add_destination_history(df: pd.DataFrame) -> pd.DataFrame:
    """History of each receiving account, using ONLY earlier hours (no leakage).

    Transactions within the same hour have no reliable order, so they are
    never counted as each other's history.
    """
    per_step = (
        df.groupby(["nameDest", "step"])
        .agg(n=("amount", "size"), total=("amount", "sum"))
        .reset_index()
        .sort_values(["nameDest", "step"])
    )
    g = per_step.groupby("nameDest")
    per_step["dest_prior_txn_count"] = g["n"].cumsum() - per_step["n"]
    per_step["dest_prior_amount_sum"] = g["total"].cumsum() - per_step["total"]
    per_step["dest_hours_since_last"] = per_step["step"] - g["step"].shift(1)

    cols = ["nameDest", "step", "dest_prior_txn_count",
            "dest_prior_amount_sum", "dest_hours_since_last"]
    return df.merge(per_step[cols], on=["nameDest", "step"], how="left")


def add_leaky_features(df: pd.DataFrame) -> pd.DataFrame:
    df["errorOrig"] = df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]
    df["errorDest"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
    return df


def build_features(df: pd.DataFrame, night_hours: list) -> pd.DataFrame:
    df = df.copy()
    df = add_transaction_features(df)
    df = add_balance_features(df)
    df = add_time_features(df, night_hours)
    df = add_leaky_features(df)
    df = add_destination_history(df)
    # Keep step for the time-based split later; drop raw IDs
    return df[["step"] + REALISTIC_FEATURES + LEAKY_FEATURES + [TARGET]]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    cfg = load_config()["features"]

    df = pd.read_parquet(PROJECT_ROOT / cfg["input_path"])
    logger.info("Loaded %d rows", len(df))

    features = build_features(df, cfg["night_hours"])

    out_path = PROJECT_ROOT / cfg["output_path"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out_path, index=False)
    logger.info("Saved %s with shape %s", out_path, features.shape)


if __name__ == "__main__":
    main()