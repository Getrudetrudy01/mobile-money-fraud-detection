import logging

import pandas as pd

from src.config import PROJECT_ROOT, load_config
from src.data.schema import RAW_DTYPES, raw_schema

logger = logging.getLogger(__name__)


def load_raw(path) -> pd.DataFrame:
    logger.info("Loading raw data from %s", path)
    return pd.read_csv(path, dtype=RAW_DTYPES)


def validate(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Validating %d rows against schema", len(df))
    return raw_schema.validate(df, lazy=True)  # lazy=True reports ALL errors at once


def clean(df: pd.DataFrame, keep_types: list, drop_columns: list) -> pd.DataFrame:
    before = len(df)
    df = df[df["type"].isin(keep_types)].copy()
    df["type"] = df["type"].cat.remove_unused_categories()
    df = df.drop(columns=drop_columns)
    df = df.drop_duplicates().reset_index(drop=True)
    logger.info("Kept %d of %d rows (%.1f%%)", len(df), before, 100 * len(df) / before)
    return df


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    cfg = load_config()["data"]

    df = load_raw(PROJECT_ROOT / cfg["raw_path"])
    df = validate(df)
    df = clean(df, cfg["keep_types"], cfg["drop_columns"])

    out_path = PROJECT_ROOT / cfg["clean_path"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    logger.info("Saved %s | fraud rate: %.4f%%", out_path, 100 * df["isFraud"].mean())


if __name__ == "__main__":
    main()