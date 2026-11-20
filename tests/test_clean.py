import pandas as pd
import pandera.errors
import pytest

from src.data.clean import clean, validate
from src.data.schema import RAW_DTYPES


def make_row(**overrides):
    """A valid transaction; override any field to create test cases."""
    row = dict(
        step=1, type="TRANSFER", amount=100.0,
        nameOrig="C1", oldbalanceOrg=100.0, newbalanceOrig=0.0,
        nameDest="C2", oldbalanceDest=0.0, newbalanceDest=100.0,
        isFraud=1, isFlaggedFraud=0,
    )
    row.update(overrides)
    return row


def make_df(rows):
    return pd.DataFrame(rows).astype(RAW_DTYPES)


# ---------- clean() ----------

def test_clean_keeps_only_selected_types():
    df = make_df([
        make_row(type="TRANSFER"),
        make_row(type="CASH_OUT", step=2),
        make_row(type="PAYMENT", nameDest="M1", step=3),
    ])
    result = clean(df, keep_types=["TRANSFER", "CASH_OUT"], drop_columns=[])
    assert len(result) == 2
    assert set(result["type"]) == {"TRANSFER", "CASH_OUT"}


def test_clean_drops_requested_columns():
    df = make_df([make_row()])
    result = clean(df, keep_types=["TRANSFER"], drop_columns=["isFlaggedFraud"])
    assert "isFlaggedFraud" not in result.columns


def test_clean_removes_exact_duplicates():
    df = make_df([make_row(), make_row()])
    result = clean(df, keep_types=["TRANSFER"], drop_columns=[])
    assert len(result) == 1


# ---------- validate() ----------

def test_validate_accepts_valid_data():
    df = make_df([make_row()])
    validate(df)  # should not raise


def test_validate_rejects_negative_amount():
    df = make_df([make_row(amount=-50.0)])
    with pytest.raises(pandera.errors.SchemaErrors):
        validate(df)


def test_validate_rejects_unknown_transaction_type():
    df = make_df([make_row(type="BITCOIN")])
    with pytest.raises(pandera.errors.SchemaErrors):
        validate(df)