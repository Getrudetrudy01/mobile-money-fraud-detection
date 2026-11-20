import numpy as np
import pandas as pd

from src.features.build_features import add_balance_features, add_destination_history


def test_destination_history_uses_only_earlier_hours():
    df = pd.DataFrame({
        "nameDest": ["C9", "C9", "C9", "C8"],
        "step":     [1,    1,    5,    5],
        "amount":   [100.0, 50.0, 30.0, 10.0],
    })
    out = add_destination_history(df)

    # Same-hour transactions don't count as each other's history
    assert out["dest_prior_txn_count"].tolist() == [0, 0, 2, 0]
    assert out["dest_prior_amount_sum"].tolist() == [0.0, 0.0, 150.0, 0.0]
    # C9 last received at hour 1, so at hour 5 that's 4 hours ago
    assert out["dest_hours_since_last"].iloc[2] == 4
    # First appearance of a receiver has no history
    assert out["dest_hours_since_last"].iloc[[0, 1, 3]].isna().all()


def test_balance_features():
    df = pd.DataFrame({
        "amount":         [50.0, 100.0, 100.0],
        "oldbalanceOrg":  [100.0, 100.0, 0.0],
        "oldbalanceDest": [0.0, 10.0, 10.0],
    })
    out = add_balance_features(df)

    assert out["amount_to_balance"].iloc[0] == 0.5
    assert np.isnan(out["amount_to_balance"].iloc[2])  # zero balance -> undefined
    assert out["takes_full_balance"].tolist() == [0, 1, 0]
    assert out["orig_zero_balance"].tolist() == [0, 0, 1]
    assert out["dest_zero_balance"].tolist() == [1, 0, 0]