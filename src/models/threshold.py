import numpy as np
import pandas as pd


def cost_curve(y_true, scores, amounts, false_alarm_cost: float, thresholds=None) -> pd.DataFrame:
    """Total business cost at each threshold.

    Missed fraud costs its transaction amount; each false alarm costs a fixed amount.
    """
    y = np.asarray(y_true)
    s = np.asarray(scores)
    a = np.asarray(amounts, dtype="float64")
    if thresholds is None:
        thresholds = np.round(np.linspace(0.01, 0.99, 99), 2)

    rows = []
    for t in thresholds:
        flagged = s >= t
        missed = ~flagged & (y == 1)
        false_alarms = flagged & (y == 0)
        caught = flagged & (y == 1)
        rows.append({
            "threshold": float(t),
            "cost": a[missed].sum() + false_alarm_cost * false_alarms.sum(),
            "missed_fraud_amount": a[missed].sum(),
            "false_alarms": int(false_alarms.sum()),
            "recall": caught.sum() / max((y == 1).sum(), 1),
            "precision": caught.sum() / max(flagged.sum(), 1),
        })
    return pd.DataFrame(rows)


def best_threshold(curve: pd.DataFrame) -> pd.Series:
    """The row with the lowest total cost."""
    return curve.loc[curve["cost"].idxmin()]