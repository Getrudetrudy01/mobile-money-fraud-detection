import pandas as pd

from src.models.evaluate import recall_at_fpr
from src.models.split import time_split
from src.models.threshold import best_threshold, cost_curve


def test_time_split_is_ordered_and_complete():
    df = pd.DataFrame({"step": list(range(1, 101))})
    train, val, test = time_split(df, 0.70, 0.15)

    assert train["step"].max() < val["step"].min()   # train is before validation
    assert val["step"].max() < test["step"].min()    # validation is before test
    assert len(train) + len(val) + len(test) == 100  # no rows lost


def test_recall_at_fpr_perfect_model():
    y = [0, 0, 0, 1, 1]
    scores = [0.1, 0.2, 0.3, 0.8, 0.9]  # fraud always scores highest
    assert recall_at_fpr(y, scores, max_fpr=0.0) == 1.0
    


def test_best_threshold_minimises_cost():
    y       = [0,   0,   1]
    scores  = [0.2, 0.6, 0.7]
    amounts = [10,  10,  1000]
    curve = cost_curve(y, scores, amounts, false_alarm_cost=5,
                       thresholds=[0.1, 0.5, 0.65, 0.9])
    # 0.1  -> blocks everything: 2 false alarms        = 10
    # 0.5  -> 1 false alarm                            = 5
    # 0.65 -> catches the fraud, no false alarms       = 0   <- best
    # 0.9  -> misses the 1000 fraud                    = 1000
    best = best_threshold(curve)
    assert best["threshold"] == 0.65
    assert best["cost"] == 0