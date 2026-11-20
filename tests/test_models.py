import pandas as pd

from src.models.evaluate import recall_at_fpr
from src.models.split import time_split


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