from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def recall_at_fpr(y_true, scores, max_fpr: float = 0.001) -> float:
    """Share of fraud caught while flagging at most `max_fpr` of legit transactions."""
    fpr, tpr, _ = roc_curve(y_true, scores)
    return float(tpr[fpr <= max_fpr].max())


def evaluate(y_true, scores) -> dict:
    return {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "recall_at_0.1pct_fpr": recall_at_fpr(y_true, scores, 0.001),
    }