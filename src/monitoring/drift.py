import numpy as np
import pandas as pd


def _distribution(x: pd.Series, edges=None, categories=None) -> np.ndarray:
    """Share of values in each bucket, plus one extra bucket for missing values."""
    if categories is not None:
        counts = x.value_counts().reindex(categories, fill_value=0).to_numpy()
    else:
        counts = pd.cut(x.dropna(), edges).value_counts(sort=False).to_numpy()
    counts = np.append(counts, x.isna().sum())
    return counts / max(len(x), 1)


def psi(reference, current, bins: int = 10, eps: float = 1e-4) -> float:
    """Population Stability Index between a reference and a current sample."""
    ref = pd.Series(reference, dtype="float64").reset_index(drop=True)
    cur = pd.Series(current, dtype="float64").reset_index(drop=True)

    if ref.nunique() <= bins:
        # Few distinct values (e.g. 0/1 flags, hour): compare each value directly
        cats = np.union1d(ref.dropna().unique(), cur.dropna().unique())
        p = _distribution(ref, categories=cats)
        q = _distribution(cur, categories=cats)
    else:
        # Continuous: bucket edges at the reference's 10%, 20%, ... 90% points
        inner = np.unique(np.nanquantile(ref, np.linspace(0, 1, bins + 1)[1:-1]))
        edges = np.concatenate([[-np.inf], inner, [np.inf]])
        p = _distribution(ref, edges=edges)
        q = _distribution(cur, edges=edges)

    p, q = p + eps, q + eps  # avoid log(0) when a bucket is empty
    return float(np.sum((q - p) * np.log(q / p)))