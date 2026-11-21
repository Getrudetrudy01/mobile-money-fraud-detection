import pandas as pd


def time_split(df: pd.DataFrame, train_frac: float, val_frac: float):
    """Split by time: earliest hours -> train, then validation, then test.

    Each hour (step) lands in exactly one split, so no hour is shared.
    """
    train_end = df["step"].quantile(train_frac)
    val_end = df["step"].quantile(train_frac + val_frac)

    train = df[df["step"] <= train_end]
    val = df[(df["step"] > train_end) & (df["step"] <= val_end)]
    test = df[df["step"] > val_end]
    return train, val, test