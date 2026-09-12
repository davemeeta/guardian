import numpy as np
import pandas as pd


def train_val_unit_split(
    df: pd.DataFrame, val_frac: float = 0.15, random_state: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out whole engines for validation, never individual cycles.

    Splitting at the row level would leak cycles from the same engine's
    trajectory across train and validation, since consecutive cycles are
    highly correlated — the model would look better than it really is.
    """
    units = df["unit_number"].unique()
    rng = np.random.default_rng(random_state)
    rng.shuffle(units)
    n_val = max(1, int(len(units) * val_frac))
    val_units = set(units[:n_val])
    val_mask = df["unit_number"].isin(val_units)
    return df[~val_mask].copy(), df[val_mask].copy()
