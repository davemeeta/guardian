import numpy as np
import pandas as pd


def make_train_windows(
    df: pd.DataFrame, feature_cols: list[str], window_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Sliding windows of length window_size over each unit's full trajectory.

    Units shorter than window_size are left-padded by repeating their first
    row, so short-lived (early-failure) engines still contribute samples
    instead of being dropped.
    """
    X, y = [], []
    for _, unit_df in df.groupby("unit_number"):
        unit_df = unit_df.sort_values("time_cycles")
        feats = unit_df[feature_cols].to_numpy()
        rul = unit_df["RUL"].to_numpy()
        pad_len = max(0, window_size - len(feats))
        if pad_len:
            pad = np.repeat(feats[:1], pad_len, axis=0)
            feats = np.concatenate([pad, feats], axis=0)
        for end in range(window_size, len(feats) + 1):
            X.append(feats[end - window_size:end])
            y.append(rul[end - 1 - pad_len])
    return np.stack(X), np.array(y)


def make_test_windows(
    df: pd.DataFrame, feature_cols: list[str], window_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Last window_size cycles of each test unit (left-padded if too short)."""
    X, unit_numbers = [], []
    for unit, unit_df in df.groupby("unit_number"):
        unit_df = unit_df.sort_values("time_cycles")
        feats = unit_df[feature_cols].to_numpy()
        if len(feats) < window_size:
            pad = np.repeat(feats[:1], window_size - len(feats), axis=0)
            feats = np.concatenate([pad, feats], axis=0)
        X.append(feats[-window_size:])
        unit_numbers.append(unit)
    return np.stack(X), np.array(unit_numbers)
