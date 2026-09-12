import pandas as pd

# Cap on training RUL labels. Early in an engine's life, RUL is not linearly
# predictable from sensor readings (the engine looks "healthy" regardless of
# how far it actually is from failure), so training the model to regress the
# true unbounded RUL there just adds noise. Clipping to a constant ceiling
# once RUL exceeds it (a piecewise-linear degradation assumption) is the
# standard treatment introduced by Heimes (2008) for this dataset and is what
# most published C-MAPSS baselines use.
RUL_CLIP = 125


def add_train_rul(df: pd.DataFrame, clip: int | None = RUL_CLIP) -> pd.DataFrame:
    df = df.copy()
    max_cycle = df.groupby("unit_number")["time_cycles"].transform("max")
    rul = max_cycle - df["time_cycles"]
    if clip is not None:
        rul = rul.clip(upper=clip)
    df["RUL"] = rul
    return df
