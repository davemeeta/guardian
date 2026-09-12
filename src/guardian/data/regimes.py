import pandas as pd
from sklearn.cluster import KMeans

from guardian.data.columns import OP_SETTING_COLS, SENSOR_COLS

# FD002/FD004 cycle through six discrete operating regimes (combinations of
# altitude/Mach/throttle). A sensor's raw value is dominated by which regime
# the engine is currently in, which drowns out the much smaller degradation
# trend. Clustering on the operational settings recovers the regime label,
# and normalizing sensors within each regime removes that confound. FD001/
# FD003 run a single regime, where this is a no-op (one cluster).
N_REGIMES = 6


class RegimeNormalizer:
    """Fit regime clusters + per-regime sensor stats on train; apply to any split.

    Fitting and applying must use the same clustering/normalization params,
    otherwise "regime 3" in train and "regime 3" in test could refer to
    different operating conditions and the z-scores would be incomparable.
    """

    def __init__(self, n_regimes: int = N_REGIMES):
        self.n_regimes = n_regimes
        self.kmeans = KMeans(n_clusters=n_regimes, n_init=10, random_state=0)
        self._regime_mean: pd.DataFrame | None = None
        self._regime_std: pd.DataFrame | None = None

    def fit(self, df: pd.DataFrame) -> "RegimeNormalizer":
        regimes = self.kmeans.fit_predict(df[OP_SETTING_COLS])
        stats = df[SENSOR_COLS].groupby(regimes)
        self._regime_mean = stats.mean()
        self._regime_std = stats.std().replace(0, 1.0)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        regimes = self.kmeans.predict(df[OP_SETTING_COLS])
        df["regime"] = regimes
        mean = self._regime_mean.loc[regimes].reset_index(drop=True)
        std = self._regime_std.loc[regimes].reset_index(drop=True)
        mean.index = df.index
        std.index = df.index
        df[SENSOR_COLS] = (df[SENSOR_COLS] - mean) / std
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
