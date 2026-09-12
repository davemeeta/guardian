import pandas as pd
from sklearn.preprocessing import StandardScaler

from guardian.data.columns import CONSTANT_SENSORS_FD001, OP_SETTING_COLS, SENSOR_COLS
from guardian.data.regimes import RegimeNormalizer

MULTI_REGIME_SUBSETS = {"FD002", "FD004"}


class FeatureBuilder:
    """Subset-appropriate preprocessing, fit on train and applied to test.

    Single-regime subsets (FD001/FD003): drop sensors that are constant
    throughout (no signal there), then z-score the rest using train
    statistics. Raw sensor units span wildly different ranges (temperatures
    in the hundreds, pressure ratios near 1), and while tree splits are
    invariant to that, the LSTM is not — gradient descent over unscaled
    features converges poorly and was observed to collapse to near-constant
    predictions without this step.
    Multi-regime subsets (FD002/FD004): cluster operating settings into
    regimes (fit on train) and z-score sensors within each regime, which
    both removes the regime confound and leaves features on a comparable
    scale for free.
    """

    def __init__(self, subset: str):
        self.subset = subset
        self.is_multi_regime = subset in MULTI_REGIME_SUBSETS
        self.normalizer = RegimeNormalizer() if self.is_multi_regime else None
        self.scaler = StandardScaler() if not self.is_multi_regime else None
        self.dropped_sensors: list[str] = []
        self.feature_cols: list[str] = []

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.is_multi_regime:
            df = self.normalizer.fit_transform(df)
            self.feature_cols = OP_SETTING_COLS + SENSOR_COLS
            return df
        self.dropped_sensors = [c for c in CONSTANT_SENSORS_FD001 if c in df.columns]
        df = df.drop(columns=self.dropped_sensors).copy()
        self.feature_cols = [c for c in OP_SETTING_COLS + SENSOR_COLS if c not in self.dropped_sensors]
        df[self.feature_cols] = self.scaler.fit_transform(df[self.feature_cols])
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.is_multi_regime:
            return self.normalizer.transform(df)
        df = df.drop(columns=self.dropped_sensors).copy()
        df[self.feature_cols] = self.scaler.transform(df[self.feature_cols])
        return df
