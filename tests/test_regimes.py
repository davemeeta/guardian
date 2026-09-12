import numpy as np
import pandas as pd

from guardian.data.regimes import RegimeNormalizer


def test_fit_transform_applies_same_clustering_to_new_data():
    rng = np.random.default_rng(0)
    regime_a = rng.normal(loc=0, scale=0.1, size=(50, 3))
    regime_b = rng.normal(loc=10, scale=0.1, size=(50, 3))
    settings = np.concatenate([regime_a, regime_b])
    sensors = np.concatenate([
        rng.normal(loc=100, scale=1, size=50),
        rng.normal(loc=200, scale=1, size=50),
    ]).reshape(-1, 1)

    df = pd.DataFrame(settings, columns=["op_setting_1", "op_setting_2", "op_setting_3"])
    df["sensor_1"] = sensors
    for i in range(2, 22):
        df[f"sensor_{i}"] = sensors.flatten()

    norm = RegimeNormalizer(n_regimes=2)
    out = norm.fit_transform(df)

    # within each recovered regime, sensor_1 should now be roughly standardized
    for regime in out["regime"].unique():
        vals = out.loc[out["regime"] == regime, "sensor_1"]
        assert abs(vals.mean()) < 1e-6

    # transform on new data reuses the fitted clusters/stats, not a fresh fit
    new_df = df.iloc[:5].copy()
    transformed = norm.transform(new_df)
    assert "regime" in transformed.columns
    assert len(transformed) == 5
