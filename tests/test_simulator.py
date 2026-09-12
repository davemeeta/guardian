import numpy as np
import pandas as pd
import pytest

from guardian.data.columns import ALL_COLS, OP_SETTING_COLS
from guardian.data.labeling import add_train_rul
from guardian.simulator.engine_simulator import TRENDING_SENSORS, EngineSimulator
from guardian.simulator.health_index import degradation_progress


def _synthetic_ground_truth(n_units=15, rng=None):
    """Hand-rolled run-to-failure data with a KNOWN degradation curve, so we
    can check that fitting recovers something close to the truth."""
    rng = rng or np.random.default_rng(0)
    rows = []
    true_a, true_b, true_k = 500.0, 2.0, 2.0
    for unit in range(1, n_units + 1):
        lifetime = rng.integers(150, 250)
        rate = rng.uniform(0.8, 1.2)
        cycles = np.arange(1, lifetime + 1)
        rul = lifetime - cycles
        d = np.clip(degradation_progress(rul, clip=125) * rate, 0, 1)
        sensor_2 = true_a + true_b * np.expm1(true_k * d) + rng.normal(0, 0.3, lifetime)
        unit_df = pd.DataFrame({"unit_number": unit, "time_cycles": cycles})
        for col in OP_SETTING_COLS:
            unit_df[col] = rng.normal(0, 0.001, lifetime)
        for sensor in TRENDING_SENSORS:
            unit_df[sensor] = sensor_2 if sensor == "sensor_2" else rng.normal(500, 1, lifetime)
        for col in ALL_COLS:
            if col not in unit_df.columns:
                unit_df[col] = 100.0
        rows.append(unit_df[ALL_COLS])
    return pd.concat(rows, ignore_index=True)


def test_fit_recovers_known_curve_shape():
    df = add_train_rul(_synthetic_ground_truth(), clip=None)
    sim = EngineSimulator().fit(df)
    curve = sim.curves["sensor_2"]
    assert curve.a == pytest.approx(500.0, abs=5)
    assert curve.k == pytest.approx(2.0, abs=0.5)


def test_sample_output_schema_and_lengths():
    df = add_train_rul(_synthetic_ground_truth(), clip=None)
    sim = EngineSimulator().fit(df)
    synthetic = sim.sample(n_engines=5, start_unit_id=1000, random_state=1)

    assert list(synthetic.columns) == ALL_COLS
    assert set(synthetic["unit_number"]) == set(range(1000, 1005))
    for _, unit_df in synthetic.groupby("unit_number"):
        assert unit_df["time_cycles"].tolist() == list(range(1, len(unit_df) + 1))


def test_short_life_quantile_biases_toward_shorter_engines():
    df = add_train_rul(_synthetic_ground_truth(n_units=40), clip=None)
    sim = EngineSimulator().fit(df)

    unbiased = sim.sample(n_engines=30, start_unit_id=2000, random_state=2)
    biased = sim.sample(n_engines=30, start_unit_id=3000, short_life_quantile=0.3, random_state=2)

    unbiased_mean_life = unbiased.groupby("unit_number").size().mean()
    biased_mean_life = biased.groupby("unit_number").size().mean()
    assert biased_mean_life < unbiased_mean_life
