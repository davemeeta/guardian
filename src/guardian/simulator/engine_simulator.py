"""Physics-informed digital-twin degradation simulator (single-regime subsets).

Design: a real fault (e.g. HPC wear) propagates through several correlated
sensor readings at once, not independently per sensor. So each synthetic
engine gets ONE latent "decline rate" driving every sensor's curve together,
plus a per-sensor manufacturing offset (the calibration variation already
visible at the start of life, before degradation begins), plus i.i.d.
per-cycle sensor noise. All four ingredients — curve shape, decline-rate
distribution, offset distribution, noise level — are estimated from real
training data in `fit`, then resampled in `sample` to synthesize complete
run-to-failure trajectories for engines that never existed.

Scoped to single-regime subsets (FD001/FD003): op settings are sampled as
i.i.d. noise around their real mean, which assumes there's only one regime
to be near. FD002/FD004 would additionally need a sampled regime sequence.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from guardian.data.columns import ALL_COLS, CONSTANT_SENSORS_FD001, OP_SETTING_COLS, SENSOR_COLS
from guardian.data.labeling import RUL_CLIP
from guardian.simulator.degradation_curves import SensorCurve, fit_sensor_curve
from guardian.simulator.health_index import degradation_progress

TRENDING_SENSORS = [c for c in SENSOR_COLS if c not in CONSTANT_SENSORS_FD001]
RATE_BOUNDS = (0.3, 3.0)


@dataclass
class EngineSimulator:
    rul_clip: int = RUL_CLIP
    curves: dict[str, SensorCurve] = field(default_factory=dict)
    noise_std: dict[str, float] = field(default_factory=dict)
    offset_std: dict[str, float] = field(default_factory=dict)
    rate_multipliers: np.ndarray = field(default_factory=lambda: np.array([1.0]))
    lifetimes: np.ndarray = field(default_factory=lambda: np.array([200]))
    flat_sensor_stats: dict[str, tuple[float, float]] = field(default_factory=dict)
    op_setting_stats: dict[str, tuple[float, float]] = field(default_factory=dict)

    def fit(self, df: pd.DataFrame) -> "EngineSimulator":
        """df must have an unclipped 'RUL' column (see labeling.add_train_rul)."""
        d_all = degradation_progress(df["RUL"].to_numpy(), self.rul_clip)

        for sensor in TRENDING_SENSORS:
            self.curves[sensor] = fit_sensor_curve(d_all, df[sensor].to_numpy())

        rates, lifetimes = [], []
        offsets = {s: [] for s in TRENDING_SENSORS}
        resid_diffs = {s: [] for s in TRENDING_SENSORS}

        for _, unit_df in df.groupby("unit_number"):
            unit_df = unit_df.sort_values("time_cycles")
            d_unit = degradation_progress(unit_df["RUL"].to_numpy(), self.rul_clip)
            lifetimes.append(len(unit_df))

            rate = self.fit_unit_rate(d_unit, unit_df)
            rates.append(rate)

            healthy = d_unit < 0.05
            for sensor in TRENDING_SENSORS:
                y = unit_df[sensor].to_numpy()
                baseline = self.curves[sensor].a
                offset = (y[healthy].mean() - baseline) if healthy.any() else 0.0
                offsets[sensor].append(offset)
                pred = self.curves[sensor](np.clip(d_unit * rate, 0, 1)) + offset
                residual = y - pred
                # First-differencing before estimating noise: the curve fit
                # is a coarse fit (one shared rate for all sensors, one
                # offset per sensor), so the raw residual still carries some
                # low-frequency curve-misfit alongside the true high-
                # frequency sensor noise. Consecutive-cycle differences
                # cancel that slow-varying part and isolate the i.i.d. noise
                # — using the raw residual std here measurably overstated
                # noise and made synthetic trajectories visibly fuzzier than
                # real ones.
                resid_diffs[sensor].extend(np.diff(residual).tolist())

        self.rate_multipliers = np.array(rates)
        self.lifetimes = np.array(lifetimes)
        for sensor in TRENDING_SENSORS:
            self.offset_std[sensor] = float(np.std(offsets[sensor]))
            self.noise_std[sensor] = float(np.std(resid_diffs[sensor]) / np.sqrt(2))

        for sensor in CONSTANT_SENSORS_FD001:
            if sensor in df.columns:
                self.flat_sensor_stats[sensor] = (df[sensor].mean(), df[sensor].std())
        for col in OP_SETTING_COLS:
            self.op_setting_stats[col] = (df[col].mean(), df[col].std())
        return self

    def fit_unit_rate(self, d_unit: np.ndarray, unit_df: pd.DataFrame) -> float:
        """Single latent decline-rate for this unit, fit jointly across all
        trending sensors (one physical fault, many correlated symptoms)."""

        def loss(rate: float) -> float:
            d_scaled = np.clip(d_unit * rate, 0, 1)
            total = 0.0
            for sensor in TRENDING_SENSORS:
                pred = self.curves[sensor](d_scaled)
                total += np.sum((unit_df[sensor].to_numpy() - pred) ** 2)
            return total

        result = minimize_scalar(loss, bounds=RATE_BOUNDS, method="bounded")
        return float(result.x)

    def sample(
        self,
        n_engines: int,
        start_unit_id: int,
        short_life_quantile: float | None = None,
        rate_boost: float = 1.0,
        random_state: int = 0,
    ) -> pd.DataFrame:
        """Synthesize n_engines complete run-to-failure trajectories.

        short_life_quantile: if set (e.g. 0.4), lifetimes are bootstrapped
        only from the shortest `short_life_quantile` fraction of real
        lifetimes, directly biasing the synthetic set toward fast-failing
        engines — the ones that contribute the most near-failure (rare
        imminent-failure) cycles per unit of generated data.
        rate_boost: multiplies sampled decline rates (>1 = faster decline),
        a second, independent knob toward the same goal.
        """
        rng = np.random.default_rng(random_state)
        lifetime_pool = self.lifetimes
        if short_life_quantile is not None:
            cutoff = np.quantile(self.lifetimes, short_life_quantile)
            lifetime_pool = self.lifetimes[self.lifetimes <= cutoff]

        rows = []
        for i in range(n_engines):
            unit_id = start_unit_id + i
            lifetime = int(rng.choice(lifetime_pool))
            rate = float(np.clip(rng.choice(self.rate_multipliers) * rate_boost, *RATE_BOUNDS))
            offsets = {
                s: rng.normal(0, self.offset_std[s]) for s in TRENDING_SENSORS
            }

            cycles = np.arange(1, lifetime + 1)
            rul = lifetime - cycles
            d = np.clip(degradation_progress(rul, self.rul_clip) * rate, 0, 1)

            unit_rows = pd.DataFrame({"unit_number": unit_id, "time_cycles": cycles})
            for col in OP_SETTING_COLS:
                mean, std = self.op_setting_stats[col]
                unit_rows[col] = rng.normal(mean, std, size=lifetime)
            for sensor in TRENDING_SENSORS:
                curve = self.curves[sensor]
                noise = rng.normal(0, self.noise_std[sensor], size=lifetime)
                unit_rows[sensor] = curve(d) + offsets[sensor] + noise
            for sensor, (mean, std) in self.flat_sensor_stats.items():
                unit_rows[sensor] = rng.normal(mean, std, size=lifetime)

            rows.append(unit_rows[ALL_COLS])

        return pd.concat(rows, ignore_index=True)
