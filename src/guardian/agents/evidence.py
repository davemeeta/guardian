"""Builds the structured evidence packet the agents reason over.

Two complementary, model-independent signals: retrospective performance
(RMSE/imminent-rate on engines that have since reached failure) and live
sensor drift (early-life readings vs. the simulator's calibrated curves).
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from guardian.data.labeling import add_train_rul
from guardian.eval.metrics import rmse
from guardian.simulator.engine_simulator import TRENDING_SENSORS, EngineSimulator
from guardian.simulator.health_index import degradation_progress

N_HEALTHY_CYCLES = 5  # first N cycles of a unit used as its "healthy baseline" proxy
N_DRIFT_SENSORS_REPORTED = 3


def _ordinal(n: int) -> str:
    if 11 <= n % 100 <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


@dataclass
class Evidence:
    scenario_id: str
    n_units: int
    baseline_rmse: float
    current_rmse: float
    rmse_delta_pct: float
    baseline_imminent_rate: float
    current_imminent_rate: float
    sensor_drift: dict[str, float]  # sensor -> z-score of healthy-phase mean shift
    decline_rate_value: float | None
    decline_rate_percentile: float | None
    top_drifted_sensors: list[tuple[str, float]] = field(init=False)

    def __post_init__(self):
        self.top_drifted_sensors = sorted(
            self.sensor_drift.items(), key=lambda kv: abs(kv[1]), reverse=True
        )[:N_DRIFT_SENSORS_REPORTED]

    @property
    def rmse_may_be_misleading(self) -> bool:
        """True when sensor drift is large but RMSE looks fine or improved.
        Tree models can't extrapolate past their training range, so a
        far-out-of-range input can collapse into a stable-looking
        prediction rather than an obviously bad one — observed directly
        here (a ~4 SD sensor shift *improved* RMSE by 20%+)."""
        max_drift = max((abs(z) for z in self.sensor_drift.values()), default=0.0)
        return max_drift >= 3.0 and self.rmse_delta_pct < 10.0

    def to_prompt_text(self) -> str:
        lines = [
            f"BATCH: {self.scenario_id} ({self.n_units} engines with completed run-to-failure data)",
            "",
            "MODEL PERFORMANCE (retrospective, now that true failure times are known):",
            f"- Historical validation RMSE: {self.baseline_rmse:.2f} cycles",
            f"- This batch's RMSE: {self.current_rmse:.2f} cycles ({self.rmse_delta_pct:+.1f}% vs. historical)",
            f"- Historical rate of \"imminent failure\" cycles (RUL<=20) in the data: {self.baseline_imminent_rate:.1%}",
            f"- This batch's rate: {self.current_imminent_rate:.1%}",
            "",
            "SENSOR DRIFT (early-life \"healthy\" readings vs. training baseline, in standard deviations):",
        ]
        for sensor, z in self.top_drifted_sensors:
            lines.append(f"- {sensor}: {z:+.1f} SD")
        if self.rmse_may_be_misleading:
            lines.append(
                "- CAVEAT: sensor drift here is large enough that RMSE may be an unreliable "
                "signal on its own. Tree-based models cannot extrapolate past their training "
                "range, so a sensor value far outside anything seen in training can look "
                "stable (or even improve) in the RMSE number even when the model is really "
                "just extrapolating blindly. Weigh the sensor drift and digital-twin "
                "comparison below more heavily than RMSE in this case."
            )
        lines.append("")
        lines.append("DIGITAL-TWIN COMPARISON:")
        if self.decline_rate_value is not None:
            pct = self.decline_rate_percentile
            if pct >= 90:
                verdict = "unusually fast — near the extreme edge of engines seen historically"
            elif pct <= 10:
                verdict = "unusually slow — near the extreme edge of engines seen historically"
            elif 35 <= pct <= 65:
                verdict = "unremarkable — squarely in the middle of normal historical variation"
            else:
                verdict = "somewhat outside the middle of normal historical variation, but not extreme"
            lines.append(
                f"- This batch's engines degrade at {self.decline_rate_value:.2f}x the typical "
                f"calibrated rate (1.0x = average)."
            )
            lines.append(
                f"- Historical percentile: {_ordinal(round(pct))} out of 100 real calibration engines, ranked "
                f"from slowest- to fastest-degrading. Verdict: {verdict}."
            )
        else:
            lines.append("- Not enough completed trajectories in this batch to estimate a decline rate.")
        return "\n".join(lines)


def evaluate_batch(window_df: pd.DataFrame, model, fb, rul_clip: int = 125) -> tuple[float, float, pd.DataFrame]:
    """RMSE and imminent-failure rate over every cycle in window_df, RUL
    clipped the same way training labels were. Also used to compute the
    "baseline" itself (see scenarios.build_context) so a batch and its
    baseline are always produced by the exact same procedure."""
    labeled = add_train_rul(window_df, clip=None)
    labeled["RUL"] = labeled["RUL"].clip(upper=rul_clip)
    transformed = fb.transform(labeled)
    preds = model.predict(transformed[fb.feature_cols])
    true_rul = labeled["RUL"].to_numpy()
    return rmse(true_rul, preds), float((true_rul <= 20).mean()), labeled


def build_evidence(
    scenario_id: str,
    window_df: pd.DataFrame,
    model,
    fb,
    baseline_rmse: float,
    baseline_imminent_rate: float,
    sim: EngineSimulator,
    rul_clip: int = 125,
) -> Evidence:
    """window_df: RAW sensor columns for a batch of COMPLETED engines
    (full run-to-failure trajectories). `baseline_rmse`/`baseline_imminent_rate`
    must come from `evaluate_batch` too (see scenarios.build_context) — a
    baseline computed some other way bakes in whatever gap exists between
    the two data sources, which looks like drift even when there isn't any.
    """
    current_rmse, current_imminent_rate, labeled = evaluate_batch(window_df, model, fb, rul_clip)
    labeled_unclipped = add_train_rul(window_df, clip=None)

    healthy = labeled.groupby("unit_number").head(N_HEALTHY_CYCLES)
    sensor_drift = {}
    for sensor in TRENDING_SENSORS:
        if sensor in sim.curves and sim.offset_std.get(sensor, 0) > 0:
            z = (healthy[sensor].mean() - sim.curves[sensor].a) / sim.offset_std[sensor]
            sensor_drift[sensor] = float(z)

    rates = []
    for _, unit_df in labeled_unclipped.groupby("unit_number"):
        unit_df = unit_df.sort_values("time_cycles")
        if unit_df["RUL"].min() <= 1:
            d = degradation_progress(unit_df["RUL"].to_numpy(), sim.rul_clip)
            rates.append(sim.fit_unit_rate(d, unit_df))
    decline_rate_value = float(np.mean(rates)) if rates else None
    decline_rate_percentile = (
        float((sim.rate_multipliers < decline_rate_value).mean() * 100) if rates else None
    )

    return Evidence(
        scenario_id=scenario_id,
        n_units=labeled["unit_number"].nunique(),
        baseline_rmse=baseline_rmse,
        current_rmse=current_rmse,
        rmse_delta_pct=(current_rmse - baseline_rmse) / baseline_rmse * 100,
        baseline_imminent_rate=baseline_imminent_rate,
        current_imminent_rate=current_imminent_rate,
        sensor_drift=sensor_drift,
        decline_rate_value=decline_rate_value,
        decline_rate_percentile=decline_rate_percentile,
    )
