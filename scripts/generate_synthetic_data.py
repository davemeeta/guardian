"""Fit the digital-twin simulator on real FD001 data, sample synthetic
engines, and save plausibility-check plots comparing real vs. synthetic
trajectories and lifetime distributions.

Usage:
    python scripts/generate_synthetic_data.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from guardian.data.labeling import add_train_rul  # noqa: E402
from guardian.data.loader import load_train  # noqa: E402
from guardian.simulator.engine_simulator import EngineSimulator  # noqa: E402

PLOTS_DIR = ROOT / "outputs" / "plots"


def main() -> None:
    real = add_train_rul(load_train("FD001"), clip=None)
    sim = EngineSimulator().fit(real)

    synthetic = sim.sample(n_engines=20, start_unit_id=1000, random_state=42)
    synthetic_short = sim.sample(
        n_engines=20, start_unit_id=2000, short_life_quantile=0.3, random_state=42
    )

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Real vs. synthetic trajectories, a few sensors, plotted against
    #    cycles-to-failure so real and synthetic engines of different
    #    lifetimes line up at the point that matters (the end).
    sensors = ["sensor_2", "sensor_4", "sensor_7", "sensor_11"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    real_units = real["unit_number"].unique()[:15]
    synth_units = synthetic["unit_number"].unique()

    for ax, sensor in zip(axes.flat, sensors):
        for u in real_units:
            udf = real[real["unit_number"] == u]
            ax.plot(-udf["RUL"], udf[sensor], color="C0", alpha=0.3, linewidth=0.8)
        for u in synth_units:
            udf = synthetic[synthetic["unit_number"] == u]
            cycles_to_failure = -(len(udf) - udf["time_cycles"])
            ax.plot(cycles_to_failure, udf[sensor], color="C1", alpha=0.5, linewidth=0.8)
        ax.set_title(sensor)
        ax.set_xlabel("cycles to failure")
    axes.flat[0].plot([], [], color="C0", label="real")
    axes.flat[0].plot([], [], color="C1", label="synthetic")
    axes.flat[0].legend()
    fig.suptitle("Real vs. synthetic degradation trajectories (FD001)")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "simulator_real_vs_synthetic_trajectories.png", dpi=150)
    plt.close(fig)

    # 2. Lifetime distributions: real, synthetic (unbiased), synthetic
    #    (short-life-biased) — shows the bias knob actually does something.
    fig, ax = plt.subplots(figsize=(6, 4))
    real_life = real.groupby("unit_number").size()
    synth_life = synthetic.groupby("unit_number").size()
    synth_short_life = synthetic_short.groupby("unit_number").size()
    bins = np.linspace(50, 400, 25)
    ax.hist(real_life, bins=bins, alpha=0.5, label="real", density=True)
    ax.hist(synth_life, bins=bins, alpha=0.5, label="synthetic (unbiased)", density=True)
    ax.hist(synth_short_life, bins=bins, alpha=0.5, label="synthetic (short_life_quantile=0.3)", density=True)
    ax.set_xlabel("engine lifetime (cycles)")
    ax.legend()
    ax.set_title("Lifetime distribution: real vs. synthetic")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "simulator_lifetime_distributions.png", dpi=150)
    plt.close(fig)

    print(f"Fit on {real['unit_number'].nunique()} real engines.")
    print(f"Rate multiplier range: [{sim.rate_multipliers.min():.2f}, {sim.rate_multipliers.max():.2f}]")
    print(f"Real lifetime range: [{real_life.min()}, {real_life.max()}], mean {real_life.mean():.1f}")
    print(f"Saved plausibility plots to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
