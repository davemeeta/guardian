"""Validation-driven sweep over augmentation intensity for FD001.

Fits the simulator once on the real training fold, then for a grid of
(n_engines, short_life_quantile) samples a synthetic set, trains the model
on real+synthetic, and evaluates on the (always-real) validation fold. The
official test set is never touched here — this is purely for picking a
good augmentation config, the same way you'd tune any other hyperparameter.

Usage:
    python scripts/sweep_augmentation.py --model gbm
    python scripts/sweep_augmentation.py --model lstm
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")

import matplotlib.pyplot as plt
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from guardian.data.features import FeatureBuilder  # noqa: E402
from guardian.data.labeling import add_train_rul  # noqa: E402
from guardian.data.loader import load_train  # noqa: E402
from guardian.data.sequences import make_train_windows  # noqa: E402
from guardian.data.split import train_val_unit_split  # noqa: E402
from guardian.eval.metrics import imminent_failure_prf, mae, rmse  # noqa: E402
from guardian.simulator.engine_simulator import EngineSimulator  # noqa: E402

SUBSET = "FD001"
RUL_CLIP = 125
N_ENGINES_GRID = {
    "gbm": [10, 20, 30, 40, 50, 75, 100],
    "lstm": [25, 50, 75, 100],
}
SHORT_LIFE_GRID = [None, 0.5, 0.3]
PLOTS_DIR = ROOT / "outputs" / "plots"


def build_fold(fit_raw: pd.DataFrame, val_raw: pd.DataFrame, subset: str):
    fb = FeatureBuilder(subset)
    fit_df = fb.fit_transform(add_train_rul(fit_raw, clip=RUL_CLIP))
    val_df = fb.transform(add_train_rul(val_raw, clip=RUL_CLIP))
    return fb, fit_df, val_df


def eval_gbm(fit_df, val_df, feature_cols) -> dict:
    from guardian.models.gbm import GBMBaseline, GBMConfig

    cfg = yaml.safe_load((ROOT / "configs" / "gbm_fd001.yaml").read_text())["gbm"]
    model = GBMBaseline(GBMConfig(**cfg))
    model.fit(
        fit_df[feature_cols], fit_df["RUL"],
        eval_set=(val_df[feature_cols], val_df["RUL"]),
    )
    pred = model.predict(val_df[feature_cols])
    return score(val_df["RUL"].to_numpy(), pred)


def eval_lstm(fit_df, val_df, feature_cols) -> dict:
    from guardian.models.lstm import LSTMBaseline, LSTMConfig

    cfg = yaml.safe_load((ROOT / "configs" / "lstm_fd001.yaml").read_text())
    window_size = cfg["window_size"]
    X_fit, y_fit = make_train_windows(fit_df, feature_cols, window_size)
    X_val, y_val = make_train_windows(val_df, feature_cols, window_size)
    model = LSTMBaseline(n_features=len(feature_cols), config=LSTMConfig(**cfg["lstm"]))
    model.fit(X_fit, y_fit)
    pred = model.predict(X_val)
    return score(y_val, pred)


def score(y_true, y_pred) -> dict:
    prf = imminent_failure_prf(y_true, y_pred)
    return {"val_rmse": rmse(y_true, y_pred), "val_mae": mae(y_true, y_pred), **{f"val_{k}": v for k, v in prf.items()}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["gbm", "lstm"], required=True)
    args = parser.parse_args()
    evaluator = eval_gbm if args.model == "gbm" else eval_lstm

    real_raw = load_train(SUBSET)
    fit_raw, val_raw = train_val_unit_split(real_raw)

    print("Fitting simulator on real training fold...")
    sim = EngineSimulator().fit(add_train_rul(fit_raw, clip=None))
    start_id = int(fit_raw["unit_number"].max()) + 1

    rows = []

    print(f"[{args.model}] baseline (no augmentation)...")
    fb, fit_df, val_df = build_fold(fit_raw, val_raw, SUBSET)
    baseline_metrics = evaluator(fit_df, val_df, fb.feature_cols)
    rows.append({"n_engines": 0, "short_life_quantile": "none", **baseline_metrics})
    print(f"  {baseline_metrics}")

    for n_engines in N_ENGINES_GRID[args.model]:
        for quantile in SHORT_LIFE_GRID:
            synthetic = sim.sample(
                n_engines=n_engines, start_unit_id=start_id,
                short_life_quantile=quantile, random_state=0,
            )
            aug_fit_raw = pd.concat([fit_raw, synthetic], ignore_index=True)
            fb, fit_df, val_df = build_fold(aug_fit_raw, val_raw, SUBSET)
            metrics = evaluator(fit_df, val_df, fb.feature_cols)
            label = f"n_engines={n_engines} short_life_quantile={quantile}"
            print(f"[{args.model}] {label}: {metrics}")
            rows.append({"n_engines": n_engines, "short_life_quantile": str(quantile), **metrics})

    results = pd.DataFrame(rows)
    out_csv = ROOT / "outputs" / f"sweep_{args.model}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_csv, index=False)

    best = results.sort_values("val_rmse").iloc[0]
    print(f"\nBest by val_rmse: {best.to_dict()}")
    print(f"Baseline was: {rows[0]}")
    print(f"Saved full sweep to {out_csv}")

    fig, ax = plt.subplots(figsize=(7, 5))
    augmented = results[results["n_engines"] > 0]  # baseline drawn separately below
    for quantile, group in augmented.groupby("short_life_quantile"):
        group = group.sort_values("n_engines")
        ax.plot(group["n_engines"], group["val_rmse"], marker="o", label=f"short_life_quantile={quantile}")
    ax.axhline(baseline_metrics["val_rmse"], color="gray", linestyle="--", label="baseline (no aug)")
    ax.set_xlabel("n_engines")
    ax.set_ylabel("validation RMSE")
    ax.set_title(f"{args.model.upper()} on FD001: augmentation sweep (validation fold)")
    ax.legend()
    fig.tight_layout()
    plot_path = PLOTS_DIR / f"sweep_{args.model}_val_rmse.png"
    fig.savefig(plot_path, dpi=150)
    print(f"Saved sweep plot to {plot_path}")


if __name__ == "__main__":
    main()
