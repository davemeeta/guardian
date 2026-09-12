"""Train a baseline RUL model (GBM or LSTM) on a C-MAPSS subset, tracked with MLflow.

Usage:
    python scripts/train.py --config configs/gbm_fd001.yaml
"""
import argparse
import os
import sys
from pathlib import Path

# MLflow phones home anonymous usage telemetry from a background thread by
# default. That's already a non-starter for a project whose selling point is
# that nothing leaves the machine. Must be set before mlflow is imported.
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")

import joblib
import mlflow
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from guardian.data.features import FeatureBuilder  # noqa: E402
from guardian.data.labeling import add_train_rul  # noqa: E402
from guardian.data.loader import load_test, load_test_rul, load_train  # noqa: E402
from guardian.data.sequences import make_test_windows, make_train_windows  # noqa: E402
from guardian.data.split import train_val_unit_split  # noqa: E402
from guardian.eval.metrics import imminent_failure_prf, mae, phm08_score, rmse  # noqa: E402
from guardian.eval.plots import plot_error_hist, plot_pred_vs_actual  # noqa: E402
from guardian.simulator.engine_simulator import EngineSimulator  # noqa: E402

# NOTE: lightgbm and torch (MPS backend) are only ever imported inside
# run_gbm()/run_lstm() respectively, never both in the same process — on
# macOS, having both loaded and *actively computing* in one process reliably
# segfaults (an Accelerate/OpenMP native conflict between the two). Since a
# single training run only ever needs one of them, keeping the imports local
# to each branch sidesteps the conflict entirely.

CHECKPOINT_DIR = ROOT / "checkpoints"
PLOTS_DIR = ROOT / "outputs" / "plots"


def augment_fit_split(fit_raw: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Extend the *training* fold with synthetic engines (never the val fold).

    Fitting the simulator on the val split too would leak those engines'
    characteristics into the sampled synthetic population; keeping
    augmentation strictly inside the fit split also means val metrics stay
    directly comparable to a non-augmented run using the same random split.
    The simulator uses unclipped RUL (true degradation progress) regardless
    of whatever rul_clip this run's model config uses for its own labels.
    """
    aug_cfg = config.get("augmentation")
    if not aug_cfg:
        return fit_raw

    subset = config["subset"]
    fit_labeled = add_train_rul(fit_raw, clip=None)
    sim = EngineSimulator().fit(fit_labeled)
    start_id = int(fit_raw["unit_number"].max()) + 1
    synthetic = sim.sample(
        n_engines=aug_cfg["n_engines"],
        start_unit_id=start_id,
        short_life_quantile=aug_cfg.get("short_life_quantile"),
        rate_boost=aug_cfg.get("rate_boost", 1.0),
        random_state=aug_cfg.get("random_state", 0),
    )
    print(f"[{subset}] augmented training fold with {aug_cfg['n_engines']} synthetic engines "
          f"({len(synthetic)} cycles, {len(fit_raw)} real training cycles)")
    return pd.concat([fit_raw, synthetic], ignore_index=True)


def evaluate_and_log(y_true: np.ndarray, y_pred: np.ndarray, tag: str) -> dict:
    metrics = {
        f"{tag}_rmse": rmse(y_true, y_pred),
        f"{tag}_mae": mae(y_true, y_pred),
        f"{tag}_phm08_score": phm08_score(y_true, y_pred),
    }
    prf = imminent_failure_prf(y_true, y_pred)
    metrics.update({f"{tag}_imminent_{k}": v for k, v in prf.items()})
    mlflow.log_metrics(metrics)
    return metrics


def run_gbm(config: dict) -> None:
    from guardian.models.gbm import GBMBaseline, GBMConfig

    subset = config["subset"]
    rul_clip = config.get("rul_clip")
    fit_raw, val_raw = train_val_unit_split(load_train(subset))
    fit_raw = augment_fit_split(fit_raw, config)

    fb = FeatureBuilder(subset)
    fit_df = fb.fit_transform(add_train_rul(fit_raw, clip=rul_clip))
    val_df = fb.transform(add_train_rul(val_raw, clip=rul_clip))

    model = GBMBaseline(GBMConfig(**config["gbm"]))
    model.fit(
        fit_df[fb.feature_cols], fit_df["RUL"],
        eval_set=(val_df[fb.feature_cols], val_df["RUL"]),
    )

    val_pred = model.predict(val_df[fb.feature_cols])
    evaluate_and_log(val_df["RUL"].to_numpy(), val_pred, "val")

    test_df = fb.transform(load_test(subset))
    last_cycle = test_df.loc[test_df.groupby("unit_number")["time_cycles"].idxmax()]
    last_cycle = last_cycle.sort_values("unit_number")
    test_pred = model.predict(last_cycle[fb.feature_cols])
    test_true = load_test_rul(subset).loc[last_cycle["unit_number"]].to_numpy()
    test_metrics = evaluate_and_log(test_true, test_pred, "test")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / f"gbm_{subset}.joblib"
    joblib.dump(model, ckpt_path)
    mlflow.log_artifact(str(ckpt_path))

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    pred_plot = PLOTS_DIR / f"gbm_{subset}_pred_vs_actual.png"
    err_plot = PLOTS_DIR / f"gbm_{subset}_error_hist.png"
    plot_pred_vs_actual(test_true, test_pred, f"GBM {subset} — test", str(pred_plot))
    plot_error_hist(test_true, test_pred, f"GBM {subset} — test error", str(err_plot))
    mlflow.log_artifact(str(pred_plot))
    mlflow.log_artifact(str(err_plot))

    print(f"[{subset} / gbm] test metrics: {test_metrics}")


def run_lstm(config: dict) -> None:
    import torch

    from guardian.models.lstm import LSTMBaseline, LSTMConfig

    subset = config["subset"]
    window_size = config["window_size"]
    rul_clip = config.get("rul_clip")
    fit_raw, val_raw = train_val_unit_split(load_train(subset))
    fit_raw = augment_fit_split(fit_raw, config)

    fb = FeatureBuilder(subset)
    fit_df = fb.fit_transform(add_train_rul(fit_raw, clip=rul_clip))
    val_df = fb.transform(add_train_rul(val_raw, clip=rul_clip))
    X_fit, y_fit = make_train_windows(fit_df, fb.feature_cols, window_size)
    X_val, y_val = make_train_windows(val_df, fb.feature_cols, window_size)

    model = LSTMBaseline(n_features=len(fb.feature_cols), config=LSTMConfig(**config["lstm"]))
    model.fit(X_fit, y_fit)

    val_pred = model.predict(X_val)
    evaluate_and_log(y_val, val_pred, "val")

    test_df = fb.transform(load_test(subset))
    X_test, test_units = make_test_windows(test_df, fb.feature_cols, window_size)
    test_pred = model.predict(X_test)
    test_true = load_test_rul(subset).loc[test_units].to_numpy()
    test_metrics = evaluate_and_log(test_true, test_pred, "test")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / f"lstm_{subset}.pt"
    torch.save(model.model.state_dict(), ckpt_path)
    mlflow.log_artifact(str(ckpt_path))

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    pred_plot = PLOTS_DIR / f"lstm_{subset}_pred_vs_actual.png"
    err_plot = PLOTS_DIR / f"lstm_{subset}_error_hist.png"
    plot_pred_vs_actual(test_true, test_pred, f"LSTM {subset} — test", str(pred_plot))
    plot_error_hist(test_true, test_pred, f"LSTM {subset} — test error", str(err_plot))
    mlflow.log_artifact(str(pred_plot))
    mlflow.log_artifact(str(err_plot))

    print(f"[{subset} / lstm] test metrics: {test_metrics}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())

    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment("guardian-baseline-rul")

    suffix = f"_{config['run_tag']}" if config.get("run_tag") else ""
    run_name = f"{config['model']}_{config['subset']}{suffix}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(_flatten(config))
        if config["model"] == "gbm":
            run_gbm(config)
        elif config["model"] == "lstm":
            run_lstm(config)
        else:
            raise ValueError(f"Unknown model type: {config['model']}")


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, prefix=f"{key}."))
        else:
            out[key] = v
    return out


if __name__ == "__main__":
    main()
