"""ML/inference service: reports current model health and can trigger a
real retraining run.

Deliberately thin — retraining shells out to the same scripts/train.py used
everywhere else in this project (as its own subprocess, not an in-process
import), so this service triggers exactly the same training path a human
would run from the command line, with no duplicated logic and no risk of
the LightGBM/PyTorch native conflict documented in scripts/train.py meeting
FastAPI's own already-imported libraries in one process.

Run: uvicorn guardian.api.ml_service:app --host 0.0.0.0 --port 8000
"""
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")

import mlflow  # noqa: E402
import pandas as pd  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_NAME = "guardian-baseline-rul"

app = FastAPI(title="Guardian ML Service")


def _tracking_uri() -> str:
    return f"sqlite:///{ROOT / 'mlflow.db'}"


class RetrainRequest(BaseModel):
    config: str = "gbm_fd001"  # matches a file in configs/<config>.yaml


@app.get("/health")
def health() -> dict:
    """Latest known metrics per run name, straight from MLflow — this is
    the "model health over time" data source the dashboard also reads."""
    mlflow.set_tracking_uri(_tracking_uri())
    try:
        runs = mlflow.search_runs(experiment_names=[EXPERIMENT_NAME])
    except Exception:
        return {"status": "no runs yet", "runs": []}

    if runs.empty:
        return {"status": "no runs yet", "runs": []}

    runs = runs.sort_values("start_time", ascending=False)
    latest_per_name = runs.drop_duplicates(subset=["tags.mlflow.runName"])
    metric_cols = [c for c in runs.columns if c.startswith("metrics.test_")]

    summary = []
    for _, row in latest_per_name.iterrows():
        entry = {"run_name": row["tags.mlflow.runName"], "start_time": str(row["start_time"])}
        for col in metric_cols:
            if pd.notna(row[col]):
                # every metric_cols entry is already scoped to "test_" (see
                # the filter above), so strip that too — "rmse" reads
                # cleaner than "test_rmse" when the endpoint only ever
                # reports test metrics in the first place.
                entry[col.replace("metrics.test_", "")] = row[col]
        summary.append(entry)

    return {"status": "ok", "runs": summary}


@app.post("/retrain")
def retrain(req: RetrainRequest) -> dict:
    """Triggers a real training run via scripts/train.py, synchronously.
    Blocks until done (GBM: seconds; LSTM: up to a minute or two) — fine
    for a demo/CI trigger, not meant for high-frequency calls."""
    config_path = ROOT / "configs" / f"{req.config}.yaml"
    if not config_path.exists():
        raise HTTPException(status_code=404, detail=f"No such config: {req.config}")

    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "train.py"), "--config", str(config_path)],
        capture_output=True, text=True, cwd=str(ROOT), timeout=900,
    )
    if result.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail={"config": req.config, "stdout": result.stdout[-4000:], "stderr": result.stderr[-4000:]},
        )
    return {"status": "trained", "config": req.config, "stdout_tail": result.stdout[-2000:]}
