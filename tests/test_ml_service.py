import pandas as pd
import pytest
from fastapi.testclient import TestClient

from guardian.api import ml_service


@pytest.fixture
def client():
    return TestClient(ml_service.app)


def test_health_reports_no_runs_when_mlflow_empty(client, monkeypatch):
    monkeypatch.setattr(ml_service.mlflow, "search_runs", lambda experiment_names: pd.DataFrame())
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "no runs yet", "runs": []}


def test_health_reports_no_runs_when_experiment_missing(client, monkeypatch):
    def raise_missing(experiment_names):
        raise Exception("experiment not found")

    monkeypatch.setattr(ml_service.mlflow, "search_runs", raise_missing)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "no runs yet"


def test_health_summarizes_latest_run_per_name(client, monkeypatch):
    df = pd.DataFrame([
        {"tags.mlflow.runName": "gbm_FD001", "start_time": pd.Timestamp("2026-01-02"), "metrics.test_rmse": 17.8},
        {"tags.mlflow.runName": "gbm_FD001", "start_time": pd.Timestamp("2026-01-01"), "metrics.test_rmse": 20.0},
    ])
    monkeypatch.setattr(ml_service.mlflow, "search_runs", lambda experiment_names: df)
    resp = client.get("/health")
    body = resp.json()
    assert body["status"] == "ok"
    assert len(body["runs"]) == 1
    assert body["runs"][0]["run_name"] == "gbm_FD001"
    assert body["runs"][0]["rmse"] == 17.8  # the more recent of the two rows


def test_retrain_rejects_unknown_config(client):
    resp = client.post("/retrain", json={"config": "not_a_real_config"})
    assert resp.status_code == 404


def test_retrain_runs_subprocess_and_returns_output(client, monkeypatch):
    class FakeResult:
        returncode = 0
        stdout = "test metrics: {...}"
        stderr = ""

    monkeypatch.setattr(ml_service.subprocess, "run", lambda *a, **k: FakeResult())
    resp = client.post("/retrain", json={"config": "gbm_fd001"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "trained"


def test_retrain_surfaces_subprocess_failure(client, monkeypatch):
    class FakeResult:
        returncode = 1
        stdout = "partial output"
        stderr = "traceback here"

    monkeypatch.setattr(ml_service.subprocess, "run", lambda *a, **k: FakeResult())
    resp = client.post("/retrain", json={"config": "gbm_fd001"})
    assert resp.status_code == 500
