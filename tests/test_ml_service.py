from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from guardian.api import ml_service


@pytest.fixture
def client():
    return TestClient(ml_service.app)


def _stub_search_runs(monkeypatch, result):
    def search_runs(experiment_names):
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(ml_service.mlflow, "search_runs", search_runs)


def _stub_train_subprocess(monkeypatch, returncode, stdout="", stderr=""):
    fake = SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)
    monkeypatch.setattr(ml_service.subprocess, "run", lambda *a, **k: fake)


@pytest.mark.parametrize("runs", [pd.DataFrame(), Exception("experiment not found")])
def test_health_reports_no_runs_when_mlflow_empty_or_missing(client, monkeypatch, runs):
    _stub_search_runs(monkeypatch, runs)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "no runs yet", "runs": []}


def test_health_summarizes_latest_run_per_name(client, monkeypatch):
    _stub_search_runs(monkeypatch, pd.DataFrame([
        {"tags.mlflow.runName": "gbm_FD001", "start_time": pd.Timestamp("2026-01-02"), "metrics.test_rmse": 17.8},
        {"tags.mlflow.runName": "gbm_FD001", "start_time": pd.Timestamp("2026-01-01"), "metrics.test_rmse": 20.0},
    ]))

    body = client.get("/health").json()

    assert body["status"] == "ok"
    assert len(body["runs"]) == 1
    assert body["runs"][0]["run_name"] == "gbm_FD001"
    assert body["runs"][0]["rmse"] == 17.8  # the more recent of the two rows


def test_retrain_rejects_unknown_config(client):
    resp = client.post("/retrain", json={"config": "not_a_real_config"})
    assert resp.status_code == 404


def test_retrain_runs_subprocess_and_returns_output(client, monkeypatch):
    _stub_train_subprocess(monkeypatch, returncode=0, stdout="test metrics: {...}")
    resp = client.post("/retrain", json={"config": "gbm_fd001"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "trained"


def test_retrain_surfaces_subprocess_failure(client, monkeypatch):
    _stub_train_subprocess(monkeypatch, returncode=1, stdout="partial output", stderr="traceback here")
    resp = client.post("/retrain", json={"config": "gbm_fd001"})
    assert resp.status_code == 500
