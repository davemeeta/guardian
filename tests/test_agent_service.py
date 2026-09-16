import json

import pytest
from fastapi.testclient import TestClient

from guardian.agents.debate import DebateResult
from guardian.agents.evidence import Evidence
from guardian.api import agent_service


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_service, "DEFAULT_LOG_PATH", tmp_path / "agent_decisions.jsonl")
    monkeypatch.setattr(agent_service, "_ctx", None)
    return TestClient(agent_service.app)


def test_run_rejects_unknown_scenario(client):
    resp = client.post("/run", json={"scenario": "not_a_real_scenario"})
    assert resp.status_code == 404


def test_decisions_empty_when_no_log(client):
    resp = client.get("/decisions")
    assert resp.status_code == 200
    assert resp.json() == {"decisions": []}


def _fake_evidence() -> Evidence:
    return Evidence(
        scenario_id="noise", n_units=5,
        baseline_rmse=20.0, current_rmse=20.0, rmse_delta_pct=0.0,
        baseline_imminent_rate=0.1, current_imminent_rate=0.1,
        sensor_drift={"sensor_4": 0.1}, decline_rate_value=1.0, decline_rate_percentile=50.0,
    )


def test_run_success_not_flagged(client, monkeypatch):
    monkeypatch.setattr(agent_service, "get_context", lambda: object())
    monkeypatch.setattr(agent_service, "generate_scenario", lambda ctx, scenario: _fake_evidence())
    fake_result = DebateResult(
        evidence=_fake_evidence(), monitor_flag=False, monitor_reasoning="fine", debated=False,
    )
    monkeypatch.setattr(agent_service, "run_debate", lambda evidence, model: fake_result)

    resp = client.post("/run", json={"scenario": "noise"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["monitor_flag"] is False
    assert body["debated"] is False
    assert body["judge_decision"] is None
    assert body["retrain_result"] is None


def test_run_auto_execute_calls_ml_service_on_approval(client, monkeypatch):
    monkeypatch.setattr(agent_service, "get_context", lambda: object())
    monkeypatch.setattr(agent_service, "generate_scenario", lambda ctx, scenario: _fake_evidence())
    fake_result = DebateResult(
        evidence=_fake_evidence(), monitor_flag=True, monitor_reasoning="x", debated=True,
        advocate_for_text="for", advocate_against_text="against",
        judge_decision="auto_approve_retrain", judge_rationale="clear drift", judge_confidence="high",
    )
    monkeypatch.setattr(agent_service, "run_debate", lambda evidence, model: fake_result)

    calls = {}

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"status": "trained"}

    def fake_post(url, json, timeout):
        calls["url"] = url
        calls["json"] = json
        return FakeResponse()

    monkeypatch.setattr(agent_service.httpx, "post", fake_post)

    resp = client.post("/run", json={"scenario": "noise", "auto_execute": True})
    assert resp.status_code == 200
    body = resp.json()
    assert body["judge_decision"] == "auto_approve_retrain"
    assert body["retrain_result"] == {"status": "trained"}
    assert calls["url"].endswith("/retrain")


def test_decisions_reads_logged_records(client):
    log_path = agent_service.DEFAULT_LOG_PATH
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        f.write(json.dumps({"scenario_id": "noise", "monitor_flag": False}) + "\n")
        f.write(json.dumps({"scenario_id": "genuine_drift", "monitor_flag": True}) + "\n")

    resp = client.get("/decisions")
    body = resp.json()
    assert len(body["decisions"]) == 2
    # most recent first
    assert body["decisions"][0]["scenario_id"] == "genuine_drift"
