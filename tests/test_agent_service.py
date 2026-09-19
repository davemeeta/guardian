import json

import pytest
from fastapi.testclient import TestClient

from guardian.agents.debate import DebateResult
from guardian.api import agent_service


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_service, "DEFAULT_LOG_PATH", tmp_path / "agent_decisions.jsonl")
    monkeypatch.setattr(agent_service, "_ctx", None)
    return TestClient(agent_service.app)


@pytest.fixture
def stub_pipeline(monkeypatch, make_evidence):
    """Skip the real simulator/LLM: /run gets a canned DebateResult."""

    def _stub(**result_fields):
        result = DebateResult(evidence=make_evidence(), **result_fields)
        monkeypatch.setattr(agent_service, "get_context", lambda: object())
        monkeypatch.setattr(agent_service, "generate_scenario", lambda ctx, scenario: make_evidence())
        monkeypatch.setattr(agent_service, "run_debate", lambda evidence, model: result)

    return _stub


def test_run_rejects_unknown_scenario(client):
    resp = client.post("/run", json={"scenario": "not_a_real_scenario"})
    assert resp.status_code == 404


def test_decisions_empty_when_no_log(client):
    resp = client.get("/decisions")
    assert resp.status_code == 200
    assert resp.json() == {"decisions": []}


def test_run_success_not_flagged(client, stub_pipeline):
    stub_pipeline(monitor_flag=False, monitor_reasoning="fine", debated=False)

    resp = client.post("/run", json={"scenario": "noise"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["monitor_flag"] is False
    assert body["debated"] is False
    assert body["judge_decision"] is None
    assert body["retrain_result"] is None


def test_run_auto_execute_calls_ml_service_on_approval(client, stub_pipeline, monkeypatch):
    stub_pipeline(
        monitor_flag=True, monitor_reasoning="x", debated=True,
        advocate_for_text="for", advocate_against_text="against",
        judge_decision="auto_approve_retrain", judge_rationale="clear drift", judge_confidence="high",
    )
    posted_urls = []

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"status": "trained"}

    def fake_post(url, json, timeout):
        posted_urls.append(url)
        return FakeResponse()

    monkeypatch.setattr(agent_service.httpx, "post", fake_post)

    resp = client.post("/run", json={"scenario": "noise", "auto_execute": True})

    assert resp.status_code == 200
    body = resp.json()
    assert body["judge_decision"] == "auto_approve_retrain"
    assert body["retrain_result"] == {"status": "trained"}
    assert posted_urls[0].endswith("/retrain")


def test_decisions_reads_logged_records_most_recent_first(client):
    records = [
        {"scenario_id": "noise", "monitor_flag": False},
        {"scenario_id": "genuine_drift", "monitor_flag": True},
    ]
    agent_service.DEFAULT_LOG_PATH.write_text("".join(json.dumps(r) + "\n" for r in records))

    decisions = client.get("/decisions").json()["decisions"]

    assert [d["scenario_id"] for d in decisions] == ["genuine_drift", "noise"]
