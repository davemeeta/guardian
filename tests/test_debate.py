from guardian.agents import debate
from guardian.agents.evidence import Evidence
from guardian.agents.ollama_client import LLMResponse


def _evidence() -> Evidence:
    return Evidence(
        scenario_id="test", n_units=5,
        baseline_rmse=20.0, current_rmse=20.0, rmse_delta_pct=0.0,
        baseline_imminent_rate=0.1, current_imminent_rate=0.1,
        sensor_drift={"sensor_4": 0.2}, decline_rate_value=1.0, decline_rate_percentile=50.0,
    )


def _resp(text: str) -> LLMResponse:
    return LLMResponse(text=text, model="test", eval_count=1, eval_duration_s=0.01)


def test_monitor_reject_skips_debate(monkeypatch):
    monkeypatch.setattr(
        debate, "generate_json",
        lambda *a, **k: ({"flag_for_review": False, "reasoning": "nothing unusual"}, _resp("")),
    )
    called = {"advocate": False, "judge": False}
    monkeypatch.setattr(debate, "generate", lambda *a, **k: called.update(advocate=True) or _resp(""))

    result = debate.run_debate(_evidence())

    assert result.monitor_flag is False
    assert result.debated is False
    assert result.judge_decision is None
    assert called["advocate"] is False


def test_monitor_flag_runs_full_debate(monkeypatch):
    monitor_and_judge = iter([
        ({"flag_for_review": True, "reasoning": "elevated drift"}, _resp("")),
        ({"decision": "auto_approve_retrain", "rationale": "clear drift", "confidence": "high"}, _resp("")),
    ])
    monkeypatch.setattr(debate, "generate_json", lambda *a, **k: next(monitor_and_judge))
    monkeypatch.setattr(debate, "generate", lambda *a, **k: _resp("some argument text"))

    result = debate.run_debate(_evidence())

    assert result.debated is True
    assert result.advocate_for_text == "some argument text"
    assert result.advocate_against_text == "some argument text"
    assert result.judge_decision == "auto_approve_retrain"
    assert result.judge_confidence == "high"
    assert result.parse_errors == []


def test_monitor_unparseable_json_fails_safe_to_flagged(monkeypatch):
    monkeypatch.setattr(debate, "generate_json", lambda *a, **k: (None, _resp("not json")))
    monkeypatch.setattr(debate, "generate", lambda *a, **k: _resp("argument"))

    result = debate.run_debate(_evidence())

    assert result.monitor_flag is True
    assert any("monitor" in e for e in result.parse_errors)


def test_judge_invalid_decision_fails_safe_to_escalate(monkeypatch):
    monitor_and_judge = iter([
        ({"flag_for_review": True, "reasoning": "x"}, _resp("")),
        ({"decision": "not_a_real_decision", "rationale": "x", "confidence": "low"}, _resp("bad")),
    ])
    monkeypatch.setattr(debate, "generate_json", lambda *a, **k: next(monitor_and_judge))
    monkeypatch.setattr(debate, "generate", lambda *a, **k: _resp("argument"))

    result = debate.run_debate(_evidence())

    assert result.judge_decision == "escalate_to_human"
    assert any("judge" in e for e in result.parse_errors)


def test_judge_none_json_fails_safe_to_escalate(monkeypatch):
    monitor_and_judge = iter([
        ({"flag_for_review": True, "reasoning": "x"}, _resp("")),
        (None, _resp("garbage")),
    ])
    monkeypatch.setattr(debate, "generate_json", lambda *a, **k: next(monitor_and_judge))
    monkeypatch.setattr(debate, "generate", lambda *a, **k: _resp("argument"))

    result = debate.run_debate(_evidence())

    assert result.judge_decision == "escalate_to_human"
    assert result.judge_confidence == "low"
