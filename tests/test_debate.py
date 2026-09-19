from guardian.agents import debate
from guardian.agents.ollama_client import LLMResponse


def _resp(text: str) -> LLMResponse:
    return LLMResponse(text=text, model="test", eval_count=1, eval_duration_s=0.01)


def _stub_llm(monkeypatch, json_results, text="argument"):
    """generate_json returns json_results in order (monitor, then judge);
    generate (the advocates) always returns `text`."""
    results = iter(json_results)
    monkeypatch.setattr(debate, "generate_json", lambda *a, **k: next(results))
    monkeypatch.setattr(debate, "generate", lambda *a, **k: _resp(text))


def test_monitor_reject_skips_debate(monkeypatch, make_evidence):
    _stub_llm(monkeypatch, [({"flag_for_review": False, "reasoning": "nothing unusual"}, _resp(""))])
    advocate_called = []
    monkeypatch.setattr(debate, "generate", lambda *a, **k: advocate_called.append(1))

    result = debate.run_debate(make_evidence())

    assert result.monitor_flag is False
    assert result.debated is False
    assert result.judge_decision is None
    assert advocate_called == []


def test_monitor_flag_runs_full_debate(monkeypatch, make_evidence):
    _stub_llm(
        monkeypatch,
        [
            ({"flag_for_review": True, "reasoning": "elevated drift"}, _resp("")),
            ({"decision": "auto_approve_retrain", "rationale": "clear drift", "confidence": "high"}, _resp("")),
        ],
        text="some argument text",
    )

    result = debate.run_debate(make_evidence())

    assert result.debated is True
    assert result.advocate_for_text == "some argument text"
    assert result.advocate_against_text == "some argument text"
    assert result.judge_decision == "auto_approve_retrain"
    assert result.judge_confidence == "high"
    assert result.parse_errors == []


def test_monitor_unparseable_json_fails_safe_to_flagged(monkeypatch, make_evidence):
    _stub_llm(monkeypatch, [(None, _resp("not json")), (None, _resp("also bad"))])

    result = debate.run_debate(make_evidence())

    assert result.monitor_flag is True
    assert any("monitor" in e for e in result.parse_errors)


def test_judge_invalid_decision_fails_safe_to_escalate(monkeypatch, make_evidence):
    _stub_llm(
        monkeypatch,
        [
            ({"flag_for_review": True, "reasoning": "x"}, _resp("")),
            ({"decision": "not_a_real_decision", "rationale": "x", "confidence": "low"}, _resp("bad")),
        ],
    )

    result = debate.run_debate(make_evidence())

    assert result.judge_decision == "escalate_to_human"
    assert any("judge" in e for e in result.parse_errors)


def test_judge_none_json_fails_safe_to_escalate(monkeypatch, make_evidence):
    _stub_llm(
        monkeypatch,
        [({"flag_for_review": True, "reasoning": "x"}, _resp("")), (None, _resp("garbage"))],
    )

    result = debate.run_debate(make_evidence())

    assert result.judge_decision == "escalate_to_human"
    assert result.judge_confidence == "low"
