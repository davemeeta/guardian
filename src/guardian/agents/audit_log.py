"""Append-only audit trail of every debate/judge decision.

Plain JSONL (one JSON object per line) on the local filesystem — no
database server needed for this scale, trivially readable by a human
(`cat logs/agent_decisions.jsonl | jq .`) or by Phase 4's dashboard
(`pandas.read_json(path, lines=True)`). Every field the debate produced is
logged, including parse errors and per-agent timings, not just the final
decision — the transcript IS the audit trail.
"""
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from guardian.agents.debate import DebateResult

DEFAULT_LOG_PATH = Path(__file__).resolve().parents[3] / "logs" / "agent_decisions.jsonl"


def append_record(result: DebateResult, log_path: Path = DEFAULT_LOG_PATH, model: str = "") -> dict:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "scenario_id": result.evidence.scenario_id,
        "evidence": asdict(result.evidence),
        "monitor_flag": result.monitor_flag,
        "monitor_reasoning": result.monitor_reasoning,
        "debated": result.debated,
        "advocate_for_text": result.advocate_for_text,
        "advocate_against_text": result.advocate_against_text,
        "judge_decision": result.judge_decision,
        "judge_rationale": result.judge_rationale,
        "judge_confidence": result.judge_confidence,
        "parse_errors": result.parse_errors,
        "timings_s": result.timings_s,
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")
    return record
