"""Orchestrates the four agent roles into one decision, sequentially.

Deliberately plain Python, not a graph framework: this pipeline is linear
(monitor -> maybe debate -> judge), and every step here is a function you
can read top to bottom — matching the project's "favor clarity over
cleverness, inspectable reasoning" principle. See prompts.py for what each
role is actually asked to do.
"""
import time
from dataclasses import dataclass, field

from guardian.agents.evidence import Evidence
from guardian.agents.ollama_client import DEFAULT_MODEL, generate, generate_json
from guardian.agents.prompts import (
    ADVOCATE_AGAINST_SYSTEM,
    ADVOCATE_FOR_SYSTEM,
    JUDGE_SYSTEM,
    MONITOR_SYSTEM,
    debate_prompt,
    evidence_prompt,
)

VALID_DECISIONS = {"auto_approve_retrain", "auto_reject", "escalate_to_human"}


@dataclass
class DebateResult:
    evidence: Evidence
    monitor_flag: bool
    monitor_reasoning: str
    debated: bool
    advocate_for_text: str | None = None
    advocate_against_text: str | None = None
    judge_decision: str | None = None
    judge_rationale: str | None = None
    judge_confidence: str | None = None
    parse_errors: list[str] = field(default_factory=list)
    timings_s: dict[str, float] = field(default_factory=dict)


def run_debate(evidence: Evidence, model: str = DEFAULT_MODEL) -> DebateResult:
    parse_errors: list[str] = []
    timings: dict[str, float] = {}
    ev_text = evidence.to_prompt_text()

    t0 = time.monotonic()
    monitor_json, monitor_resp = generate_json(
        evidence_prompt(ev_text), system=MONITOR_SYSTEM, model=model
    )
    timings["monitor_s"] = time.monotonic() - t0

    if monitor_json is None:
        parse_errors.append(f"monitor: could not parse JSON from: {monitor_resp.text!r}")
        # Fail safe: if we can't even parse whether to look closer, look closer.
        monitor_flag, monitor_reasoning = True, "Could not parse monitor response; defaulting to review."
    else:
        monitor_flag = bool(monitor_json.get("flag_for_review", True))
        monitor_reasoning = str(monitor_json.get("reasoning", ""))

    result = DebateResult(
        evidence=evidence,
        monitor_flag=monitor_flag,
        monitor_reasoning=monitor_reasoning,
        debated=False,
        parse_errors=parse_errors,
        timings_s=timings,
    )
    if not monitor_flag:
        return result

    result.debated = True

    t0 = time.monotonic()
    for_resp = generate(evidence_prompt(ev_text), system=ADVOCATE_FOR_SYSTEM, model=model)
    result.timings_s["advocate_for_s"] = time.monotonic() - t0
    result.advocate_for_text = for_resp.text.strip()

    t0 = time.monotonic()
    against_resp = generate(evidence_prompt(ev_text), system=ADVOCATE_AGAINST_SYSTEM, model=model)
    result.timings_s["advocate_against_s"] = time.monotonic() - t0
    result.advocate_against_text = against_resp.text.strip()

    t0 = time.monotonic()
    judge_json, judge_resp = generate_json(
        debate_prompt(ev_text, result.advocate_for_text, result.advocate_against_text),
        system=JUDGE_SYSTEM,
        model=model,
    )
    result.timings_s["judge_s"] = time.monotonic() - t0

    if judge_json is None or judge_json.get("decision") not in VALID_DECISIONS:
        result.parse_errors.append(f"judge: could not parse a valid decision from: {judge_resp.text!r}")
        # Fail safe: an unparseable or invalid verdict is never auto-approved
        # or auto-rejected silently — it always goes to a human.
        result.judge_decision = "escalate_to_human"
        result.judge_rationale = "Judge response could not be parsed into a valid decision; escalating by default."
        result.judge_confidence = "low"
    else:
        result.judge_decision = judge_json["decision"]
        result.judge_rationale = str(judge_json.get("rationale", ""))
        result.judge_confidence = str(judge_json.get("confidence", "low"))

    return result
