"""Agent orchestrator service: runs the monitor -> debate -> judge pipeline
and, on approval, calls the ML service to actually retrain — the one place
in this project where a judge's decision has a real, executable
consequence, and it's a plain HTTP call to another container, not
something hidden inside a shared process.

Run: uvicorn guardian.api.agent_service:app --host 0.0.0.0 --port 8000
"""
import json
import os
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")

import httpx  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from guardian.agents.audit_log import DEFAULT_LOG_PATH, append_record  # noqa: E402
from guardian.agents.debate import run_debate  # noqa: E402
from guardian.agents.ollama_client import DEFAULT_MODEL  # noqa: E402
from guardian.agents.scenarios import SCENARIOS, GuardianContext, build_context, generate_scenario  # noqa: E402

ML_SERVICE_URL = os.environ.get("ML_SERVICE_URL", "http://localhost:8001")

app = FastAPI(title="Guardian Agent Service")
_ctx: GuardianContext | None = None


def get_context() -> GuardianContext:
    """Built lazily and cached — fitting the simulator + baseline GBM takes
    a few seconds, no need to repeat that on every request."""
    global _ctx
    if _ctx is None:
        _ctx = build_context()
    return _ctx


class RunRequest(BaseModel):
    scenario: str = "noise"
    model: str = DEFAULT_MODEL
    auto_execute: bool = False  # if True, actually call the ML service's /retrain on approval
    train_config: str = "gbm_fd001"


@app.post("/run")
def run(req: RunRequest) -> dict:
    if req.scenario not in SCENARIOS:
        raise HTTPException(status_code=404, detail=f"Unknown scenario '{req.scenario}', expected one of {SCENARIOS}")

    ctx = get_context()
    evidence = generate_scenario(ctx, req.scenario)
    result = run_debate(evidence, model=req.model)
    # Pass DEFAULT_LOG_PATH explicitly (looked up from this module's own
    # namespace at call time) rather than relying on append_record's own
    # default — that default is bound once when audit_log.py is imported,
    # so it wouldn't respect a path patched onto this module (e.g. in tests).
    record = append_record(result, log_path=DEFAULT_LOG_PATH, model=req.model)

    retrain_result = None
    if req.auto_execute and result.judge_decision == "auto_approve_retrain":
        try:
            resp = httpx.post(
                f"{ML_SERVICE_URL}/retrain", json={"config": req.train_config}, timeout=900,
            )
            resp.raise_for_status()
            retrain_result = resp.json()
        except httpx.HTTPError as e:
            retrain_result = {"status": "retrain_failed", "error": str(e)}

    return {
        "scenario": req.scenario,
        "monitor_flag": result.monitor_flag,
        "debated": result.debated,
        "judge_decision": result.judge_decision,
        "judge_confidence": result.judge_confidence,
        "retrain_result": retrain_result,
        "record_timestamp": record["timestamp"],
    }


@app.get("/decisions")
def decisions(limit: int = 50) -> dict:
    """Full audit trail, most recent first — same file the dashboard reads."""
    path: Path = DEFAULT_LOG_PATH
    if not path.exists():
        return {"decisions": []}
    lines = path.read_text().strip().splitlines()
    records = [json.loads(line) for line in lines[-limit:]]
    records.reverse()
    return {"decisions": records}
