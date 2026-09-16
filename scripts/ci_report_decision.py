"""Reads the most recent agent decision and writes GitHub Actions outputs:
a step output ('decision') and a Job Summary section. Kept as a real script
rather than inline shell/Python in the workflow YAML for readability.

Usage: python scripts/ci_report_decision.py
Requires GITHUB_OUTPUT and GITHUB_STEP_SUMMARY env vars (set by Actions).
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = ROOT / "logs" / "agent_decisions.jsonl"


def main() -> None:
    lines = LOG_PATH.read_text().strip().splitlines()
    record = json.loads(lines[-1])
    decision = record["judge_decision"] or "no_debate_needed"

    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"decision={decision}\n")

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        lines_out = [
            f"## Guardian decision: `{decision}`",
            "",
            f"**Scenario:** `{record['scenario_id']}`  ",
            f"**Monitor flagged for review:** `{record['monitor_flag']}`",
            "",
            f"> {record['monitor_reasoning']}",
        ]
        if record["debated"]:
            lines_out += [
                "",
                f"**Judge confidence:** `{record['judge_confidence']}`",
                "",
                f"> {record['judge_rationale']}",
            ]
        with open(summary_path, "a") as f:
            f.write("\n".join(lines_out) + "\n")

    print(f"decision: {decision}")


if __name__ == "__main__":
    main()
