"""Run Guardian's monitor -> debate -> judge pipeline on a scenario.

Usage:
    python scripts/run_agent_pipeline.py --scenario noise
    python scripts/run_agent_pipeline.py --scenario all
    python scripts/run_agent_pipeline.py --scenario genuine_drift --model qwen2.5:7b-instruct-q5_K_M
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from guardian.agents.audit_log import append_record  # noqa: E402
from guardian.agents.debate import run_debate  # noqa: E402
from guardian.agents.ollama_client import DEFAULT_MODEL  # noqa: E402
from guardian.agents.scenarios import SCENARIOS, build_context, generate_scenario  # noqa: E402


def print_result(result) -> None:
    print("=" * 72)
    print(result.evidence.to_prompt_text())
    print()
    print(f"MONITOR: flag_for_review={result.monitor_flag}")
    print(f"  {result.monitor_reasoning}")
    if not result.debated:
        print("\n-> No debate needed, no retrain decision required.")
        return

    print(f"\nADVOCATE-FOR-RETRAIN:\n  {result.advocate_for_text}")
    print(f"\nADVOCATE-AGAINST-RETRAIN:\n  {result.advocate_against_text}")
    print(f"\nJUDGE: {result.judge_decision} (confidence: {result.judge_confidence})")
    print(f"  {result.judge_rationale}")
    if result.parse_errors:
        print(f"\n[parse errors: {result.parse_errors}]")
    total_s = sum(result.timings_s.values())
    print(f"\n[wall time: {total_s:.1f}s across {len(result.timings_s)} calls]")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=SCENARIOS + ["all"], default="all")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    print(f"Building context (real data + simulator + GBM baseline)...")
    ctx = build_context()

    scenarios = SCENARIOS if args.scenario == "all" else [args.scenario]
    for scenario in scenarios:
        print(f"\nGenerating evidence for scenario '{scenario}'...")
        evidence = generate_scenario(ctx, scenario)

        print(f"Running debate pipeline with model={args.model}...")
        result = run_debate(evidence, model=args.model)

        print_result(result)
        record = append_record(result, model=args.model)
        print(f"\n[logged to logs/agent_decisions.jsonl at {record['timestamp']}]")


if __name__ == "__main__":
    main()
