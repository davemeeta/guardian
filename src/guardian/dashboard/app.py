"""Guardian dashboard: model health over time, and the full audit trail of
every autonomous agent decision (monitor flag, both arguments, judge
verdict + rationale) — the explainability surface for the debate/judge
mechanism.

Run: streamlit run src/guardian/dashboard/app.py
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from guardian.agents.evidence import Evidence  # noqa: E402

EXPERIMENT_NAME = "guardian-baseline-rul"
AUDIT_LOG_PATH = ROOT / "logs" / "agent_decisions.jsonl"

st.set_page_config(page_title="Guardian", layout="wide")
st.title("Guardian — model health & agent decision audit trail")
st.caption(
    "A self-governing predictive-maintenance system. Every retrain decision below was made "
    "by a local, open-weight LLM debate — no cloud calls, full transcript logged."
)


@st.cache_data(ttl=30)
def load_runs() -> pd.DataFrame:
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    try:
        return mlflow.search_runs(experiment_names=[EXPERIMENT_NAME])
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=10)
def load_decisions() -> list[dict]:
    if not AUDIT_LOG_PATH.exists():
        return []
    lines = AUDIT_LOG_PATH.read_text().strip().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


tab_health, tab_decisions = st.tabs(["Model health", "Agent decisions & audit trail"])

with tab_health:
    runs = load_runs()
    if runs.empty:
        st.info("No MLflow runs found yet. Run `python scripts/train.py --config configs/<name>.yaml` first.")
    else:
        runs = runs.sort_values("start_time")
        run_names = sorted(runs["tags.mlflow.runName"].dropna().unique())
        selected = st.multiselect("Runs to show", run_names, default=run_names)
        filtered = runs[runs["tags.mlflow.runName"].isin(selected)]

        metric_options = [
            c.replace("metrics.test_", "") for c in runs.columns if c.startswith("metrics.test_")
        ]
        metric = st.selectbox("Metric", metric_options, index=metric_options.index("rmse") if "rmse" in metric_options else 0)
        col = f"metrics.test_{metric}"

        if col in filtered.columns and not filtered.empty:
            fig, ax = plt.subplots(figsize=(10, 4))
            plot_df = filtered.dropna(subset=[col]).sort_values("start_time")
            ax.bar(plot_df["tags.mlflow.runName"], plot_df[col])
            ax.set_ylabel(f"test_{metric}")
            ax.set_title(f"test_{metric} across all logged training runs")
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
            fig.tight_layout()
            st.pyplot(fig)

        st.subheader("Latest metrics per run")
        display_cols = ["tags.mlflow.runName", "start_time"] + [
            c for c in filtered.columns if c.startswith("metrics.test_")
        ]
        display_cols = [c for c in display_cols if c in filtered.columns]
        latest = filtered.sort_values("start_time", ascending=False).drop_duplicates(
            subset=["tags.mlflow.runName"]
        )
        st.dataframe(
            latest[display_cols].rename(columns=lambda c: c.replace("metrics.test_", "").replace("tags.mlflow.runName", "run")),
            width='stretch',
        )

with tab_decisions:
    decisions = load_decisions()
    if not decisions:
        st.info("No agent decisions logged yet. Run `python scripts/run_agent_pipeline.py --scenario all` first.")
    else:
        decisions = list(reversed(decisions))  # most recent first
        summary_rows = [
            {
                "timestamp": d["timestamp"],
                "scenario": d["scenario_id"],
                "model": d.get("model", ""),
                "monitor_flagged": d["monitor_flag"],
                "debated": d["debated"],
                "judge_decision": d.get("judge_decision") or "—",
                "confidence": d.get("judge_confidence") or "—",
            }
            for d in decisions
        ]
        summary = pd.DataFrame(summary_rows)
        st.subheader(f"Decision log ({len(summary)} total)")
        st.dataframe(summary, width='stretch')

        st.subheader("Full transcript")
        labels = [f"{r['timestamp']} — {r['scenario']} — {r['judge_decision']}" for r in summary_rows]
        idx = st.selectbox("Select a decision", range(len(labels)), format_func=lambda i: labels[i])
        record = decisions[idx]

        evidence_dict = {k: v for k, v in record["evidence"].items() if k != "top_drifted_sensors"}
        evidence = Evidence(**evidence_dict)

        st.markdown("**Evidence presented to the agents:**")
        st.code(evidence.to_prompt_text(), language=None)

        st.markdown(f"**Monitor** — flagged for review: `{record['monitor_flag']}`")
        st.write(record["monitor_reasoning"])

        if record["debated"]:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Advocate for retrain**")
                st.write(record["advocate_for_text"])
            with col2:
                st.markdown("**Advocate against retrain**")
                st.write(record["advocate_against_text"])

            st.markdown(f"**Judge verdict: `{record['judge_decision']}`** (confidence: {record['judge_confidence']})")
            st.write(record["judge_rationale"])
        else:
            st.write("No debate was needed — the Monitor didn't flag this batch for review.")

        if record.get("parse_errors"):
            st.warning(f"Parse errors during this run: {record['parse_errors']}")
        if record.get("timings_s"):
            st.caption(f"Timings: {record['timings_s']}")
