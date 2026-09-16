# Guardian — interview demo script

A ~10-minute walkthrough that shows: real ML on real data, an honestly-
reported experiment (not just wins), and the actual differentiator — a
local, auditable multi-agent decision layer. Times are approximate; cut
anything if you're short.

## Before you start (do this 5-10 minutes before the interview)

Live LLM calls are the one part of this demo that can be slow or flaky —
prepare so you're never staring at a spinner on a call.

```bash
# 1. Ollama running with the model already warm (first call after a cold
#    start can take 30-90s just to load the model into memory)
ollama serve &
ollama run llama3.2:3b "say ok"

# 2. venv active, dashboard up in a browser tab
source .venv/bin/activate
streamlit run src/guardian/dashboard/app.py
# -> leave this running in a terminal tab, open http://localhost:8501

# 3. Have notebooks/01, 02, 03 already executed (they are, in the repo) and
#    open in a browser tab or Jupyter, ready to scroll to — don't re-run
#    them live, just show the saved output.

# 4. Know which decision in logs/agent_decisions.jsonl you'll show in the
#    dashboard as backup, in case a live run is slow — the genuine_drift
#    one is the most interesting (full debate, escalate_to_human verdict).
```

## The walkthrough

### 1. The pitch (30s, no screen needed)

"Guardian predicts when a jet engine will fail, using real NASA turbofan
degradation data. The differentiator isn't the prediction model — it's
that the system also monitors its own accuracy, and instead of silently
retraining itself when performance looks off, three separate local LLMs
argue about whether a change is real drift or just noise, and only a
fourth — the judge — decides. Everything runs on-machine, nothing calls
out to a cloud API, and every decision is fully logged."

### 2. The data problem (1 min) — `notebooks/01_eda.ipynb`

Scroll to the class-imbalance section. Say: "Fewer than 1 in 10 cycles in
this dataset represent an engine actually close to failure. A model that
never predicts 'about to fail' still looks 90%+ accurate — which is
exactly the trap a naive predictive-maintenance pitch falls into. This is
what motivated Phase 2."

### 3. The digital twin (1.5 min) — `notebooks/02_simulator_and_augmentation.ipynb`

Show the real-vs-synthetic trajectory plot and the before/after
augmentation results table. Say: "I built a physics-informed simulator —
not a GAN, an actual calibrated degradation-curve model — to generate
synthetic failure trajectories and rebalance training data toward the
rare class. The honest result: it helped the LSTM a lot (29% RMSE
reduction) and didn't help the tree model at all, confirmed with a proper
validation sweep, not just two lucky configs. I'd rather show a real,
explainable negative result than a cherry-picked win."

### 4. The debate mechanism, live (3-4 min) — the main event

```bash
python scripts/run_agent_pipeline.py --scenario genuine_drift
```

While it runs (~15-25s with a warm model), narrate: "This feeds the
pipeline a batch of evidence built from the simulator — a faster decline
rate and a sensor shift outside anything it was calibrated on. The Monitor
agent decides if it's worth a debate. If it flags it, two more agents
argue opposite sides using the *same* evidence, and a Judge weighs both
arguments and picks: approve retrain, reject, or escalate to a human."

Point at the output as it streams:
- Monitor's flag + reasoning
- Both advocates' arguments
- The Judge's verdict + rationale

**If the Judge lands on `escalate_to_human`** (it often does on this
scenario): lean into it, don't apologize for it. "This is actually the
interesting outcome. A naive system would force a yes/no. Guardian's whole
design principle is that genuinely mixed evidence should go to a person —
and if you look at the evidence, one number (RMSE) actually looked
*better*, while sensor drift and the decline-rate comparison screamed
'unusual.' That's a real, deliberately-surfaced finding from building
this: tree models can look *more* accurate under exactly the kind of
distribution shift that should worry you, because they can't extrapolate
past their training range. The evidence text literally warns the agents
about that. A cautious verdict here is the system working as intended."

### 5. The audit trail (1.5 min) — dashboard, "Agent decisions" tab

Switch to the already-open dashboard. Click the run you just made (or the
backed-up `genuine_drift` entry). Show:
- The decision log table (sortable, all runs)
- The full transcript view — evidence, both arguments, verdict, side by
  side

Say: "This is the compliance story. Nothing about *why* the system decided
what it decided is hidden — an auditor or an engineer can read the actual
argument, not reverse-engineer a threshold."

### 6. It's real infrastructure, not a notebook (1 min) — optional if time allows

```bash
docker compose up --build
```
(or, if already built, just `docker compose up -d` — much faster)

Say: "Three services — the ML/inference layer, the agent orchestrator, and
this dashboard — each containerized. Ollama itself stays on the host
deliberately; there's no Metal GPU passthrough into Docker on macOS, so
containerizing it would just make it slower for no benefit." Show
`docker compose ps`, hit `curl localhost:8001/health`.

If you have GitHub open: show `.github/workflows/guardian-pipeline.yml`
and a past Actions run. "This installs Ollama fresh on the GitHub runner
and runs the real pipeline in CI — not a mock. Retraining only happens if
the judge actually approved it."

### 7. Close (30s)

"Everything here — the simulator, the debate mechanism, the evidence
design — is designed to be explainable in a sentence, because the whole
point of a system like this in a regulated or safety-adjacent industry is
that a human can trust *why* it made a call, not just that it made one."

## If you have more time

- Show `outputs/sweep_gbm.csv` / `sweep_lstm.csv` and the sweep plots —
  the validation-driven methodology behind the augmentation claim.
- Run `--scenario noise` live to show the Monitor correctly declining to
  start a debate at all — the common-case behavior, not just the dramatic
  one.
- Open `src/guardian/agents/prompts.py` — the actual prompts are short,
  readable, and worth having memorized the gist of if asked "how do you
  keep a small model honest?" (answer: explicit instructions not to
  invent evidence, and a fail-safe default to escalate on any parse
  failure — see `run_debate()` in `src/guardian/agents/debate.py`).

## If something breaks live

- **Ollama call hangs/errors:** fall back to the dashboard's existing
  decision log — every scenario has already been run and logged at least
  once, so the transcripts are there regardless.
- **Docker isn't up / slow to start:** skip step 6 entirely, it's the
  lowest-value part of this specific demo. The agent pipeline and
  dashboard both run standalone without Docker.
