# Guardian

A self-governing ML system for rare-event prediction in industrial maintenance.

Guardian doesn't just predict machine failures — it's designed (across later
phases) to monitor its own model's health, debate internally whether observed
performance drift is real or noise, and decide whether to retrain, with a
full audit trail and everything running on local, open-weight LLMs. See the
project brief for the full five-phase plan; this repo currently covers
**Phase 1: data & baseline ML**.

## Phase 1 status

- Dataset: NASA C-MAPSS turbofan degradation simulation, subsets **FD001**
  (1 operating condition, 1 fault mode — the clean case) and **FD004**
  (6 operating conditions, 2 fault modes — the hard case). See
  [`data/DATA.md`](data/DATA.md) for provenance and format.
- Baseline models: a per-cycle LightGBM regressor and a sliding-window LSTM,
  both predicting Remaining Useful Life (RUL). See
  [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) for the EDA and, in
  particular, for how it documents the **class imbalance in the failure
  signal** — fewer than 1 in 10 cycles represent an engine actually close to
  failure. That imbalance is the problem Phase 2's digital-twin simulator
  exists to address.
- Experiment tracking: MLflow, entirely local (SQLite backend, no server).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python scripts/download_data.py
```

## Training a baseline

```bash
python scripts/train.py --config configs/gbm_fd001.yaml
python scripts/train.py --config configs/lstm_fd004.yaml
```

Each run trains on the subset's training trajectories, validates on a
held-out slice of *whole engines* (never individual cycles — see
[`src/guardian/data/split.py`](src/guardian/data/split.py)), evaluates on the
official test set against the true RUL values NASA provides, and logs
params/metrics/plots/the model checkpoint to MLflow.

To inspect results:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Metrics logged per run: RMSE, MAE, the PHM08 challenge's asymmetric scoring
function (penalizes overestimating remaining life — the dangerous direction
— more than underestimating it), and precision/recall/F1 on the
"imminent failure" class (RUL ≤ 20 cycles) — the metric that actually
reflects the rare-event detection problem, as opposed to overall regression
error. See [`src/guardian/eval/metrics.py`](src/guardian/eval/metrics.py).

## Design decisions worth knowing about

- **Training RUL labels are clipped at 125 cycles** (piecewise-linear
  degradation assumption from Heimes 2008): early in an engine's life, RUL
  isn't predictable from sensor readings, so training against the raw
  unbounded value adds noise rather than signal. Evaluation always compares
  against NASA's true (unclipped) test RUL — the clip is a training-only
  trick. See [`src/guardian/data/labeling.py`](src/guardian/data/labeling.py).
- **FD002/FD004 sensors are z-scored within their operating regime**, not
  globally. These subsets cycle through six discrete operating conditions
  that dominate a sensor's raw value far more than degradation does; without
  this step, a model mostly learns "which regime is this" rather than "how
  degraded is this engine." Regime clustering and normalization stats are
  fit on train and reused on test — never re-fit per split. See
  [`src/guardian/data/regimes.py`](src/guardian/data/regimes.py).
- **FD001/FD003 features are z-scored too** (using train statistics), even
  though tree models don't need it, because the LSTM does — unscaled
  sensor inputs (temperatures in the hundreds, ratios near 1) made it
  collapse to near-constant predictions during development. See
  [`src/guardian/data/features.py`](src/guardian/data/features.py).
- **LightGBM and PyTorch's MPS backend cannot be active in the same process**
  on this machine (macOS) — importing both and running real computation from
  each causes a native segfault. `scripts/train.py` imports each model
  library lazily, inside its own branch, so a single run only ever loads one.
- **MLflow's background telemetry thread is disabled** via
  `MLFLOW_DISABLE_TELEMETRY=true`, set before MLflow is imported. Consistent
  with this project's no-data-leaves-the-machine principle, and it also
  happened to race with the MPS backend and crash training.

## Repo layout

```
data/            raw (gitignored) + provenance docs
src/guardian/    data loading/labeling/features, models, eval metrics
scripts/         download_data.py, train.py
configs/         one YAML per (model, subset) experiment
notebooks/       EDA
tests/           unit tests for labeling, regime normalization, sequence
                 windowing, and metrics
```
