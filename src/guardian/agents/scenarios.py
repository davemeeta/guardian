"""Builds the shared model+simulator context once, and generates controlled
test scenarios for exercising the debate pipeline end to end.

Three scenarios, deliberately designed with known ground truth so we can
sanity-check the debate mechanism makes the right call:
  - "noise": sampled from the simulator's own calibrated distribution —
    should look like ordinary sampling variation. Expected: auto_reject.
  - "genuine_drift": a much faster decline rate PLUS a sensor shift outside
    the simulator's own fitted family (simulating e.g. a sensor
    recalibration or new fault mode) — a real regime change, not just "more
    of the same." Expected: auto_approve_retrain.
  - "ambiguous": a small batch with a modest rate increase — enough signal
    to be suspicious, not enough (and too few samples) to be conclusive.
    Expected: escalate_to_human.
"""
import yaml
from dataclasses import dataclass
from pathlib import Path

from guardian.agents.evidence import Evidence, build_evidence, evaluate_batch
from guardian.data.features import FeatureBuilder
from guardian.data.labeling import add_train_rul
from guardian.data.loader import load_train
from guardian.data.split import train_val_unit_split
from guardian.models.gbm import GBMBaseline, GBMConfig
from guardian.simulator.engine_simulator import EngineSimulator

SUBSET = "FD001"
RUL_CLIP = 125
ROOT = Path(__file__).resolve().parents[3]
SCENARIOS = ["noise", "genuine_drift", "ambiguous"]
BASELINE_ENGINES = 60  # one large pooled draw, not several small ones averaged —
# pooling all cycles into a single RMSE computation gives a tighter estimate
# than averaging several noisier small-sample estimates (batch-to-batch RMSE
# on 15-engine draws was observed to swing +-4-5 cycles from noise alone,
# comparable in size to some of the effects we're trying to detect).


@dataclass
class GuardianContext:
    sim: EngineSimulator
    model: GBMBaseline
    fb: FeatureBuilder
    baseline_rmse: float
    baseline_imminent_rate: float
    next_unit_id: int


def build_context() -> GuardianContext:
    """Refits the Phase 1 GBM baseline and Phase 2 simulator on the same
    train/val split used throughout the project.

    The "historical baseline" RMSE/imminent-rate agents compare each batch
    against is deliberately computed from several FRESH, unperturbed
    simulator draws rather than from real validation data. GBM has a known
    real-vs-synthetic performance gap (see Phase 2's augmentation sweep —
    it's genuinely worse on any simulator-generated batch, even an
    unperturbed one, than on real data). If the baseline came from real
    data, every simulator-drawn scenario — including "noise" — would look
    like drift just from that gap, which isn't the thing we're trying to
    detect. Comparing simulator-drawn batches against a simulator-drawn
    baseline isolates the one thing we actually inject differently: whether
    a scenario is perturbed or not.
    """
    real_raw = load_train(SUBSET)
    fit_raw, val_raw = train_val_unit_split(real_raw)

    sim = EngineSimulator().fit(add_train_rul(fit_raw, clip=None))

    fb = FeatureBuilder(SUBSET)
    fit_df = fb.fit_transform(add_train_rul(fit_raw, clip=RUL_CLIP))
    val_df = fb.transform(add_train_rul(val_raw, clip=RUL_CLIP))

    gbm_cfg = yaml.safe_load((ROOT / "configs" / "gbm_fd001.yaml").read_text())["gbm"]
    model = GBMBaseline(GBMConfig(**gbm_cfg))
    model.fit(
        fit_df[fb.feature_cols], fit_df["RUL"],
        eval_set=(val_df[fb.feature_cols], val_df["RUL"]),
    )

    next_unit_id = int(real_raw["unit_number"].max()) + 1
    baseline_draw = sim.sample(
        n_engines=BASELINE_ENGINES, start_unit_id=next_unit_id, rate_boost=1.0, random_state=1000
    )
    baseline_rmse, baseline_imminent_rate, _ = evaluate_batch(baseline_draw, model, fb, rul_clip=RUL_CLIP)

    return GuardianContext(
        sim=sim, model=model, fb=fb,
        baseline_rmse=baseline_rmse,
        baseline_imminent_rate=baseline_imminent_rate,
        next_unit_id=next_unit_id,
    )


def generate_scenario(ctx: GuardianContext, scenario: str) -> Evidence:
    if scenario == "noise":
        batch = ctx.sim.sample(
            n_engines=30, start_unit_id=ctx.next_unit_id, rate_boost=1.0, random_state=101
        )
    elif scenario == "genuine_drift":
        batch = ctx.sim.sample(
            n_engines=30, start_unit_id=ctx.next_unit_id, rate_boost=1.3, random_state=102
        )
        # An extra shift outside the simulator's own fitted family (not just
        # a faster version of normal wear) — stands in for something like a
        # sensor recalibration or a new fault mode the model has never seen.
        for sensor, shift_sds in [("sensor_4", 4.0), ("sensor_11", -4.0)]:
            batch[sensor] = batch[sensor] + shift_sds * ctx.sim.offset_std[sensor]
    elif scenario == "ambiguous":
        batch = ctx.sim.sample(
            n_engines=6, start_unit_id=ctx.next_unit_id, rate_boost=1.3, random_state=103
        )
    else:
        raise ValueError(f"Unknown scenario '{scenario}', expected one of {SCENARIOS}")

    return build_evidence(
        scenario_id=scenario,
        window_df=batch,
        model=ctx.model,
        fb=ctx.fb,
        baseline_rmse=ctx.baseline_rmse,
        baseline_imminent_rate=ctx.baseline_imminent_rate,
        sim=ctx.sim,
    )
