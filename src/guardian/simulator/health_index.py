import numpy as np

from guardian.data.labeling import RUL_CLIP


def degradation_progress(rul: np.ndarray, clip: int = RUL_CLIP) -> np.ndarray:
    """Map RUL to a 0-1 "how degraded is this engine right now" health index.

    0 while healthy (RUL >= clip), ramping linearly to 1 at failure (RUL=0).
    Deliberately reuses Phase 1's own RUL_CLIP boundary: that value already
    encodes "no informative degradation signal further than this many cycles
    from failure" for the baseline models, so the simulator's notion of
    when degradation "starts" matches what the models were trained to
    expect, rather than introducing a second, inconsistent assumption.
    """
    rul = np.asarray(rul, dtype=float)
    return np.clip(clip - rul, 0, clip) / clip
