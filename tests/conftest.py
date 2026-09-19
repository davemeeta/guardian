import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from guardian.agents.evidence import Evidence  # noqa: E402


@pytest.fixture
def make_evidence():
    def _make(**overrides) -> Evidence:
        fields = dict(
            scenario_id="test", n_units=5,
            baseline_rmse=20.0, current_rmse=20.0, rmse_delta_pct=0.0,
            baseline_imminent_rate=0.1, current_imminent_rate=0.1,
            sensor_drift={"sensor_4": 0.5, "sensor_11": -0.3},
            decline_rate_value=1.0, decline_rate_percentile=50.0,
        )
        fields.update(overrides)
        return Evidence(**fields)

    return _make
