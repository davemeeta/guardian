import pytest


def test_top_drifted_sensors_sorted_by_magnitude(make_evidence):
    ev = make_evidence(sensor_drift={"sensor_4": 0.5, "sensor_11": -4.2, "sensor_7": 1.1})
    assert ev.top_drifted_sensors[0] == ("sensor_11", -4.2)
    assert ev.top_drifted_sensors[1] == ("sensor_7", 1.1)


@pytest.mark.parametrize(
    "drift, rmse_delta_pct, expected",
    [
        (4.3, -5.0, True),   # large drift, RMSE flat/improved -> RMSE untrustworthy
        (1.2, -5.0, False),  # drift too small to matter
        (4.3, 50.0, False),  # RMSE clearly worse, so it isn't misleading
    ],
)
def test_rmse_may_be_misleading(make_evidence, drift, rmse_delta_pct, expected):
    ev = make_evidence(sensor_drift={"sensor_4": drift}, rmse_delta_pct=rmse_delta_pct)
    assert ev.rmse_may_be_misleading is expected


def test_prompt_text_includes_caveat_only_when_flagged(make_evidence):
    flagged = make_evidence(sensor_drift={"sensor_4": 4.3}, rmse_delta_pct=-5.0)
    clean = make_evidence(sensor_drift={"sensor_4": 0.5}, rmse_delta_pct=-5.0)
    assert "CAVEAT" in flagged.to_prompt_text()
    assert "CAVEAT" not in clean.to_prompt_text()


def test_prompt_text_handles_missing_decline_rate(make_evidence):
    ev = make_evidence(decline_rate_value=None, decline_rate_percentile=None)
    assert "Not enough completed trajectories" in ev.to_prompt_text()
