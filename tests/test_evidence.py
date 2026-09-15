from guardian.agents.evidence import Evidence


def _evidence(**overrides) -> Evidence:
    defaults = dict(
        scenario_id="test",
        n_units=10,
        baseline_rmse=20.0,
        current_rmse=20.0,
        rmse_delta_pct=0.0,
        baseline_imminent_rate=0.1,
        current_imminent_rate=0.1,
        sensor_drift={"sensor_4": 0.5, "sensor_11": -0.3},
        decline_rate_value=1.0,
        decline_rate_percentile=50.0,
    )
    defaults.update(overrides)
    return Evidence(**defaults)


def test_top_drifted_sensors_sorted_by_magnitude():
    ev = _evidence(sensor_drift={"sensor_4": 0.5, "sensor_11": -4.2, "sensor_7": 1.1})
    assert ev.top_drifted_sensors[0] == ("sensor_11", -4.2)
    assert ev.top_drifted_sensors[1] == ("sensor_7", 1.1)


def test_rmse_may_be_misleading_flags_large_drift_with_flat_rmse():
    ev = _evidence(sensor_drift={"sensor_4": 4.3}, rmse_delta_pct=-5.0)
    assert ev.rmse_may_be_misleading is True


def test_rmse_may_be_misleading_false_when_drift_small():
    ev = _evidence(sensor_drift={"sensor_4": 1.2}, rmse_delta_pct=-5.0)
    assert ev.rmse_may_be_misleading is False


def test_rmse_may_be_misleading_false_when_rmse_clearly_worse():
    ev = _evidence(sensor_drift={"sensor_4": 4.3}, rmse_delta_pct=50.0)
    assert ev.rmse_may_be_misleading is False


def test_prompt_text_includes_caveat_only_when_flagged():
    flagged = _evidence(sensor_drift={"sensor_4": 4.3}, rmse_delta_pct=-5.0)
    clean = _evidence(sensor_drift={"sensor_4": 0.5}, rmse_delta_pct=-5.0)
    assert "CAVEAT" in flagged.to_prompt_text()
    assert "CAVEAT" not in clean.to_prompt_text()


def test_prompt_text_handles_missing_decline_rate():
    ev = _evidence(decline_rate_value=None, decline_rate_percentile=None)
    assert "Not enough completed trajectories" in ev.to_prompt_text()
