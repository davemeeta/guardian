import pandas as pd

from guardian.data.sequences import make_test_windows, make_train_windows


def _unit_df(unit, n_cycles):
    return pd.DataFrame({
        "unit_number": unit,
        "time_cycles": range(1, n_cycles + 1),
        "f1": range(n_cycles),
        "RUL": [n_cycles - c for c in range(1, n_cycles + 1)],
    })


def test_train_windows_label_matches_last_cycle_in_window():
    df = _unit_df(1, 10)
    X, y = make_train_windows(df, ["f1"], window_size=3)
    # window ending at cycle 3 (f1 values 0,1,2) has RUL for cycle 3 = 10-3 = 7
    assert X[0].flatten().tolist() == [0, 1, 2]
    assert y[0] == 7
    # last window ends at the final cycle, RUL 0
    assert y[-1] == 0


def test_train_windows_pads_short_trajectory():
    df = _unit_df(1, 2)
    X, y = make_train_windows(df, ["f1"], window_size=5)
    assert X.shape == (1, 5, 1)
    # padded with repeats of first row (f1=0), then real rows 0, 1
    assert X[0].flatten().tolist() == [0, 0, 0, 0, 1]
    assert y[0] == 0  # RUL at final real cycle


def test_test_windows_takes_last_cycles_per_unit():
    df = pd.concat([_unit_df(1, 10), _unit_df(2, 4)], ignore_index=True)
    X, units = make_test_windows(df, ["f1"], window_size=3)
    assert units.tolist() == [1, 2]
    assert X[0].flatten().tolist() == [7, 8, 9]  # last 3 cycles of unit 1
    assert X[1].flatten().tolist() == [1, 2, 3]  # last 3 cycles of unit 2
