import pandas as pd

from guardian.data.labeling import add_train_rul


def test_rul_counts_down_to_zero_at_last_cycle():
    df = pd.DataFrame({
        "unit_number": [1, 1, 1, 2, 2],
        "time_cycles": [1, 2, 3, 1, 2],
    })
    out = add_train_rul(df, clip=None)
    assert out.loc[out["unit_number"] == 1, "RUL"].tolist() == [2, 1, 0]
    assert out.loc[out["unit_number"] == 2, "RUL"].tolist() == [1, 0]


def test_rul_is_clipped():
    df = pd.DataFrame({"unit_number": [1] * 5, "time_cycles": [1, 2, 3, 4, 5]})
    out = add_train_rul(df, clip=2)
    assert out["RUL"].max() <= 2
    assert out["RUL"].tolist() == [2, 2, 2, 1, 0]
