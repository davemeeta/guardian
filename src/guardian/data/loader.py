from pathlib import Path

import pandas as pd

from guardian.data.columns import ALL_COLS

DEFAULT_RAW_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / "cmapss"


def _read_space_delimited(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=None, names=ALL_COLS)


def load_train(subset: str, raw_dir: Path = DEFAULT_RAW_DIR) -> pd.DataFrame:
    return _read_space_delimited(raw_dir / f"train_{subset}.txt")


def load_test(subset: str, raw_dir: Path = DEFAULT_RAW_DIR) -> pd.DataFrame:
    return _read_space_delimited(raw_dir / f"test_{subset}.txt")


def load_test_rul(subset: str, raw_dir: Path = DEFAULT_RAW_DIR) -> pd.Series:
    """True RUL at the final cycle of each test unit, indexed by unit_number (1-based)."""
    values = pd.read_csv(raw_dir / f"RUL_{subset}.txt", header=None)[0]
    values.index = pd.RangeIndex(1, len(values) + 1, name="unit_number")
    return values
