"""Fetch the NASA C-MAPSS dataset into data/raw/cmapss/.

Idempotent: skips the download if the target files already exist.
"""
import shutil
import urllib.request
import zipfile
from pathlib import Path

URL = "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
TARGET_DIR = RAW_DIR / "cmapss"

EXPECTED_FILES = [
    "train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt",
    "train_FD004.txt", "test_FD004.txt", "RUL_FD004.txt",
    "readme.txt",
]


def main() -> None:
    if all((TARGET_DIR / f).exists() for f in EXPECTED_FILES):
        print(f"Dataset already present at {TARGET_DIR}, skipping download.")
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / "CMAPSSData.zip"

    print(f"Downloading {URL} ...")
    urllib.request.urlretrieve(URL, zip_path)

    extract_dir = RAW_DIR / "_extract"
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)

    # the outer zip contains one more nested zip with the actual data files
    nested_zips = list(extract_dir.rglob("CMAPSSData.zip"))
    if nested_zips:
        with zipfile.ZipFile(nested_zips[0]) as zf:
            zf.extractall(extract_dir)

    for f in EXPECTED_FILES + ["Damage Propagation Modeling.pdf"]:
        matches = list(extract_dir.rglob(f))
        if matches:
            shutil.copy(matches[0], TARGET_DIR / f)

    shutil.rmtree(extract_dir)
    zip_path.unlink()

    missing = [f for f in EXPECTED_FILES if not (TARGET_DIR / f).exists()]
    if missing:
        raise RuntimeError(f"Download/extract incomplete, missing: {missing}")

    print(f"Dataset ready at {TARGET_DIR}")


if __name__ == "__main__":
    main()
