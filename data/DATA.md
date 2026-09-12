# Dataset: NASA C-MAPSS Turbofan Engine Degradation Simulation

**Source:** NASA Prognostics Center of Excellence (PCoE) Data Set Repository.
Downloaded from the official PCoE mirror:
`https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip`

**Reference:** A. Saxena, K. Goebel, D. Simon, and N. Eklund, "Damage Propagation
Modeling for Aircraft Engine Run-to-Failure Simulation," PHM08, Denver CO, Oct 2008.

**License:** Public domain (NASA open data).

Not versioned in git — run `scripts/download_data.py` to fetch it into `data/raw/cmapss/`.

## Subsets used in this project

We use **FD001** and **FD004**, the two extremes of the four available subsets:

| Subset | Train units | Test units | Operating conditions | Fault modes |
|--------|------------|-----------|----------------------|-------------|
| FD001  | 100        | 100       | 1 (sea level)         | 1 (HPC degradation) |
| FD004  | 248        | 249       | 6                     | 2 (HPC + fan degradation) |

FD001 is the clean baseline case; FD004 is the hardest case (multiple regimes,
multiple fault modes) and is the more realistic stand-in for a fleet operating
across varying conditions. Contrasting the two makes the class-imbalance and
augmentation story in Phase 2 more concrete.

## File format

Each `train_FD00X.txt` / `test_FD00X.txt` is whitespace-delimited, no header,
26 columns per row (one row = one engine-cycle):

1. `unit_number` — engine ID within the subset
2. `time_cycles` — cycle index (starts at 1)
3–5. `op_setting_1..3` — operational settings
6–26. `sensor_1..21` — sensor measurements

`RUL_FD00X.txt` gives the true Remaining Useful Life for the **last** cycle of
each engine in the corresponding test file (the test trajectories are truncated
before failure; the task is to predict how many cycles are left at that point).

Training trajectories run all the way to failure, so RUL for training rows is
derived directly: `RUL = max(time_cycles for unit) - time_cycles`.
