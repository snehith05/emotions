"""Paths and the few analysis choices the paper leaves open (each one is documented in the README)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RDS = RAW / "dataset_rds"                         # Zenodo mirror (sensing, EMA, surveys)
ORIG = RAW / "original" / "studentlife_github"    # original-release files (stress EMA, grades)
PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

TZ = "America/New_York"         # Dartmouth local time - days and epochs are local
TERM_START = "2013-03-27"       # first day of term in education/deadlines.csv
TERM_END = "2013-06-04"         # 10 weeks (70 days) later
WINDOW_S = 600                  # the paper's 10-minute windows

# paper: "For each 10-min period, we calculate the ratio of non-stationary inferences. If the ratio is
# greater than a threshold, we consider this period active." The threshold is not given.
ACTIVE_RATIO = 0.5

# a day counts when its activity sensor covered at least this many 10-min windows (the paper removes
# days when the phone was off or left behind, without giving a rule)
MIN_DAY_WINDOWS = 96            # 16 hours

# epochs used throughout the paper
EPOCHS = {"day": (9, 18), "evening": (18, 24), "night": (0, 9)}

SEED = 42
