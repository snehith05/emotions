"""
Run the whole project:  python run_all.py

  1. get_data.py        download + checksum the data (skipped if already there)
  2. features           raw sensors -> one row per student-day (cached in data/processed/)
  3. eda                exploratory analysis          -> figures/eda*.png, results/eda_*
  4. reproduce          PART 1: the paper's analysis  -> results/reproduction_*, figures/fig1-3
  5. extension          PART 2: our ML extension      -> results/ml_*, figures/fig4-8

Add --refresh to rebuild the cached features from the raw data.
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def main():
    refresh = "--refresh" in sys.argv
    t = time.time()
    subprocess.run([sys.executable, str(ROOT / "get_data.py")], check=True)
    import features
    if refresh:
        for f in (ROOT / "data" / "processed").glob("*.pkl"):
            f.unlink()
    features.build_daily()
    import eda, reproduce, extension
    print("\n=== EDA ===")
    eda.main()
    print("\n=== PART 1: reproduction of Wang et al. (2014) ===")
    reproduce.main()
    print("\n=== PART 2: ML extension (ours) ===")
    extension.main()
    print(f"\nDone in {(time.time() - t) / 60:.1f} min. See results/ and figures/.")


if __name__ == "__main__":
    main()
