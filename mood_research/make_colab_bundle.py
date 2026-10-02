"""
Pack what the Colab notebook needs into colab_bundle.zip (upload it in the notebook's first cell).

Contains the code, the checksum manifest and - if already built - the daily feature table
(data/processed/daily.pkl, a few MB), so Colab can skip the 230 MB download and the slow feature step.

Run:  python make_colab_bundle.py
"""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FILES = ["get_data.py", "run_all.py", "requirements.txt", "data/github_manifest.json",
         "data/processed/daily.pkl"] + [f"src/{p.name}" for p in (ROOT / "src").glob("*.py")]


def main():
    out = ROOT / "colab_bundle.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in FILES:
            p = ROOT / rel
            if p.exists():
                z.write(p, f"mood_research/{rel}")
            elif rel.endswith("daily.pkl"):
                print("note: no data/processed/daily.pkl yet - Colab will rebuild it from the raw data")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
