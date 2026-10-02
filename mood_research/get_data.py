"""
Download the StudentLife data and check every file.

1. The full dataset (sensing, EMA, surveys) from the Zenodo mirror of the original Dartmouth release:
   Fryer, D. "studentlife data in RData format", Zenodo, doi:10.5281/zenodo.3529253 (CC-BY-4.0).
   The original host, studentlife.cs.dartmouth.edu, was unreachable when this project was built.

2. Two things that mirror lost or never had, taken from the ORIGINAL release files (JSON/CSV):
   - EMA/response/Stress/*.json  (the Zenodo conversion dropped the stress "level" field)
   - education/grades.csv         (not in the Zenodo mirror; needed for the GPA table)
   - EMA/EMA_definition.json      (the official wording and answer codes of every EMA question)
   These come from a public GitHub copy of the original release, pinned to one commit. When this was
   built, two independent copies were byte-identical, and every stress row in the Zenodo mirror matched
   them exactly. The checksums are in data/github_manifest.json.

Run:  python get_data.py
"""
import hashlib
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
ZENODO_URL = "https://zenodo.org/api/records/3529253/files/dataset_rds.zip/content"
ZENODO_MD5 = "353ae79f157097a57bff1c004634ba1b"
GITHUB_REPO = "realfty/vad-lab-dmss"
GITHUB_COMMIT = "e4489442ed0c8e60c6e5e328c8e6db078e5dd476"
ORIG = RAW / "original" / "studentlife_github"


def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def download(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:7.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()
    tmp.replace(dest)


def main():
    zpath = RAW / "dataset_rds.zip"
    if not zpath.exists() or md5(zpath) != ZENODO_MD5:
        print("Downloading the StudentLife dataset from Zenodo (230 MB)...")
        download(ZENODO_URL, zpath)
    if md5(zpath) != ZENODO_MD5:
        sys.exit("Checksum mismatch for dataset_rds.zip - delete it and run again.")
    print("dataset_rds.zip: checksum OK")
    if not (RAW / "dataset_rds" / "sensing").exists():
        print("Unzipping...")
        zipfile.ZipFile(zpath).extractall(RAW)

    manifest = json.loads((ROOT / "data" / "github_manifest.json").read_text())
    bad = []
    for rel, want in manifest.items():
        dest = ORIG / rel
        if not dest.exists() or md5(dest) != want:
            url = f"https://raw.githubusercontent.com/{GITHUB_REPO}/{GITHUB_COMMIT}/dataset/{rel}"
            download(url, dest)
        if md5(dest) != want:
            bad.append(rel)
    if bad:
        sys.exit(f"Checksum mismatch for {len(bad)} original files, e.g. {bad[:3]}")
    print(f"{len(manifest)} original-release files: checksums OK")
    print("Data ready.")


if __name__ == "__main__":
    main()
