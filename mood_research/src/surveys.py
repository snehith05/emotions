"""Score the pre/post mental-health surveys exactly as each instrument's authors define them."""
import numpy as np
import pandas as pd

from load import rds

PHQ = {"Not at all": 0, "Several days": 1, "More than half the days": 2, "Nearly every day": 3}
PSS = {"Never": 0, "Almost never": 1, "Sometime": 2, "Fairly often": 3, "Very often": 4}
UCLA = {"Never": 1, "Rarely": 2, "Sometimes": 3, "Often": 4}

# (file, answer map, items, reverse-scored items, reverse base, name)
INSTRUMENTS = [
    # PHQ-9 (Kroenke et al. 2001): items 1-9 summed, 0-27. Item 10 (difficulty) is not part of the score.
    ("PHQ-9", PHQ, range(1, 10), (), None, "phq9"),
    # PSS-10 (Cohen et al. 1983): 0-4 per item, items 4, 5, 7, 8 reversed, 0-40.
    ("PerceivedStressScale", PSS, range(1, 11), (4, 5, 7, 8), 4, "pss"),
    # Flourishing Scale (Diener et al. 2010): 8 items 1-7 summed, 8-56.
    ("FlourishingScale", None, range(1, 9), (), None, "flourishing"),
    # UCLA Loneliness v3 (Russell 1996): 20 items 1-4, items 1,5,6,9,10,15,16,19,20 reversed, 20-80.
    ("LonelinessScale", UCLA, range(1, 21), (1, 5, 6, 9, 10, 15, 16, 19, 20), 5, "loneliness"),
]


def score(file, amap, items, reverse, base, max_missing=1):
    d = rds("survey", file)
    cols = [f"Q{i}" for i in items]
    vals = d[cols].apply(lambda s: pd.to_numeric(s.map(amap) if amap else s, errors="coerce"))
    for i in reverse:
        vals[f"Q{i}"] = base - vals[f"Q{i}"]
    n_missing = vals.isna().sum(axis=1)
    # one skipped item: prorate (mean of answered items x number of items); more: no score
    total = vals.mean(axis=1) * len(cols)
    total[n_missing > max_missing] = np.nan
    out = pd.DataFrame({"uid": d.uid, "type": d["type"], "score": total})
    # a few students answered twice; keep the mean
    return out.groupby(["uid", "type"], as_index=False)["score"].mean()


def all_surveys():
    """One row per student: phq9_pre, phq9_post, pss_pre, ... (NaN when not answered)."""
    wide = []
    for file, amap, items, reverse, base, name in INSTRUMENTS:
        s = score(file, amap, items, reverse, base)
        s = s.pivot(index="uid", columns="type", values="score")
        s.columns = [f"{name}_{c}" for c in s.columns]
        wide.append(s)
    return pd.concat(wide, axis=1).reset_index()
