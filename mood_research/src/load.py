"""Read the raw StudentLife tables into tidy pandas frames (uid as 'u00'..'u59', unix seconds)."""
import json
import pickle

import numpy as np
import pandas as pd
import pyreadr

from config import RDS, ORIG, PROC


def _uid(s):
    return "u" + pd.Series(s).astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(2).to_numpy()


def cached(name):
    """Keep the result of an expensive step on disk (data/processed/<name>.pkl)."""
    def wrap(fn):
        def inner(*a, refresh=False, **kw):
            path = PROC / f"{name}.pkl"
            if path.exists() and not refresh:
                with open(path, "rb") as f:
                    return pickle.load(f)
            out = fn(*a, **kw)
            PROC.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
            return out
        return inner
    return wrap


def rds(sub, name):
    df = next(iter(pyreadr.read_r(str(RDS / sub / f"{name}.Rds")).values()))
    df["uid"] = _uid(df["uid"])
    return df


def intervals(name):
    """dark / phonelock / phonecharge / conversation: [uid, start, end] in unix seconds."""
    df = rds("sensing", name)
    out = pd.DataFrame({"uid": df["uid"], "start": pd.to_numeric(df["start_timestamp"]),
                        "end": pd.to_numeric(df["end_timestamp"])})
    return out[out.end > out.start].reset_index(drop=True)


def point_codes(name, col):
    """activity / audio: numpy arrays (uid codes, unix seconds, inference code) - kept as arrays because
    audio alone has ~99 million rows."""
    df = rds("sensing", name)
    uid = df["uid"].astype("category")
    out = (uid.cat.codes.to_numpy(np.int16), list(uid.cat.categories),
           pd.to_numeric(df["timestamp"]).to_numpy(np.float64), pd.to_numeric(df[col]).to_numpy(np.float32))
    del df
    return out


def pam():
    df = rds("EMA", "PAM")
    return pd.DataFrame({"uid": df.uid, "ts": pd.to_numeric(df.timestamp),
                         "pam": pd.to_numeric(df.picture_idx)}).dropna()


def stress():
    """Stress EMA from the original-release JSON (the Zenodo mirror lost the 'level' field).
    Answer codes (EMA_definition.json): 1 a little stressed, 2 definitely stressed, 3 stressed out,
    4 feeling good, 5 feeling great. Re-ordered into a 1 (feeling great) .. 5 (stressed out) scale."""
    order = {5: 1, 4: 2, 1: 3, 2: 4, 3: 5}
    rows = []
    for f in sorted((ORIG / "EMA" / "response" / "Stress").glob("Stress_u*.json")):
        uid = f.stem.split("_")[1]
        for r in json.loads(f.read_text()):
            try:
                lv = int(r["level"])
            except (KeyError, TypeError, ValueError):
                continue
            if lv in order:
                rows.append({"uid": uid, "ts": float(r["resp_time"]), "stress": order[lv]})
    return pd.DataFrame(rows)


SLEEP_HOURS = {1: 2.5, **{c: 3.0 + 0.5 * (c - 1) for c in range(2, 20)}}   # [1]<3 [2]3.5 ... [19]12


def sleep_ema():
    """'How many hours did you sleep last night?' (codes -> hours) and 'How would you rate your sleep'
    (1 very good .. 4 very bad)."""
    df = rds("EMA", "Sleep")
    out = pd.DataFrame({"uid": df.uid, "ts": pd.to_numeric(df.timestamp),
                        "sleep_hours_sr": pd.to_numeric(df.hour).map(SLEEP_HOURS),
                        "sleep_quality_sr": pd.to_numeric(df.rate)})
    return out.dropna(subset=["sleep_hours_sr"])


def exercise_ema():
    df = rds("EMA", "Exercise")
    return pd.DataFrame({"uid": df.uid, "ts": pd.to_numeric(df.timestamp),
                         "exercise_min_code": pd.to_numeric(df.exercise),
                         "walk_min_code": pd.to_numeric(df.walk)})


def grades():
    g = pd.read_csv(ORIG / "education" / "grades.csv", skipinitialspace=True)
    g.columns = [c.strip() for c in g.columns]
    return g.rename(columns={"gpa all": "gpa_overall", "gpa 13s": "gpa_spring", "cs 65": "cs65"})


def deadlines():
    """education/deadlines.csv: one row per student, one column per date, value = deadlines that day."""
    d = pd.read_csv(ORIG / "education" / "deadlines.csv")
    d = d.melt(id_vars="uid", var_name="date", value_name="deadlines")
    d["date"] = pd.to_datetime(d["date"]).dt.date
    return d
