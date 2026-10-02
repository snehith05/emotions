"""
Raw sensor streams -> one row per student per day.

Paper definitions reproduced (Wang et al. 2014, "StudentLife App and Sensing System"):
  activity duration   10-min windows whose share of non-stationary inferences exceeds a threshold, summed
  conversation        number and total duration of detected conversations, per day and per epoch
  co-location         number of distinct Bluetooth devices seen in a day
  traveled distance   distance between consecutive GPS fixes in a day (outdoor mobility)
  indoor mobility     time moving while inside a campus building (approximation - see README)
  sleep               linear "best effort sleep" model: sleep = sum(alpha_i * F_i), alpha_i >= 0, on
                      light, phone-usage, activity and sound features (Chen et al. 2013)
Epochs: day 9am-6pm, evening 6pm-12am, night 12am-9am, local time.
"""
import numpy as np
import pandas as pd
from scipy.optimize import nnls

import load
from config import TZ, TERM_START, TERM_END, WINDOW_S, ACTIVE_RATIO, MIN_DAY_WINDOWS, EPOCHS

T0 = pd.Timestamp("2013-03-20", tz=TZ).timestamp()        # anything outside this range is a clock error
T1 = pd.Timestamp("2013-09-01", tz=TZ).timestamp()


def local(ts):
    return pd.to_datetime(np.asarray(ts), unit="s", utc=True).tz_convert(TZ)


def epoch_of(hour):
    h = np.asarray(hour)
    return np.select([(h >= 9) & (h < 18), h >= 18], ["day", "evening"], "night")


# ------------------------------------------------------------------ 10-minute windows
@load.cached("windows_activity_audio")
def windows():
    """Per student and 10-min window: counts of each activity code (0 stationary, 1 walking, 2 running,
    3 unknown) and audio code (0 silence, 1 voice, 2 noise, 3 unknown)."""
    nwin = int((T1 - T0) // WINDOW_S) + 1
    frames = {}
    users = None
    for name, col, prefix in (("activity", "activity_inference", "act"), ("audio", "audio_inference", "aud")):
        print(f"  reading {name} (this can take a few minutes)...", flush=True)
        uidc, cats, ts, code = load.point_codes(name, col)
        ok = (ts >= T0) & (ts < T1) & np.isfinite(code) & (code >= 0) & (code <= 3)
        w = ((ts[ok] - T0) // WINDOW_S).astype(np.int64)
        key = (uidc[ok].astype(np.int64) * nwin + w) * 4 + code[ok].astype(np.int64)
        counts = np.bincount(key, minlength=len(cats) * nwin * 4).reshape(len(cats) * nwin, 4)
        del uidc, ts, code, ok, w, key
        idx = np.flatnonzero(counts.sum(1))
        df = pd.DataFrame(counts[idx], columns=[f"{prefix}{k}" for k in range(4)])
        df["uid"] = np.asarray(cats)[idx // nwin]
        df["win"] = idx % nwin
        frames[name] = df
        users = cats
    out = frames["activity"].merge(frames["audio"], on=["uid", "win"], how="outer").fillna(0)
    out["ts"] = T0 + out["win"] * WINDOW_S
    t = local(out["ts"])
    out["date"] = t.date
    out["hour"] = t.hour + t.minute / 60
    out["epoch"] = epoch_of(t.hour)
    n_act = out[["act0", "act1", "act2"]].sum(1)
    out["act_seen"] = n_act > 0
    out["moving"] = np.where(n_act > 0, (out.act1 + out.act2) / n_act.clip(lower=1), np.nan)
    out["active"] = out["moving"] > ACTIVE_RATIO
    n_aud = out[["aud0", "aud1", "aud2"]].sum(1)
    out["voice_noise"] = np.where(n_aud > 0, (out.aud1 + out.aud2) / n_aud.clip(lower=1), np.nan)
    return out.sort_values(["uid", "win"]).reset_index(drop=True)


# ------------------------------------------------------------------ daily sensing features
def _by_epoch(df, value, name, how="sum"):
    """Daily total + one column per epoch."""
    g = df.groupby(["uid", "date"])[value].agg(how).rename(name)
    e = df.groupby(["uid", "date", "epoch"])[value].agg(how).unstack("epoch")
    e.columns = [f"{name}_{c}" for c in e.columns]
    return pd.concat([g, e], axis=1)


def activity_daily(win):
    w = win[win.act_seen].copy()
    w["active_min"] = w["active"] * (WINDOW_S / 60)
    out = _by_epoch(w, "active_min", "activity_min").fillna(0)
    out["coverage_windows"] = w.groupby(["uid", "date"]).size()
    return out


def conversation_daily():
    c = load.intervals("conversation")
    t = local(c.start)
    c["date"], c["epoch"] = t.date, epoch_of(t.hour)
    c["dur_min"] = (c.end - c.start) / 60
    c["one"] = 1
    freq = _by_epoch(c, "one", "conv_freq")
    dur = _by_epoch(c, "dur_min", "conv_min")
    return pd.concat([freq, dur], axis=1)


def colocation_daily():
    b = load.rds("sensing", "bluetooth")
    b["date"] = local(pd.to_numeric(b.timestamp)).date
    return b.groupby(["uid", "date"])["MAC"].nunique().rename("colocations").to_frame()


def gps_daily():
    g = load.rds("sensing", "gps")
    g = pd.DataFrame({"uid": g.uid, "ts": pd.to_numeric(g.timestamp), "lat": pd.to_numeric(g.latitude),
                      "lon": pd.to_numeric(g.longitude), "acc": pd.to_numeric(g.accuracy)})
    g = g[(g.acc <= 100) & g.lat.notna()].sort_values(["uid", "ts"])
    near = 111 * np.hypot(g.lat - CAMPUS[0], (g.lon - CAMPUS[1]) * np.cos(np.radians(CAMPUS[0]))) <= CAMPUS_KM
    g = g[near]                                                           # trips away are not campus mobility
    lat, lon = np.radians(g.lat.to_numpy()), np.radians(g.lon.to_numpy())
    dlat, dlon = np.diff(lat, prepend=lat[:1]), np.diff(lon, prepend=lon[:1])
    a = np.sin(dlat / 2) ** 2 + np.cos(lat) * np.cos(np.roll(lat, 1)) * np.sin(dlon / 2) ** 2
    d = 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))                   # km
    dt = np.diff(g.ts.to_numpy(), prepend=g.ts.to_numpy()[:1])
    same = g.uid.to_numpy() == np.roll(g.uid.to_numpy(), 1)
    d = np.where(same & (dt > 0) & (d / np.maximum(dt, 1) < 0.05), d, 0.0)  # drop jumps > 180 km/h
    t = local(g.ts)
    g["km"], g["date"], g["epoch"] = d, t.date, epoch_of(t.hour)
    return _by_epoch(g, "km", "distance_km")


def indoor_daily(win):
    """Indoor mobility (approximation): active 10-min windows while Wi-Fi places the phone inside a
    campus building. The paper used its own campus access-point map, which was not released."""
    wl = load.rds("sensing", "wifi_location")
    ts = pd.to_numeric(wl.timestamp)
    wl = pd.DataFrame({"uid": wl.uid, "win": ((ts - T0) // WINDOW_S).astype("int64"),
                       "inside": wl.location.astype(str).str.startswith("in[")})
    inside = wl.groupby(["uid", "win"])["inside"].mean().gt(0.5).rename("inside").reset_index()
    w = win[win.act_seen].merge(inside, on=["uid", "win"], how="left")
    w["indoor_moving_min"] = (w["active"] & w["inside"].fillna(False).astype(bool)) * (WINDOW_S / 60)
    return _by_epoch(w, "indoor_moving_min", "indoor_mobility_min").fillna(0)


def darkness_daily():
    """Minutes the phone's light sensor reported darkness, by epoch. A phone in a pocket or bag is
    also 'dark' - this is NOT a measure of the person's light exposure."""
    d = load.intervals("dark")
    rows = []
    for _, r in d.iterrows():                                        # split intervals at hour boundaries
        s, e = r.start, r.end
        while s < e:
            nxt = min(e, (s // 3600 + 1) * 3600)
            rows.append((r.uid, s, (nxt - s) / 60))
            s = nxt
    x = pd.DataFrame(rows, columns=["uid", "ts", "dark_min"])
    t = local(x.ts)
    x["date"], x["epoch"], x["hour"] = t.date, epoch_of(t.hour), t.hour
    out = _by_epoch(x, "dark_min", "dark_min")
    morning = x[(x.hour >= 6) & (x.hour < 10)].groupby(["uid", "date"])["dark_min"].sum()
    out["dark_morning_min"] = morning
    return out


# ------------------------------------------------------------------ sleep (best-effort sleep model)
NIGHT_START_H = 18          # a night runs from 6pm the day before to noon of its date
NIGHT_LEN_W = 18 * 6        # 108 ten-minute windows
SLEEP_FEATURES = ["sl_dark_h", "sl_lock_h", "sl_charge_h", "sl_stationary_h", "sl_silence_h", "quiet_h"]
CAMPUS = (43.7044, -72.2887)    # Dartmouth Green - "distance a student travels around campus"
CAMPUS_KM = 20


def _night_index(ts):
    """For each timestamp: (date of the night it belongs to, window number within that night)."""
    t = local(ts) - pd.Timedelta(hours=NIGHT_START_H)
    date = (t + pd.Timedelta(days=1)).date
    pos = ((t.hour * 60 + t.minute) // 10).to_numpy()
    return date, pos


def _longest_interval(iv):
    """Longest (merged) interval per night, in hours, from [uid, start, end] intervals."""
    rows = []
    for (uid,), g in iv.sort_values("start").groupby(["uid"]):
        s0, e0 = None, None
        for s, e in zip(g.start.to_numpy(), g.end.to_numpy()):
            if s0 is not None and s - e0 <= 300:                 # merge gaps of 5 min or less
                e0 = max(e0, e)
                continue
            if s0 is not None:
                rows.append((uid, s0, e0))
            s0, e0 = s, e
        if s0 is not None:
            rows.append((uid, s0, e0))
    m = pd.DataFrame(rows, columns=["uid", "start", "end"])
    out = []
    for uid, s, e in m.itertuples(index=False):                  # clip each to the nights it overlaps
        d0 = (local([s])[0] - pd.Timedelta(hours=NIGHT_START_H)).normalize() + pd.Timedelta(hours=NIGHT_START_H)
        while d0.timestamp() < e:
            n0, n1 = d0.timestamp(), d0.timestamp() + NIGHT_LEN_W * WINDOW_S
            ov = min(e, n1) - max(s, n0)
            if ov > 0:
                out.append((uid, (d0 + pd.Timedelta(days=1)).date(), ov / 3600))
            d0 = d0 + pd.Timedelta(days=1)
    df = pd.DataFrame(out, columns=["uid", "date", "h"])
    return df.groupby(["uid", "date"])["h"].max()


def _longest_run(flags, bridge=3):
    """Longest run of True in a 0/1/NaN series; gaps of missing data up to `bridge` windows are bridged
    (duty-cycled sensors leave holes), longer gaps break the run. Returns (length, start, end)."""
    f = np.asarray(flags, dtype=float)
    miss = np.isnan(f)
    good = np.where(miss, 0, f) > 0
    # bridge short missing gaps that sit between quiet windows
    i, n = 0, len(f)
    while i < n:
        if miss[i]:
            j = i
            while j < n and miss[j]:
                j += 1
            if j - i <= bridge and i > 0 and j < n and good[i - 1] and good[j]:
                good[i:j] = True
            i = j
        else:
            i += 1
    best, start, cur, cs = 0, 0, 0, 0
    for k, g in enumerate(good):
        if g:
            if cur == 0:
                cs = k
            cur += 1
            if cur > best:
                best, start = cur, cs
        else:
            cur = 0
    return best, start, start + best


@load.cached("sleep_nights")
def sleep_nights(win):
    """Per student-night: the BES inputs (longest dark, lock, charge, stationary and silent stretches,
    in hours) and the timing of the longest quiet stretch (candidate bedtime / wake time)."""
    w = win.copy()
    date, pos = _night_index(w.ts)
    w["night"], w["pos"] = date, pos
    w = w[w.pos < NIGHT_LEN_W]
    w["still"] = np.where(w.act_seen, (w.moving <= 0.05).astype(float), np.nan)
    w["silent"] = np.where(w.voice_noise.notna(), (w.voice_noise <= 0.1).astype(float), np.nan)
    rows = []
    for (uid, night), g in w.groupby(["uid", "night"]):
        grid = np.full((NIGHT_LEN_W, 2), np.nan)
        grid[g.pos.to_numpy(), 0] = g.still.to_numpy()
        grid[g.pos.to_numpy(), 1] = g.silent.to_numpy()
        st, _, _ = _longest_run(grid[:, 0])
        si, _, _ = _longest_run(grid[:, 1])
        both = np.where(np.isnan(grid[:, 0]) | np.isnan(grid[:, 1]), np.nan, grid[:, 0] * grid[:, 1])
        q, qs, qe = _longest_run(both)
        rows.append({"uid": uid, "date": night, "sl_stationary_h": st / 6, "sl_silence_h": si / 6,
                     "quiet_h": q / 6, "night_windows": int(np.isfinite(grid[:, 0]).sum()),
                     # clock hours: 24.5 = 12:30am, 31 = 7am
                     "bedtime_h": NIGHT_START_H + qs / 6 if q else np.nan,
                     "waketime_h": NIGHT_START_H + qe / 6 if q else np.nan})
    out = pd.DataFrame(rows).set_index(["uid", "date"])
    for name, col in (("dark", "sl_dark_h"), ("phonelock", "sl_lock_h"), ("phonecharge", "sl_charge_h")):
        out[col] = _longest_interval(load.intervals(name))
    out[["sl_dark_h", "sl_lock_h", "sl_charge_h"]] = out[["sl_dark_h", "sl_lock_h", "sl_charge_h"]].fillna(0)
    return out.reset_index()


def fit_sleep_model(nights, sr):
    """Fit sleep = sum(alpha_i * F_i), alpha_i >= 0 (the paper's model form) against the students' own
    'hours slept last night' answers, which stand in for the Jawbone ground truth the paper used.
    Accuracy is measured on students the weights never saw (leave-students-out)."""
    d = nights.merge(sr, on=["uid", "date"]).dropna(subset=SLEEP_FEATURES + ["sleep_hours_sr"])
    d = d[d.night_windows >= 54]                                   # at least 9 of the 18 night hours seen
    # a constant term is added (alpha_0 >= 0): without it the model is worse than predicting the
    # average (75 vs 48 min mean error, leave-students-out) - a documented deviation from the paper
    X = np.c_[d[SLEEP_FEATURES].to_numpy(), np.ones(len(d))]
    y, groups = d["sleep_hours_sr"].to_numpy(), d["uid"].to_numpy()
    users = np.unique(groups)
    rng = np.random.default_rng(0)
    folds = np.array_split(rng.permutation(users), 5)
    pred = np.full(len(y), np.nan)
    for test in folds:
        tr = ~np.isin(groups, test)
        a, _ = nnls(X[tr], y[tr])
        pred[~tr] = X[~tr] @ a
    alpha, _ = nnls(X, y)
    err = pred - y
    report = {"nights": int(len(y)), "students": int(len(users)), "alpha": dict(zip(SLEEP_FEATURES + ["constant"], alpha.round(3))),
              "mae_min_predict_mean": float(np.mean(np.abs(y - y.mean())) * 60),
              "mae_min": float(np.mean(np.abs(err)) * 60), "within_32min": float(np.mean(np.abs(err) <= 32 / 60)),
              "r": float(np.corrcoef(pred, y)[0, 1])}
    return alpha, report, d.assign(sleep_pred_cv=pred)


# ------------------------------------------------------------------ EMA per day
PAM_GRID = {  # PAM score -> (valence column 1-4, arousal row 1-4), Pollak et al. 2011, Figure 3
    1: (1, 1), 2: (1, 2), 3: (2, 1), 4: (2, 2), 5: (1, 3), 6: (1, 4), 7: (2, 3), 8: (2, 4),
    9: (3, 1), 10: (3, 2), 11: (4, 1), 12: (4, 2), 13: (3, 3), 14: (3, 4), 15: (4, 3), 16: (4, 4)}
SIGNED = {1: -2, 2: -1, 3: 1, 4: 2}


def ema_daily():
    p = load.pam()
    p = p[p.pam.between(1, 16)]
    p["valence"] = p.pam.map(lambda s: SIGNED[PAM_GRID[int(s)][0]])
    p["arousal"] = p.pam.map(lambda s: SIGNED[PAM_GRID[int(s)][1]])
    p["date"] = local(p.ts).date
    pam = p.groupby(["uid", "date"]).agg(pa=("pam", "mean"), valence=("valence", "mean"),
                                         arousal=("arousal", "mean"), n_pam=("pam", "size"))
    s = load.stress()
    s["date"] = local(s.ts).date
    st = s.groupby(["uid", "date"]).agg(stress=("stress", "mean"), n_stress=("stress", "size"))
    sl = load.sleep_ema()
    sl["date"] = local(sl.ts).date
    sl = sl.groupby(["uid", "date"])[["sleep_hours_sr", "sleep_quality_sr"]].mean()
    ex = load.exercise_ema()
    ex["date"] = local(ex.ts).date
    ex = ex.groupby(["uid", "date"])[["exercise_min_code", "walk_min_code"]].mean()
    return pd.concat([pam, st, sl, ex], axis=1)


# ------------------------------------------------------------------ everything, one row per student-day
@load.cached("daily")
def build_daily():
    print("Building 10-minute windows...", flush=True)
    win = windows()
    print("Daily features...", flush=True)
    parts = [activity_daily(win), conversation_daily(), colocation_daily(), gps_daily(), indoor_daily(win),
             darkness_daily(), ema_daily()]
    daily = pd.concat(parts, axis=1)
    daily.index = daily.index.set_names(["uid", "date"])
    daily = daily.reset_index()
    print("Sleep model...", flush=True)
    nights = sleep_nights(win)
    sr = daily[["uid", "date", "sleep_hours_sr"]].dropna()
    alpha, report, _ = fit_sleep_model(nights, sr)
    nights["sleep_h"] = (nights[SLEEP_FEATURES].to_numpy() @ alpha[:-1] + alpha[-1]).clip(0, 14)
    nights.loc[nights.night_windows < 54, "sleep_h"] = np.nan          # too little of the night observed
    daily = daily.merge(nights[["uid", "date", "sleep_h", "bedtime_h", "waketime_h", "night_windows"]
                               + SLEEP_FEATURES], on=["uid", "date"], how="left")
    dl = load.deadlines()
    daily = daily.merge(dl, on=["uid", "date"], how="left")
    start, end = pd.Timestamp(TERM_START).date(), pd.Timestamp(TERM_END).date()
    daily = daily[(daily.date >= start) & (daily.date <= end)].copy()
    daily["valid_day"] = daily["coverage_windows"].fillna(0) >= MIN_DAY_WINDOWS
    daily["term_day"] = [(d - start).days + 1 for d in daily.date]
    daily["week"] = (daily.term_day - 1) // 7 + 1
    daily["weekday"] = [d.weekday() for d in daily.date]
    return {"daily": daily.sort_values(["uid", "date"]).reset_index(drop=True), "sleep_model": report}
