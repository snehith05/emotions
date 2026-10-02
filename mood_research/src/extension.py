"""
PART 2 - OUR AI/ML EXTENSION (not part of the original paper).

Question: can the previous night's sleep, the previous day's activity / sociability / phone darkness and
the person's recent mood PREDICT how they feel tomorrow?

Target   daily mean positive affect from the PAM EMA (1-16), the paper's mood measure. Secondary: stress.
Features only what is known before the target day starts (yesterday's sensing and mood, last night's
         sleep, the calendar, known deadlines). Nothing from the target day itself.
Splits   (A) temporal, within each student: first 60% of their days train, next 20% validation,
             last 20% test - "can it forecast the rest of the term for people it already knows?"
         (B) leave-students-out (5 folds by student) - "does it work for someone it has never seen?"
Models   baselines (training mean, person mean, yesterday's mood, person's running mean) vs
         ridge regression, random forest, gradient boosting.
Then     permutation importance, ablations by feature group, a within-person mixed model for
         associations, and error analysis.

Prediction is not causation: a feature can predict mood because of a common cause (e.g. midterms cut
sleep AND lower mood). Only an experiment could show that changing sleep changes mood.
"""
import json
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import RESULTS, FIGURES, SEED

try:        # tree models need compiled files that Windows Smart App Control can block
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.inspection import permutation_importance
    TREES = True
except ImportError as err:
    TREES = False
    print(f"NOTE: tree models unavailable here ({err}).\n"
          "      Running ridge + baselines only. Run the Colab notebook for the full model set.")

    def permutation_importance(est, X, y, scoring=None, n_repeats=20, random_state=None, n_jobs=None):
        """Same idea as sklearn's: how much worse (MAE) the model gets when one column is shuffled."""
        from types import SimpleNamespace
        rng = np.random.default_rng(random_state)
        base = mean_absolute_error(y, est.predict(X))
        imp = np.zeros((X.shape[1], n_repeats))
        for j, c in enumerate(X.columns):
            for r in range(n_repeats):
                Xp = X.copy()
                Xp[c] = rng.permutation(Xp[c].to_numpy())
                imp[j, r] = mean_absolute_error(y, est.predict(Xp)) - base
        return SimpleNamespace(importances_mean=imp.mean(1), importances_std=imp.std(1), importances=imp)
from features import build_daily

warnings.filterwarnings("ignore", category=FutureWarning)

LAG_SENSING = ["activity_min", "activity_min_evening", "conv_min", "conv_freq", "colocations", "distance_km",
               "indoor_mobility_min", "dark_min_day", "dark_morning_min", "exercise_min_code", "walk_min_code"]
GROUPS = {
    "mood history": ["pa_lag1", "pa_lag3", "pa_past_mean", "stress_lag1", "valence_lag1", "arousal_lag1"],
    "sleep": ["sleep_h", "bedtime_h", "waketime_h", "sleep_hours_sr", "sleep_quality_sr"],
    "activity & social": [f"{c}_lag1" for c in LAG_SENSING if c not in ("dark_min_day", "dark_morning_min")],
    "phone darkness (light proxy)": ["dark_min_day_lag1", "dark_morning_min_lag1"],
    "calendar & workload": ["weekday", "weekend", "week", "deadlines", "deadlines_next2"],
}
FEATURES = [f for g in GROUPS.values() for f in g]


# ------------------------------------------------------------------ dataset
def make_dataset(daily, target="pa"):
    """One row per student-day that has the target; every feature comes from before that day."""
    rows = []
    for uid, g in daily.groupby("uid"):
        g = g.set_index("date").sort_index()
        full = pd.date_range(min(g.index), max(g.index), freq="D").date
        g = g.reindex(full)                                    # calendar days, so lag 1 = yesterday
        out = pd.DataFrame(index=g.index)
        out["uid"] = uid
        out["y"] = g[target]
        out["n_target"] = g["n_pam"] if target == "pa" else g["n_stress"]
        for c in ("pa", "stress", "valence", "arousal"):
            out[f"{c}_lag1"] = g[c].shift(1)
        out["pa_lag3"] = g["pa"].shift(1).rolling(3, min_periods=1).mean()
        out["pa_past_mean"] = g["pa"].shift(1).expanding().mean()
        out["y_past_mean"] = g[target].shift(1).expanding().mean()
        out["y_lag1"] = g[target].shift(1)
        valid = g["valid_day"].fillna(False).astype(bool)
        for c in LAG_SENSING:
            v = g[c].where(valid) if c not in ("exercise_min_code", "walk_min_code") else g[c]
            out[f"{c}_lag1"] = v.shift(1)
        for c in GROUPS["sleep"]:
            out[c] = g[c]                                      # last night's sleep (night ending this date)
        out["deadlines"] = g["deadlines"]
        out["deadlines_next2"] = g["deadlines"].shift(-1).fillna(0) + g["deadlines"].shift(-2).fillna(0)
        idx = pd.to_datetime(pd.Series(g.index, index=g.index))
        out["weekday"] = idx.dt.weekday
        out["weekend"] = (out["weekday"] >= 5).astype(int)
        out["week"] = g["week"].ffill().bfill()
        out["date"] = g.index
        rows.append(out[out["y"].notna()])
    return pd.concat(rows, ignore_index=True)


def temporal_split(data, train=0.6, val=0.2, min_days=10):
    """Per student, by date. Students with too few days are used for training only."""
    part = pd.Series("train", index=data.index)
    for uid, g in data.groupby("uid"):
        if len(g) < min_days:
            continue
        g = g.sort_values("date")
        n = len(g)
        a, b = int(round(n * train)), int(round(n * (train + val)))
        part[g.index[a:b]] = "val"
        part[g.index[b:]] = "test"
    return part


# ------------------------------------------------------------------ models
def models():
    ridge = {"ridge": lambda: make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                                            Ridge(alpha=10.0))}
    if not TREES:
        return ridge
    return {
        **ridge,
        "random_forest": lambda: make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                                               RandomForestRegressor(n_estimators=400, min_samples_leaf=10,
                                                                     max_features=0.4, n_jobs=-1, random_state=SEED)),
        "gradient_boosting": lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.04, max_depth=3,
                                                                   min_samples_leaf=20, l2_regularization=1.0,
                                                                   random_state=SEED),
    }


GRID = {"ridge": [{"ridge__alpha": a} for a in (1, 10, 100)],
        "random_forest": [{"randomforestregressor__min_samples_leaf": m} for m in (5, 10, 25)],
        "gradient_boosting": [{"learning_rate": lr, "max_depth": d} for lr in (0.03, 0.08) for d in (2, 3)]}


def metrics(y, p):
    return {"MAE": mean_absolute_error(y, p), "RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "R2": r2_score(y, p), "Spearman": stats.spearmanr(y, p).statistic, "n": int(len(y))}


def baselines(tr, te):
    gm = tr["y"].mean()
    pm = tr.groupby("uid")["y"].mean()
    person = te["uid"].map(pm).fillna(gm)
    return {"baseline: training mean": np.full(len(te), gm),
            "baseline: person mean (training days)": person.to_numpy(),
            "baseline: yesterday's value": te["y_lag1"].fillna(te["y_past_mean"]).fillna(person).to_numpy(),
            "baseline: person running mean": te["y_past_mean"].fillna(gm).to_numpy()}


def tune(name, tr, va, feats):
    best, best_mae = None, np.inf
    for params in GRID[name]:
        m = models()[name]()
        m.set_params(**params)
        m.fit(tr[feats], tr["y"])
        mae = mean_absolute_error(va["y"], m.predict(va[feats]))
        if mae < best_mae:
            best, best_mae = params, mae
    return best


def cluster_bootstrap(te, pred_a, pred_b, n=2000, seed=SEED):
    """95% CI of MAE(a) - MAE(b), resampling whole students (days of one person are not independent)."""
    rng = np.random.default_rng(seed)
    err = pd.DataFrame({"uid": te["uid"].to_numpy(), "a": np.abs(pred_a - te["y"].to_numpy()),
                        "b": np.abs(pred_b - te["y"].to_numpy())})
    per = err.groupby("uid")[["a", "b"]].agg(["sum", "count"])
    users = per.index.to_numpy()
    diffs = []
    for _ in range(n):
        pick = per.loc[rng.choice(users, len(users))]
        cnt = pick[("a", "count")].sum()
        diffs.append((pick[("a", "sum")].sum() - pick[("b", "sum")].sum()) / cnt)
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


# ------------------------------------------------------------------ experiments
def run_temporal(data, feats, label, tuned=None):
    part = temporal_split(data)
    tr, va, te = data[part == "train"], data[part == "val"], data[part == "test"]
    preds, params = {}, {}
    for name in models():
        params[name] = (tuned or {}).get(name) or tune(name, tr, va, feats)
        m = models()[name]()
        m.set_params(**params[name])
        trva = pd.concat([tr, va])
        m.fit(trva[feats], trva["y"])
        preds[name] = m.predict(te[feats])
    base = baselines(pd.concat([tr, va]), te)
    rows = [{"split": "temporal", "features": label, "model": k, **metrics(te["y"], v)} for k, v in {**base, **preds}.items()]
    return pd.DataFrame(rows), preds, base, te, params, pd.concat([tr, va])


def run_loso(data, feats, label, params):
    gkf = GroupKFold(n_splits=5)
    pred = {k: np.full(len(data), np.nan) for k in list(models()) + ["baseline: training mean",
                                                                     "baseline: yesterday's value",
                                                                     "baseline: person running mean"]}
    for tr_i, te_i in gkf.split(data, groups=data["uid"]):
        tr, te = data.iloc[tr_i], data.iloc[te_i]
        for name in models():
            m = models()[name]()
            m.set_params(**params[name])
            m.fit(tr[feats], tr["y"])
            pred[name][te_i] = m.predict(te[feats])
        b = baselines(tr, te)
        for k in ("baseline: training mean", "baseline: yesterday's value", "baseline: person running mean"):
            pred[k][te_i] = b[k]
    rows = [{"split": "leave-students-out", "features": label, "model": k, **metrics(data["y"], v)}
            for k, v in pred.items()]
    return pd.DataFrame(rows), pred


def mixed_model(daily):
    """Within-person associations (statsmodels MixedLM, random intercept per student).
    Each predictor is split into the student's own average (between) and the day's deviation from it
    (within). 'Within' answers: on nights a student slept more than usual, was their next-day mood higher?
    'Between' answers: do students who sleep more on average have higher average mood?"""
    import statsmodels.formula.api as smf
    d = make_dataset(daily, "pa")
    preds = {"sleep_h": "sleep (inferred, h)", "activity_min_lag1": "activity yesterday (min)",
             "conv_min_lag1": "conversation yesterday (min)", "dark_morning_min_lag1": "phone dark 6-10am yesterday (min)",
             "sleep_hours_sr": "sleep (self-report, h)"}
    d = d.dropna(subset=["y"])
    for c in preds:
        m = d.groupby("uid")[c].transform("mean")
        sd = d[c].std()
        d[f"{c}_w"] = (d[c] - m) / sd                           # per 1 SD, so effects are comparable
        d[f"{c}_b"] = (m - d[c].mean()) / sd
    out = []
    for c, label in preds.items():
        # standard within-between ("hybrid") model; yesterday's mood is left out on purpose - a lagged
        # outcome next to a per-person intercept biases short panels (Nickell bias) and broke the fit
        dd = d.dropna(subset=[c])
        f = f"y ~ {c}_w + {c}_b + weekend + week"
        fit = smf.mixedlm(f, dd, groups=dd["uid"]).fit(reml=True, method="powell")   # lbfgs got stuck at a singular start
        ci = fit.conf_int()
        for part in ("w", "b"):
            k = f"{c}_{part}"
            out.append({"predictor": label, "level": "within-person" if part == "w" else "between-person",
                        "coef_per_SD": fit.params[k], "ci_low": ci.loc[k, 0], "ci_high": ci.loc[k, 1],
                        "p": fit.pvalues[k], "n_days": len(dd), "n_students": dd.uid.nunique()})
    return pd.DataFrame(out)


# ------------------------------------------------------------------ figures
def fig_models(res):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=False)
    for ax, split in zip(axes, ("temporal", "leave-students-out")):
        r = res[(res.split == split) & (res.features == "all features")
                & (res.target == "positive affect (1-16)")].sort_values("MAE")
        colors = ["#9a988f" if m.startswith("baseline") else "#2a78d6" for m in r.model]
        ax.barh(r.model, r.MAE, color=colors)
        for i, (mae, r2) in enumerate(zip(r.MAE, r.R2)):
            ax.text(mae + 0.02, i, f"{mae:.2f}  (R² {r2:.2f})", va="center", fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("MAE on held-out days (PAM positive affect, 1-16) - lower is better")
        ax.set_title({"temporal": "(A) later days of known students", "leave-students-out": "(B) students never seen"}[split])
    fig.suptitle("Predicting tomorrow's mood: models (blue) vs simple baselines (grey)")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig4_model_comparison.png", dpi=150)
    plt.close(fig)


def fig_pred(te, pred, name):
    fig, ax = plt.subplots(figsize=(5.4, 5))
    ax.scatter(te["y"], pred, s=10, alpha=0.4)
    ax.plot([1, 16], [1, 16], color="grey", ls="--", lw=1)
    ax.set_xlabel("actual daily mean PAM positive affect")
    ax.set_ylabel(f"predicted ({name})")
    ax.set_title("Held-out days (temporal split)")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig5_predicted_vs_actual.png", dpi=150)
    plt.close(fig)


def fig_importance(imp):
    imp = imp.sort_values("importance").tail(15)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(imp.feature, imp.importance, xerr=imp["std"], color="#2a78d6")
    ax.set_xlabel("increase in MAE when the feature is shuffled (test set)")
    ax.set_title("Permutation importance (what the model relies on - not causes)")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig6_permutation_importance.png", dpi=150)
    plt.close(fig)


def fig_mixed(mm):
    d = mm.copy()
    d["label"] = d.predictor + " - " + d.level
    d = d.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.errorbar(d.coef_per_SD, range(len(d)), xerr=[d.coef_per_SD - d.ci_low, d.ci_high - d.coef_per_SD],
                fmt="o", color="#2a78d6", capsize=3)
    ax.axvline(0, color="grey", lw=1)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(d.label, fontsize=8)
    ax.set_xlabel("change in next-day positive affect per 1 SD (95% CI)")
    ax.set_title("Mixed model: associations with next-day mood\n(adjusted for weekend and term week)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig7_mixed_model_associations.png", dpi=150)
    plt.close(fig)


def fig_errors(te, pred):
    e = te.assign(abs_err=np.abs(pred - te["y"].to_numpy()))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    per = e.groupby("uid").agg(mae=("abs_err", "mean"), sd=("y", "std"), n=("y", "size")).sort_values("mae")
    axes[0].bar(range(len(per)), per.mae, color="#2a78d6")
    axes[0].set_xticks(range(len(per)))
    axes[0].set_xticklabels(per.index, rotation=90, fontsize=6)
    axes[0].set_ylabel("MAE")
    axes[0].set_title("Error per student (test days)")
    axes[1].scatter(per.sd, per.mae)
    axes[1].set_xlabel("how much the student's mood varied (SD on test days)")
    axes[1].set_ylabel("MAE")
    r = stats.spearmanr(per.sd, per.mae, nan_policy="omit").statistic
    axes[1].set_title(f"Harder for students whose mood swings (rho = {r:.2f})")
    e["n_bin"] = pd.cut(e["n_target"], [0, 1, 2, 4, 100], labels=["1", "2", "3-4", "5+"])
    g = e.groupby("n_bin", observed=True)["abs_err"].mean()
    axes[2].bar(g.index.astype(str), g.values, color="#2a78d6")
    axes[2].set_xlabel("PAM answers that day (the target is their mean)")
    axes[2].set_ylabel("MAE")
    axes[2].set_title("Days with one answer are noisier targets")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig8_error_analysis.png", dpi=150)
    plt.close(fig)
    return per


# ------------------------------------------------------------------ main
def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    daily = build_daily()["daily"]
    data = make_dataset(daily, "pa")
    print(f"PA dataset: {len(data)} student-days, {data.uid.nunique()} students")
    res_t, preds, base, te, params, trva = run_temporal(data, FEATURES, "all features")
    res_l, pred_l = run_loso(data, FEATURES, "all features", params)
    results = [res_t, res_l]

    # ablations: one feature group at a time (temporal split), and everything except mood history
    for gname, feats in list(GROUPS.items()) + [("all except mood history",
                                                 [f for f in FEATURES if f not in GROUPS["mood history"]])]:
        r, *_ = run_temporal(data, feats, gname, tuned=params)
        results.append(r[~r.model.str.startswith("baseline")])
    # secondary target: stress
    sdata = make_dataset(daily, "stress")
    sdata = sdata.rename(columns={})
    rs, *_ = run_temporal(sdata, [f for f in FEATURES if f != "stress_lag1"] + ["y_lag1"], "all features")
    rs["target"] = "stress (1-5)"
    res = pd.concat(results, ignore_index=True)
    res["target"] = "positive affect (1-16)"
    res = pd.concat([res, rs], ignore_index=True)
    res.to_csv(RESULTS / "ml_results.csv", index=False)

    # best model vs best baseline on the temporal test set, with a student-level bootstrap CI
    rt = res_t.set_index("model")
    best_model = rt.loc[list(models())]["MAE"].idxmin()
    best_base = rt.loc[[m for m in rt.index if m.startswith("baseline")]]["MAE"].idxmin()
    ci = cluster_bootstrap(te, preds[best_model], base[best_base])

    # importance on the temporal test set
    m = models()[best_model]()
    m.set_params(**params[best_model])
    m.fit(trva[FEATURES], trva["y"])
    pi = permutation_importance(m, te[FEATURES], te["y"], scoring="neg_mean_absolute_error", n_repeats=20,
                                random_state=SEED, n_jobs=-1)
    imp = pd.DataFrame({"feature": FEATURES, "importance": pi.importances_mean, "std": pi.importances_std})
    imp.sort_values("importance", ascending=False).to_csv(RESULTS / "ml_permutation_importance.csv", index=False)

    mm = mixed_model(daily)
    mm.to_csv(RESULTS / "mixed_model_associations.csv", index=False)

    fig_models(res)
    fig_pred(te, preds[best_model], best_model)
    fig_importance(imp)
    fig_mixed(mm)
    per_student = fig_errors(te, preds[best_model])
    per_student.to_csv(RESULTS / "ml_error_by_student.csv")

    summary = {"dataset": {"student_days": int(len(data)), "students": int(data.uid.nunique()),
                           "test_days_temporal": int(len(te))},
               "tuned_params": params, "best_model": best_model, "best_baseline": best_base,
               "temporal": res_t.round(3).to_dict("records"), "leave_students_out": res_l.round(3).to_dict("records"),
               "mae_diff_best_model_minus_best_baseline_95ci": ci,
               "top_features": imp.sort_values("importance", ascending=False).head(8).round(4).to_dict("records")}
    (RESULTS / "ml_summary.json").write_text(json.dumps(summary, indent=1, default=float))
    pd.set_option("display.width", 200)
    print(res[res.features == "all features"].round(3).to_string(index=False))
    print(res[(res.split == "temporal") & (res.features != "all features")].round(3).to_string(index=False))
    print("best model:", best_model, "| best baseline:", best_base, "| MAE diff 95% CI:", np.round(ci, 3))
    print(imp.sort_values("importance", ascending=False).head(10).round(4).to_string(index=False))
    print(mm.round(3).to_string(index=False))
    return summary


if __name__ == "__main__":
    main()
