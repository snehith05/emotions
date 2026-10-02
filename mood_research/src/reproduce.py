"""
PART 1 - ORIGINAL PAPER METHODOLOGY (Wang et al. 2014).

The paper averages each student's sensing data over the term and correlates those averages with the
pre- and post-term survey scores using Pearson's r (no correction for multiple tests). This script does
the same, puts our r and p next to the paper's for every row of Tables 3-9, and reproduces Table 4 and
the "Dartmouth term lifecycle" (Figure 5).
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import load
from config import RESULTS, FIGURES
from features import build_daily
from paper_values import PAPER, TABLE4, SLEEP_ACCURACY_MIN
from surveys import all_surveys

SENSING = ["sleep_h", "activity_min", "activity_min_day", "activity_min_evening", "activity_min_night",
           "conv_freq", "conv_freq_day", "conv_freq_evening", "conv_freq_night",
           "conv_min", "conv_min_day", "conv_min_evening", "conv_min_night", "colocations",
           "distance_km", "distance_km_day", "indoor_mobility_min", "indoor_mobility_min_day",
           "indoor_mobility_min_night", "dark_min", "dark_morning_min"]
MIN_DAYS = 7            # a student's sensing average needs at least a week of usable days


def bh_fdr(p):
    """Benjamini-Hochberg adjusted p-values (not in the paper - added because it tests many pairs)."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(adj)
    out[order] = np.minimum(adj, 1)
    return out


def student_level(daily):
    """One row per student: term means of the sensing features (valid days only), EMA means, surveys, GPA."""
    sens = daily[daily.valid_day]
    agg = sens.groupby("uid")[SENSING].mean()
    agg["activity_min_sd"] = sens.groupby("uid")["activity_min"].std()
    agg["valid_days"] = sens.groupby("uid").size()
    # sleep uses the nights the model could score
    agg["sleep_h"] = daily.groupby("uid")["sleep_h"].mean()
    agg["sleep_nights"] = daily.groupby("uid")["sleep_h"].count()
    agg.loc[agg.valid_days < MIN_DAYS, [c for c in agg.columns if c != "sleep_h" and c != "sleep_nights"]] = np.nan
    agg.loc[agg.sleep_nights < MIN_DAYS, "sleep_h"] = np.nan
    # EMA: mean over every response in the term, weighted by responses (daily means x counts)
    for col, n in (("pa", "n_pam"), ("stress", "n_stress")):
        d = daily.dropna(subset=[col])
        agg[col] = (d[col] * d[n]).groupby(d.uid).sum() / d.groupby("uid")[n].sum()
        agg[f"{n}_total"] = d.groupby("uid")[n].sum()
    agg["sleep_hours_sr"] = daily.groupby("uid")["sleep_hours_sr"].mean()
    out = agg.reset_index().merge(all_surveys(), on="uid", how="left")
    return out.merge(load.grades(), on="uid", how="left")


def pearson(df, x, y):
    d = df[[x, y]].dropna()
    if len(d) < 5:
        return np.nan, np.nan, len(d)
    r, p = stats.pearsonr(d[x], d[y])
    return r, p, len(d)


def compare_with_paper(students):
    rows = []
    for table, label, x, y, r0, p0 in PAPER:
        r, p, n = pearson(students, x, y)
        rows.append({"table": table, "paper_row": label, "feature": x, "outcome": y, "paper_r": r0, "paper_p": p0,
                     "our_r": round(r, 3), "our_p": round(p, 4), "n": n,
                     "same_direction": bool(np.sign(r) == np.sign(r0)) if np.isfinite(r) else None,
                     "our_p_le_0.05": bool(p <= 0.05) if np.isfinite(p) else None})
    return pd.DataFrame(rows)


def all_pairs(students):
    outcomes = [c for c in students.columns if c.endswith(("_pre", "_post"))] + ["gpa_spring", "gpa_overall"]
    feats = SENSING + ["activity_min_sd", "pa", "stress", "sleep_hours_sr"]
    rows = []
    for x in feats:
        for y in outcomes:
            r, p, n = pearson(students, x, y)
            rows.append({"feature": x, "outcome": y, "r": r, "p": p, "n": n})
    out = pd.DataFrame(rows).dropna(subset=["p"])
    out["p_fdr"] = bh_fdr(out["p"])
    return out.sort_values("p")


def table4(students):
    rows = []
    for k, (n0, m0, s0, n1, m1, s1) in TABLE4.items():
        for when, (n, m, s) in (("pre", (n0, m0, s0)), ("post", (n1, m1, s1))):
            v = students[f"{k}_{when}"].dropna()
            rows.append({"survey": k, "when": when, "paper_n": n, "paper_mean": m, "paper_sd": s,
                         "our_n": len(v), "our_mean": round(v.mean(), 1), "our_sd": round(v.std(), 1)})
    return pd.DataFrame(rows)


def lifecycle(daily):
    """Figure 5(a,b): term trends, averaged over students per term day, scaled to [0, 1], with the
    paper's polynomial smoothing."""
    cols = {"pa": "positive affect (PAM)", "stress": "stress (EMA)", "sleep_h": "sleep duration (inferred)",
            "activity_min": "activity duration", "conv_min": "conversation duration", "conv_freq": "conversation frequency"}
    valid = daily[daily.valid_day | daily[["pa", "stress"]].notna().any(axis=1)]
    by_day = valid.groupby("term_day")[list(cols)].mean()
    by_week = valid.groupby("week")[list(cols)].mean().round(2)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharex=True)
    for ax, keys in zip(axes, (["pa", "stress", "sleep_h"], ["activity_min", "conv_min", "conv_freq"])):
        for k in keys:
            s = by_day[k].dropna()
            z = (s - s.min()) / (s.max() - s.min())
            ax.plot(s.index, z, alpha=0.25)
            fit = np.polyval(np.polyfit(s.index, z, 4), s.index)
            ax.plot(s.index, fit, lw=2.2, label=cols[k], color=ax.lines[-1].get_color())
        for d in (22, 36):
            ax.axvline(d, color="grey", ls=":", lw=1)
        ax.set_xlabel("term day (dotted: midterm period, days 22-36 in the paper)")
        ax.set_ylabel("scaled to 0-1 (min-max per series)")
        ax.legend(fontsize=8, loc="lower left")
    axes[0].set_title("EMA and sleep over the term")
    axes[1].set_title("Automatic sensing over the term")
    fig.suptitle("Reproduction of the 'Dartmouth term lifecycle' (paper Figure 5a-b)")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig1_term_lifecycle.png", dpi=150)
    plt.close(fig)
    return by_week


def sensitivity(daily, students):
    """Robustness checks (ours, not in the paper):
    1. the sleep rows with SELF-REPORTED sleep instead of the phone estimate;
    2. Table 8's positive-affect rows with other reasonable definitions of 'positive affect'."""
    rows = []
    for label, x, y, r0 in (("sleep (self-report) ~ PHQ-9 pre", "sleep_hours_sr", "phq9_pre", -0.360),
                            ("sleep (self-report) ~ PHQ-9 post", "sleep_hours_sr", "phq9_post", -0.382),
                            ("sleep (self-report) ~ PSS pre", "sleep_hours_sr", "pss_pre", -0.355)):
        r, p, n = pearson(students, x, y)
        rows.append({"check": "self-reported sleep", "row": label, "paper_r": r0, "our_r": round(r, 3),
                     "our_p": round(p, 4), "n": n})

    def pa_of(df, col):
        d = df.dropna(subset=[col])
        return ((d[col] * d.n_pam).groupby(d.uid).sum() / d.groupby("uid").n_pam.sum()).rename("x")
    variants = {"PA, whole term (primary)": pa_of(daily, "pa"), "valence, whole term": pa_of(daily, "valence"),
                "PA, weeks 1-2": pa_of(daily[daily.week <= 2], "pa"), "PA, weeks 9-10": pa_of(daily[daily.week >= 9], "pa"),
                "PA, unweighted mean of days": daily.groupby("uid").pa.mean().rename("x")}
    for name, x in variants.items():
        d = students.merge(x.reset_index(), on="uid")
        for y, r0 in (("flourishing_pre", 0.470), ("loneliness_post", -0.390), ("pss_pre", -0.387), ("pss_post", -0.373)):
            r, p, n = pearson(d, "x", y)
            rows.append({"check": f"positive affect definition: {name}", "row": f"{y} ~ positive affect",
                         "paper_r": r0, "our_r": round(r, 3), "our_p": round(p, 4), "n": n})
    return pd.DataFrame(rows)


def plot_comparison(comp):
    d = comp.dropna(subset=["our_r"])
    fig, ax = plt.subplots(figsize=(6.4, 6))
    colors = {"3": "#2a78d6", "5": "#1baf7a", "6": "#eb6834", "7": "#8a63d2", "8": "#e87ba4", "9": "#7d8a3a"}
    for t, g in d.groupby("table"):
        ax.scatter(g.paper_r, g.our_r, s=40, color=colors[t], label=f"Table {t}", alpha=0.85)
    lim = [-0.75, 0.75]
    ax.plot(lim, lim, color="grey", lw=1, ls="--")
    ax.axhline(0, color="#ccc", lw=0.8)
    ax.axvline(0, color="#ccc", lw=0.8)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("r reported in the paper")
    ax.set_ylabel("r in this reproduction")
    ax.set_title("Paper vs reproduction: every reported correlation\n(dashed line = identical)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig2_paper_vs_reproduction_r.png", dpi=150)
    plt.close(fig)


def plot_sleep(students, sleep_report, check):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    ax = axes[0]
    ax.scatter(check.sleep_hours_sr, check.sleep_pred_cv, s=8, alpha=0.35)
    ax.plot([2, 12], [2, 12], color="grey", ls="--", lw=1)
    ax.set_xlabel("self-reported hours slept")
    ax.set_ylabel("phone estimate (students not used to fit)")
    ax.set_title(f"Phone sleep estimate vs self-report\nMAE {sleep_report['mae_min']:.0f} min, r = {sleep_report['r']:.2f} "
                 f"(paper: +/-{SLEEP_ACCURACY_MIN} min vs Jawbone)", fontsize=10)
    for ax, x, name in ((axes[1], "sleep_h", "phone-estimated sleep"), (axes[2], "sleep_hours_sr", "self-reported sleep")):
        d = students[[x, "phq9_pre"]].dropna()
        r, p = stats.pearsonr(d[x], d["phq9_pre"])
        ax.scatter(d[x], d["phq9_pre"], s=30)
        b = np.polyfit(d[x], d["phq9_pre"], 1)
        xs = np.linspace(d[x].min(), d[x].max(), 10)
        ax.plot(xs, np.polyval(b, xs), color="#e34948")
        ax.set_title(f"Depression (PHQ-9, pre) vs {name}\nr = {r:.2f}, p = {p:.3f}, n = {len(d)}   (paper: r = -0.36)",
                     fontsize=10)
        ax.set_xlabel(f"student's mean {name} over the term (h)")
        ax.set_ylabel("PHQ-9 (pre)")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig3_sleep_model_and_depression.png", dpi=150)
    plt.close(fig)


def main():
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    built = build_daily()
    daily, sleep_report = built["daily"], built["sleep_model"]
    students = student_level(daily)
    students.to_csv(RESULTS / "students_term_level.csv", index=False)

    t4 = table4(students)
    t4.to_csv(RESULTS / "table4_surveys.csv", index=False)
    comp = compare_with_paper(students)
    comp.to_csv(RESULTS / "reproduction_vs_paper.csv", index=False)
    pairs = all_pairs(students)
    pairs.to_csv(RESULTS / "all_correlations_fdr.csv", index=False)
    sens = sensitivity(daily, students)
    sens.to_csv(RESULTS / "sensitivity_checks.csv", index=False)
    weeks = lifecycle(daily)
    weeks.to_csv(RESULTS / "term_lifecycle_by_week.csv")
    plot_comparison(comp)
    from features import fit_sleep_model, sleep_nights, windows
    _, _, check = fit_sleep_model(sleep_nights(windows()), daily[["uid", "date", "sleep_hours_sr"]].dropna())
    plot_sleep(students, sleep_report, check)

    ok = comp.dropna(subset=["our_r"])
    summary = {
        "students_with_any_data": int(daily.uid.nunique()),
        "student_days_in_term": int(len(daily)), "valid_days": int(daily.valid_day.sum()),
        "sleep_model": sleep_report,
        "paper_correlations": len(comp), "computable": len(ok),
        "same_direction": int(ok.same_direction.sum()),
        "same_direction_and_p_le_0.05": int((ok.same_direction & ok["our_p_le_0.05"]).sum()),
        "median_abs_r_paper": float(ok.paper_r.abs().median()), "median_abs_r_ours": float(ok.our_r.abs().median()),
        "r_agreement_paper_vs_ours": float(np.corrcoef(ok.paper_r, ok.our_r)[0, 1]),
        "pairs_tested_by_us": len(pairs), "pairs_p_le_0.05": int((pairs.p <= 0.05).sum()),
        "pairs_fdr_le_0.05": int((pairs.p_fdr <= 0.05).sum()),
    }
    (RESULTS / "reproduction_summary.json").write_text(json.dumps(summary, indent=1))
    print(t4.to_string(index=False))
    print(comp[["table", "paper_row", "paper_r", "paper_p", "our_r", "our_p", "n", "same_direction"]].to_string(index=False))
    print(sens.to_string(index=False))
    print(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    main()
