"""Exploratory data analysis: what data exists, how much is missing, and how the daily measures relate."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import RESULTS, FIGURES
from features import build_daily

SHOW = {"pa": "positive affect (PAM)", "valence": "valence", "arousal": "arousal", "stress": "stress EMA",
        "sleep_h": "sleep (inferred)", "sleep_hours_sr": "sleep (self-report)", "bedtime_h": "bedtime",
        "activity_min": "activity", "conv_min": "conversation min", "conv_freq": "conversations",
        "colocations": "co-locations", "distance_km": "distance", "dark_min": "phone dark",
        "dark_morning_min": "phone dark 6-10am", "deadlines": "deadlines"}


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    daily = build_daily()["daily"]
    v = daily[daily.valid_day]
    info = {"students": int(daily.uid.nunique()), "student_days": int(len(daily)),
            "valid_sensing_days": int(daily.valid_day.sum()),
            "days_with_pam": int(daily.pa.notna().sum()), "pam_answers": int(daily.n_pam.sum()),
            "days_with_stress": int(daily.stress.notna().sum()), "nights_with_inferred_sleep": int(daily.sleep_h.notna().sum()),
            "nights_with_self_report": int(daily.sleep_hours_sr.notna().sum())}
    miss = daily[list(SHOW)].isna().mean().round(3).rename("share_missing")
    desc = v[list(SHOW)].describe().T.round(2)
    pd.concat([desc, miss], axis=1).to_csv(RESULTS / "eda_daily_summary.csv")
    (RESULTS / "eda_counts.json").write_text(json.dumps(info, indent=1))

    # 1. data coverage per student over the term
    cov = daily.pivot_table(index="uid", columns="term_day", values="coverage_windows", aggfunc="sum").fillna(0) / 144
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(cov.clip(0, 1), cmap="Blues", cbar_kws={"label": "share of the day sensed"}, ax=ax)
    ax.set_title("Sensing coverage per student and term day (white = no data)")
    fig.tight_layout()
    fig.savefig(FIGURES / "eda1_coverage.png", dpi=130)
    plt.close(fig)

    # 2. distributions
    cols = ["pa", "stress", "sleep_h", "sleep_hours_sr", "activity_min", "conv_min", "colocations", "dark_morning_min"]
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.5))
    for ax, c in zip(axes.ravel(), cols):
        d = (daily if c in ("pa", "stress", "sleep_h", "sleep_hours_sr") else v)[c].dropna()
        ax.hist(d, bins=30, color="#2a78d6")
        ax.set_title(f"{SHOW[c]} (n={len(d)})", fontsize=10)
    fig.suptitle("Daily distributions")
    fig.tight_layout()
    fig.savefig(FIGURES / "eda2_distributions.png", dpi=130)
    plt.close(fig)

    # 3. correlations between daily measures (pooled across students - mixes between- and within-person)
    corr = daily[list(SHOW)].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(10, 8.5))
    sns.heatmap(corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1, annot=True, fmt=".2f", annot_kws={"size": 7},
                xticklabels=[SHOW[c] for c in corr.columns], yticklabels=[SHOW[c] for c in corr.index], ax=ax)
    ax.set_title("Spearman correlations between daily measures (same day, all students pooled)")
    fig.tight_layout()
    fig.savefig(FIGURES / "eda3_daily_correlations.png", dpi=130)
    plt.close(fig)

    # 4. people differ: mood level and spread per student
    order = daily.groupby("uid")["pa"].mean().sort_values().index
    fig, ax = plt.subplots(figsize=(13, 4.5))
    sns.boxplot(data=daily.dropna(subset=["pa"]), x="uid", y="pa", order=order, color="#9ec5f8", fliersize=2, ax=ax)
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.set_ylabel("daily mean PAM positive affect")
    ax.set_title("Each student has their own mood level - why person-level baselines matter")
    fig.tight_layout()
    fig.savefig(FIGURES / "eda4_mood_by_student.png", dpi=130)
    plt.close(fig)
    icc = daily.groupby("uid")["pa"].mean().var() / daily["pa"].var()
    info["pa_share_of_variance_between_students"] = round(float(icc), 3)
    (RESULTS / "eda_counts.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))
    print(pd.concat([desc, miss], axis=1).to_string())


if __name__ == "__main__":
    main()
